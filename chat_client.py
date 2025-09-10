import os
import re
import io
import json
import base64
from typing import List, Optional
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ===== 默认配置（可用 .env 覆盖）=====
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai-proxy.org/v1")
DEFAULT_API_KEY  = os.getenv("LLM_API_KEY", "")
DEFAULT_MODEL    = os.getenv("LLM_MODEL", "gemini-1.5-pro")
DEFAULT_SYSTEM   = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful AI assistant.")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ===== 提示词模板（可持久化到 templates.json）=====
TEMPLATES = {
    "系统-严谨中文回答": {
        "scope": "system",
        "text": (
            "你是专业助理，默认使用中文，给出简洁、结构化回答。"
            "必要时列清单，涉及代码需自检可运行。"
        )
    },
    "代码讲解": {
        "scope": "user",
        "text": (
            "请逐步解释这段代码的功能、复杂度与潜在 bug，并给出更优写法：\n"
            "```{lang}\n{code}\n```"
        )
    },
    "写作-改写润色": {
        "scope": "user",
        "text": (
            "把下面文本改写为更{tone}的风格，保持含义不变，并输出三个版本：\n{content}"
        )
    },
}
_TPL_PATH = "templates.json"

def _load_templates():
    global TEMPLATES
    if os.path.exists(_TPL_PATH):
        try:
            with open(_TPL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    TEMPLATES.update(data)
        except Exception:
            pass

def _save_templates():
    try:
        with open(_TPL_PATH, "w", encoding="utf-8") as f:
            json.dump(TEMPLATES, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_load_templates()

# ===== Base URL 规范化 / 客户端工厂 =====
def _normalize_base_url(url: str) -> str:
    """若末尾不是 /v{数字} 或 /v{数字}/ ，则自动补 /v1。"""
    if not url:
        return url
    u = url.rstrip("/")
    # 已经是 /v1、/v1beta、/v2 等版本路径就不再追加
    if re.search(r"/v\d+([a-zA-Z].*)?$", u):
        return u
    return u + "/v1"

def _make_client(api_key: str, base_url: str) -> OpenAI:
    if not api_key:
        raise ValueError("请填写 API Key")
    if not base_url:
        raise ValueError("请填写 Base URL")
    nb = _normalize_base_url(base_url)
    return OpenAI(api_key=api_key, base_url=nb)

# ===== Chatbot 历史：list[[user, assistant], ...] =====
def _ensure_pairs(hist):
    """把历史修正为 list[[user, assistant], ...]，并把 None/非字符串转为字符串。"""
    fixed = []
    for m in hist or []:
        if isinstance(m, tuple):
            m = list(m)
        if not (isinstance(m, list) and len(m) == 2):
            continue
        u, a = m
        u = "" if u is None else str(u)
        a = "" if a is None else str(a)
        fixed.append([u, a])
    return fixed

# ===== 图片处理（压缩为 data URL 以便 image_url 传入）=====
def _compress_to_data_url(img: Image.Image, fmt="JPEG", max_side=1600, quality=85) -> str:
    # 缩放到较小边界，降低 token / 费用
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            nw, nh = max_side, int(h * (max_side / w))
        else:
            nh, nw = max_side, int(w * (max_side / h))
        img = img.resize((nw, nh))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, fmt, quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"

def _filepath_to_data_url(path: str) -> Optional[str]:
    try:
        with Image.open(path) as im:
            return _compress_to_data_url(im, fmt="JPEG")
    except Exception:
        return None

# ===== messages 组装 =====
def _messages_for_chat(system_prompt, history_pairs, user_parts):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for u, a in history_pairs:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_parts})
    return msgs

def _msgs_to_text_for_fallback(msgs):
    """responses 兜底时，仅保文本内容"""
    lines = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                if part.get("type") in ("text", "input_text"):
                    texts.append(part.get("text") or part.get("content") or "")
            content = "\n".join(texts)
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)

# ===== 提示词模板：变量解析与渲染 =====
def _parse_kv_lines(kv_text: str) -> dict:
    """
    key=value 每行一对，允许 value 含 '='（仅分割第一个 '=').
    例：
      lang=python
      tone=学术
      code=print('hello')
    """
    vars = {}
    if not kv_text:
        return vars
    for line in kv_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            vars[k.strip()] = v.strip()
    return vars

def _render_template(name: str, kv_text: str) -> str:
    """用 {变量} 进行安全替换；缺失变量会原样保留。"""
    tpl = TEMPLATES.get(name, {}).get("text", "")
    class _SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return tpl.format_map(_SafeDict(_parse_kv_lines(kv_text)))

# ===== 构建本轮用户输入（文本 + 图片）=====
def _build_user_content(user_text: str,
                        file_paths: Optional[List[str]],
                        url_text: Optional[str]):
    """
    返回 messages 中 user 的 content（list[parts]），以及用于可视化的占位说明。
    """
    parts = []
    if user_text:
        parts.append({"type": "text", "text": user_text})

    # URL（多行，每行一个）
    url_list = []
    if url_text:
        for line in url_text.splitlines():
            u = line.strip()
            if u:
                url_list.append(u)

    for u in url_list:
        parts.append({"type": "image_url", "image_url": {"url": u, "detail": "high"}})

    # 本地文件 → data URL
    count_files = 0
    if file_paths:
        for p in file_paths:
            data_url = _filepath_to_data_url(p)
            if data_url:
                parts.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
                count_files += 1

    # 可视化占位文案（不把 base64 塞进聊天窗口，避免 UI 过大）
    attach_note = ""
    total_imgs = len(url_list) + count_files
    if total_imgs > 0:
        attach_note = f"\n\n🖼️ 已附加 {total_imgs} 张图片（{count_files} 本地 / {len(url_list)} URL）"

    visible_user_text = (user_text or "").strip()
    visible_user_text = visible_user_text + attach_note

    return parts, visible_user_text

# ===== 核心：流式 + 兜底 =====
def stream_chat(api_key, base_url, model, system_prompt, temperature,
                history, user_msg, image_files, image_urls):
    """
    生成器：每次 yield 返回 (chat_history, state) 两个输出。
    history: list[[user, assistant], ...]
    image_files: list[str] (gr.Files 返回文件路径列表或带 name/path 的对象)
    image_urls: str (多行，每行一个 URL)
    """
    history = _ensure_pairs(history)
    client = _make_client(api_key, base_url)

    # 收集文件路径
    file_paths = []
    if image_files:
        for f in image_files:
            if isinstance(f, str):
                file_paths.append(f)
            else:
                path = getattr(f, "name", None) or getattr(f, "path", None)
                if path:
                    file_paths.append(path)

    # 构造本轮用户消息（文本 + 图片）
    user_parts, visible_user = _build_user_content(user_msg, file_paths, image_urls)
    if len(user_parts) == 0:
        history.append(["（空消息）", "⚠️ 请输入文本或附加图片/URL"])
        yield _ensure_pairs(history), _ensure_pairs(history)
        return

    # 组装 messages
    messages = _messages_for_chat(system_prompt, history, user_parts)

    # 先在 UI 占位
    history.append([visible_user, ""])
    yield _ensure_pairs(history), _ensure_pairs(history)

    # 首选：chat.completions 流式
    partial = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                partial += delta.content
                history[-1][1] = partial
                yield _ensure_pairs(history), _ensure_pairs(history)
        return
    except Exception as e1:
        err1 = f"{type(e1).__name__}: {e1}"

    # 兜底：responses（多数代理对图片不一定兼容，此处仅保文本）
    try:
        r = client.responses.create(
            model=model,
            input=_msgs_to_text_for_fallback(messages),
            temperature=temperature,
        )
        reply = getattr(r, "output_text", None) or str(r)
        history[-1][1] = reply
        yield _ensure_pairs(history), _ensure_pairs(history)
    except Exception as e2:
        history[-1][1] = f"⚠️ 请求出错（已尝试 chat.completions 与 responses）：\n1) {err1}\n2) {type(e2).__name__}: {e2}"
        yield _ensure_pairs(history), _ensure_pairs(history)

# ===== 清空 =====
def clear_all():
    empty = []
    return empty, empty

# ===== 模板按钮回调 =====
def _apply_to_system(name, kv, cur_text):
    if not name:
        return gr.update(value=cur_text)
    return gr.update(value=_render_template(name, kv))

def _apply_to_msg(name, kv, cur_text):
    if not name:
        return gr.update(value=cur_text)
    new_text = (cur_text or "").rstrip()
    rendered = _render_template(name, kv)
    if new_text:
        new_text = new_text + "\n\n" + rendered
    else:
        new_text = rendered
    return gr.update(value=new_text)

def _save_from_text(name, scope, text):
    if not name or not text:
        return gr.update(choices=sorted(TEMPLATES.keys()))
    TEMPLATES[name] = {"scope": scope, "text": text}
    _save_templates()
    return gr.update(choices=sorted(TEMPLATES.keys()), value=name)

def _delete_tpl(name):
    if name in TEMPLATES:
        del TEMPLATES[name]
        _save_templates()
    # 同时更新两个下拉框（若你用了两个）
    upd = gr.update(choices=sorted(TEMPLATES.keys()), value=None)
    return upd, upd

def _rename_tpl(old_name, new_name):
    if old_name and new_name and old_name in TEMPLATES:
        TEMPLATES[new_name] = TEMPLATES.pop(old_name)
        _save_templates()
    upd = gr.update(choices=sorted(TEMPLATES.keys()), value=new_name or old_name)
    return upd, upd

# ===== Gradio UI =====
with gr.Blocks(title="LLM Client (OpenAI-Compatible)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 LLM Client · OpenAI-Compatible（文本 + 图片 + 模板）")

    with gr.Row():
        base_url = gr.Textbox(label="Base URL", value=DEFAULT_BASE_URL, placeholder="如：https://api.openai-proxy.org/v1")
        api_key  = gr.Textbox(label="API Key", value=DEFAULT_API_KEY, type="password", placeholder="你的密钥")

    with gr.Row():
        model = gr.Textbox(label="Model", value=DEFAULT_MODEL, placeholder="如：gemini-1.5-pro / gpt-4o-mini")
        temperature = gr.Slider(0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.1, label="Temperature")

    system_prompt = gr.Textbox(label="System Prompt（可选）", value=DEFAULT_SYSTEM, lines=2)

    # —— 提示词模板区 ——
    with gr.Accordion("提示词模板（可选）", open=False):
        tpl_name = gr.Dropdown(
            label="选择模板",
            choices=sorted(TEMPLATES.keys()),
            allow_custom_value=False
        )
        tpl_vars = gr.Textbox(
            label="模板变量（key=value，每行一对）",
            placeholder="lang=python\ntone=学术\ncode=print('hello')\ncontent=待改写文本...",
            lines=5
        )
        with gr.Row():
            btn_apply_to_system = gr.Button("应用到 System Prompt")
            btn_apply_to_msg    = gr.Button("应用到 Message")

        with gr.Row():
            tpl_new_name = gr.Textbox(label="保存为模板的名称")
            tpl_scope = gr.Radio(["system", "user"], value="user", label="模板类型")
            btn_save_tpl_from_msg = gr.Button("从 Message 保存")
            btn_save_tpl_from_sys = gr.Button("从 System 保存")

        with gr.Row():
            tpl_sel_manage = gr.Dropdown(label="选择模板（删除/重命名）",
                                        choices=sorted(TEMPLATES.keys()))
            tpl_new_name2 = gr.Textbox(label="新名称（用于重命名）")

        with gr.Row():
            btn_delete_tpl = gr.Button("删除所选模板", variant="stop")
            btn_rename_tpl = gr.Button("重命名模板")

        # 绑定事件（同时更新 管理下拉框 与 选择模板下拉框）
        btn_delete_tpl.click(_delete_tpl, inputs=[tpl_sel_manage], outputs=[tpl_name, tpl_sel_manage])
        btn_rename_tpl.click(_rename_tpl, inputs=[tpl_sel_manage, tpl_new_name2], outputs=[tpl_name, tpl_sel_manage])

    # —— 图片输入（本地与 URL） ——
    with gr.Accordion("附加图片（可选）", open=False):
        image_files = gr.Files(label="上传图片（可多选）", file_types=["image"], file_count="multiple")
        image_urls  = gr.Textbox(label="图片 URL（每行一个，可多行）", lines=3, placeholder="https://.../a.jpg\nhttps://.../b.png")

    chat = gr.Chatbot(height=480, avatar_images=(None, None))
    msg = gr.Textbox(placeholder="输入文本，可选附加图片或图片URL", label="Message")
    with gr.Row():
        send_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话", variant="secondary")

    state = gr.State([])

    # 发送 / 回车提交
    inputs = [api_key, base_url, model, system_prompt, temperature, state, msg, image_files, image_urls]
    outputs = [chat, state]
    send_btn.click(stream_chat, inputs=inputs, outputs=outputs)
    msg.submit(stream_chat, inputs=inputs, outputs=outputs)

    # 清空
    clear_btn.click(clear_all, None, outputs)

    # 模板按钮绑定
    btn_apply_to_system.click(
        _apply_to_system,
        inputs=[tpl_name, tpl_vars, system_prompt],
        outputs=[system_prompt]
    )
    btn_apply_to_msg.click(
        _apply_to_msg,
        inputs=[tpl_name, tpl_vars, msg],
        outputs=[msg]
    )
    btn_save_tpl_from_msg.click(
        _save_from_text,
        inputs=[tpl_new_name, tpl_scope, msg],
        outputs=[tpl_name]
    )
    btn_save_tpl_from_sys.click(
        _save_from_text,
        inputs=[tpl_new_name, tpl_scope, system_prompt],
        outputs=[tpl_name]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
