import os
import re
import io
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

# ===== 工具函数 =====
def _normalize_base_url(url: str) -> str:
    if not url:
        return url
    u = url.rstrip("/")
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

def _ensure_pairs(hist):
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
    image_files: list[str] (gr.Files 返回文件路径列表)
    image_urls: str (多行，每行一个 URL)
    """
    history = _ensure_pairs(history)
    client = _make_client(api_key, base_url)

    # 准备 user 消息的 content（文字 + 图片）
    file_paths = []
    if image_files:
        # gr.Files 在 Gradio v4 中传入的是含有 .name 属性的对象或纯路径；都按路径收集
        for f in image_files:
            if isinstance(f, str):
                file_paths.append(f)
            else:
                # TemporaryFile or UploadFile-like
                path = getattr(f, "name", None) or getattr(f, "path", None)
                if path:
                    file_paths.append(path)

    user_parts, visible_user = _build_user_content(user_msg, file_paths, image_urls)

    if len(user_parts) == 0:
        # 既没有文本也没有图片，直接提示
        history.append(["（空消息）", "⚠️ 请输入文本或附加图片/URL"])
        yield _ensure_pairs(history), _ensure_pairs(history)
        return

    # 组装完整 messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for u, a in history:
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_parts})

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
        # 把 messages 压成纯文本做降级
        def _msgs_to_text(msgs):
            lines = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    # 仅拼接文本部分
                    texts = []
                    for part in content:
                        if part.get("type") in ("text", "input_text"):
                            texts.append(part.get("text") or part.get("content") or "")
                    content = "\n".join(texts)
                lines.append(f"{role.upper()}: {content}")
            return "\n".join(lines)

        r = client.responses.create(
            model=model,
            input=_msgs_to_text(messages),
            temperature=temperature,
        )
        reply = getattr(r, "output_text", None) or str(r)
        history[-1][1] = reply
        yield _ensure_pairs(history), _ensure_pairs(history)
    except Exception as e2:
        history[-1][1] = f"⚠️ 请求出错（已尝试 chat.completions 与 responses）：\n1) {err1}\n2) {type(e2).__name__}: {e2}"
        yield _ensure_pairs(history), _ensure_pairs(history)

def clear_all():
    empty = []
    return empty, empty

# ===== Gradio UI =====
with gr.Blocks(title="LLM Client (OpenAI-Compatible)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 LLM Client · OpenAI-Compatible（文本 + 图片）")

    with gr.Row():
        base_url = gr.Textbox(label="Base URL", value=DEFAULT_BASE_URL, placeholder="如：https://api.openai-proxy.org/v1")
        api_key  = gr.Textbox(label="API Key", value=DEFAULT_API_KEY, type="password", placeholder="你的密钥")

    with gr.Row():
        model = gr.Textbox(label="Model", value=DEFAULT_MODEL, placeholder="如：gemini-1.5-pro / gpt-4o-mini")
        temperature = gr.Slider(0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.1, label="Temperature")

    system_prompt = gr.Textbox(label="System Prompt（可选）", value=DEFAULT_SYSTEM, lines=2)

    # 新增：图片输入（本地与 URL）
    with gr.Accordion("附加图片（可选）", open=False):
        image_files = gr.Files(label="上传图片（可多选）", file_types=["image"], file_count="multiple")
        image_urls  = gr.Textbox(label="图片 URL（每行一个，可多行）", lines=3, placeholder="https://.../a.jpg\nhttps://.../b.png")

    chat = gr.Chatbot(height=420, avatar_images=(None, None))
    msg = gr.Textbox(placeholder="输入文本，可选附加图片或图片URL", label="Message")
    with gr.Row():
        send_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话", variant="secondary")

    state = gr.State([])

    inputs = [api_key, base_url, model, system_prompt, temperature, state, msg, image_files, image_urls]
    outputs = [chat, state]

    send_btn.click(stream_chat, inputs=inputs, outputs=outputs)
    msg.submit(stream_chat, inputs=inputs, outputs=outputs)
    clear_btn.click(clear_all, None, outputs)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
