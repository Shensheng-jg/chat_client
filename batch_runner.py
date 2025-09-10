import os, re, csv, json, time, argparse, io, base64
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# =============== 基础配置 ===============
load_dotenv()
DEF_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai-proxy.org/v1")
DEF_API_KEY  = os.getenv("LLM_API_KEY", "")
DEF_MODEL    = os.getenv("LLM_MODEL", "gemini-1.5-pro")
DEF_TEMP     = float(os.getenv("LLM_TEMPERATURE", "0.7"))

TPL_PATH = "templates.json"

# =============== 工具函数 ===============
def normalize_base_url(url: str) -> str:
    if not url:
        return url
    u = url.rstrip("/")
    if re.search(r"/v\d+([a-zA-Z].*)?$", u):  # /v1 /v1beta /v2...
        return u
    return u + "/v1"

def make_client(api_key: str, base_url: str) -> OpenAI:
    if not api_key:
        raise ValueError("必须提供 API Key（环境变量 LLM_API_KEY 或 --api-key）")
    if not base_url:
        raise ValueError("必须提供 Base URL")
    return OpenAI(api_key=api_key, base_url=normalize_base_url(base_url))

def load_templates(path: str) -> Dict[str, Dict]:
    if os.path.exists(path):
        # 用 utf-8-sig 兼容带 BOM 的文件
        with open(path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    return {}

class SafeDict(dict):
    def __missing__(self, k):  # 未提供的变量保留原样
        return "{" + k + "}"

def render_template(tpl_text: str, vars_dict: Dict[str, str]) -> str:
    return tpl_text.format_map(SafeDict(vars_dict or {}))

def read_rows(path: str) -> Iterable[Dict[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield {k: (v if v is not None else "") for k, v in row.items()}
    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield {k: str(v) if v is not None else "" for k, v in obj.items()}
    else:
        raise ValueError("仅支持 .csv 或 .jsonl 输入")

def split_multi(s: str) -> List[str]:
    if not s:
        return []
    # 逗号/分号/换行分隔
    parts = re.split(r"[,\n;]+", s)
    return [p.strip() for p in parts if p.strip()]

# =============== 图片处理 ===============
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

def filepath_to_data_url(path: str, max_side=1600, quality=85) -> Optional[str]:
    if not PIL_OK or not path or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as im:
            w, h = im.size
            if max(w, h) > max_side:
                if w >= h:
                    nw, nh = max_side, int(h * (max_side / w))
                else:
                    nh, nw = max_side, int(w * (max_side / h))
                im = im.resize((nw, nh))
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None

def build_user_content(text: str, image_urls: List[str], image_paths: List[str]):
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    for u in image_urls:
        parts.append({"type": "image_url", "image_url": {"url": u, "detail": "high"}})
    for p in image_paths:
        du = filepath_to_data_url(p)
        if du:
            parts.append({"type": "image_url", "image_url": {"url": du, "detail": "high"}})
    return parts

# =============== 调用与兜底 ===============
def call_llm(client: OpenAI, model: str, system_prompt: str, user_content, temperature: float):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    # 主通道：chat.completions（非流式，批处理更稳）
    try:
        r = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        answer = r.choices[0].message.content
        usage = getattr(r, "usage", None)
        return answer, ("chat", usage)
    except Exception as e1:
        err1 = f"{type(e1).__name__}: {e1}"

    # 兜底：responses（仅文本）
    try:
        def msgs_to_text(msgs):
            lines = []
            for m in msgs:
                role, content = m.get("role"), m.get("content")
                if isinstance(content, list):  # 仅提取文本片
                    texts = []
                    for part in content:
                        if part.get("type") in ("text", "input_text"):
                            texts.append(part.get("text") or part.get("content") or "")
                    content = "\n".join(texts)
                lines.append(f"{role.upper()}: {content}")
            return "\n".join(lines)

        r2 = client.responses.create(
            model=model, input=msgs_to_text(messages), temperature=temperature
        )
        answer = getattr(r2, "output_text", None) or str(r2)
        usage = getattr(r2, "usage", None)
        return answer, ("responses", usage)
    except Exception as e2:
        raise RuntimeError(f"两种接口均失败：\n1) {err1}\n2) {type(e2).__name__}: {e2}")

# =============== 主流程 ===============
def run(args):
    templates = load_templates(TPL_PATH)
    if args.user_template not in templates:
        raise ValueError(f"在 {TPL_PATH} 中未找到模板：{args.user_template}")

    user_tpl_text = templates[args.user_template]["text"]
    sys_prompt = ""
    if args.system_template:
        if args.system_template not in templates:
            raise ValueError(f"在 {TPL_PATH} 中未找到系统模板：{args.system_template}")
        sys_prompt = templates[args.system_template]["text"]
    elif args.system_prompt:
        sys_prompt = args.system_prompt

    client = make_client(api_key=args.api_key or DEF_API_KEY,
                         base_url=args.base_url or DEF_BASE_URL)
    model = args.model or DEF_MODEL
    temperature = args.temperature if args.temperature is not None else DEF_TEMP

    # 输出准备
    out_ext = os.path.splitext(args.output)[1].lower()
    is_csv = out_ext == ".csv"
    is_jsonl = out_ext == ".jsonl"
    if not (is_csv or is_jsonl):
        raise ValueError("输出仅支持 .csv 或 .jsonl")

    # 若 CSV，新建则写表头
    csv_writer = None
    f_out = open(args.output, "a", encoding="utf-8", newline="") if is_csv else open(args.output, "a", encoding="utf-8")

    if is_csv and f_out.tell() == 0:
        csv_writer = csv.DictWriter(f_out, fieldnames=[
            "row_index", "request", "answer", "ok", "channel", "prompt_tokens", "completion_tokens",
            "total_tokens", "elapsed_s", "model", "error"
        ])
        csv_writer.writeheader()

    processed = 0
    for i, row in enumerate(read_rows(args.input), start=1):
        # 1) 渲染模板（变量名=列名）
        try:
            user_text = render_template(user_tpl_text, row)
        except Exception as e:
            err = f"模板渲染失败: {e}"
            _write_line(f_out, csv_writer, is_csv, i, request="", answer="", ok=False,
                        channel="", usage=None, elapsed=0.0, model=model, error=err)
            continue

        # 2) 图片列（可选）
        img_urls = split_multi(row.get(args.image_urls_col, "")) if args.image_urls_col else []
        img_paths = split_multi(row.get(args.image_paths_col, "")) if args.image_paths_col else []
        user_content = build_user_content(user_text, img_urls, img_paths)

        # 3) 调用
        t0 = time.time()
        ok, answer, channel, usage, err = True, "", "", None, ""
        try:
            answer, (channel, usage) = call_llm(client, model, sys_prompt, user_content, temperature)
        except Exception as e:
            ok, err = False, str(e)
        elapsed = time.time() - t0

        # 4) 记录
        _write_line(f_out, csv_writer, is_csv, i, request=user_text, answer=answer, ok=ok,
                    channel=channel, usage=usage, elapsed=elapsed, model=model, error=err)

        processed += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    f_out.close()
    print(f"完成：{processed} 条。输出 → {args.output}")

def _write_line(f_out, csv_writer, is_csv, row_idx, request, answer, ok,
                channel, usage, elapsed, model, error):
    prompt_t, compl_t, total_t = None, None, None
    if usage:
        # openai==1.x 的 usage 可能有 .prompt_tokens 等属性；尽量兼容
        prompt_t = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        compl_t  = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        total_t  = getattr(usage, "total_tokens", None)
    rec = {
        "row_index": row_idx,
        "request": request,
        "answer": answer,
        "ok": ok,
        "channel": channel,
        "prompt_tokens": prompt_t,
        "completion_tokens": compl_t,
        "total_tokens": total_t,
        "elapsed_s": round(elapsed, 3),
        "model": model,
        "error": error
    }
    if is_csv:
        (csv_writer or csv.DictWriter(f_out, fieldnames=list(rec.keys()))).writerow(rec)
    else:
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

# =============== CLI ===============
def build_args():
    ap = argparse.ArgumentParser(
        description="批量调用：同一模板 + 多行变量，输出问答结果到 CSV/JSONL"
    )
    ap.add_argument("--input", required=True, help="输入 .csv 或 .jsonl")
    ap.add_argument("--output", required=True, help="输出 .csv 或 .jsonl")
    ap.add_argument("--user-template", required=True, help="templates.json 中的模板名（用于 Message）")
    ap.add_argument("--system-template", help="templates.json 中的系统模板名（用于 System Prompt）")
    ap.add_argument("--system-prompt", help="若不指定 system-template，可直接给一段 System Prompt")
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--base-url", default=DEF_BASE_URL)
    ap.add_argument("--api-key", default=DEF_API_KEY)
    ap.add_argument("--temperature", type=float, default=DEF_TEMP)
    ap.add_argument("--image-urls-col", help="图片URL所在列名（可多值，逗号/分号/换行分隔）")
    ap.add_argument("--image-paths-col", help="本地图片路径所在列名（可多值）")
    ap.add_argument("--sleep", type=float, default=0.4, help="每条之间的间隔秒（防限频）")
    return ap.parse_args()

if __name__ == "__main__":
    run(build_args())
