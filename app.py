#!/usr/bin/env python3
"""Local recording bench for two OpenAI-compatible chat models.

The app uses only Python's standard library. It calls a user-selected
OpenAI-compatible Chat Completions endpoint, stores streamed reasoning/final
text with timing, extracts generated HTML, and serves a browser UI for
synchronized replay, previews, and metrics.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RUNS_DIR = BASE_DIR / "runs"
PROMPT_PATH = BASE_DIR / "prompt.txt"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_COMPLETION_TOKENS: int | None = None

SYSTEM_PROMPT = (
    "You are participating in a controlled coding comparison. Produce a single "
    "self-contained HTML document that runs directly in a modern browser. Follow "
    "the user's requirements exactly, but prefer the smallest reliable implementation "
    "over extra features. Prioritize valid executable code and a visibly working "
    "animation. Keep planning brief and move to the final document early enough to "
    "finish it. Do not use external libraries, network requests, remote assets, "
    "external fonts, TypeScript syntax, pseudo-code, placeholder functions, or "
    "markdown fences. Before returning, verify that every queried DOM element exists, "
    "every event target is defined, and the animation initialization and loop are "
    "called. Return only the complete HTML document in the final answer."
)

MODEL_SLOTS = ("a", "b")

RUNS: dict[str, "RunState"] = {}
RUNS_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def round_number(value: float | int | None, digits: int = 4) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return round(value, digits)


def public_model(model: dict[str, Any], include_timeline: bool = True) -> dict[str, Any]:
    result = {
        "slug": model["slug"],
        "id": model["id"],
        "label": model["label"],
        "status": model["status"],
        "reasoning": model["reasoning"],
        "content": model["content"],
        "metrics": model["metrics"],
        "output_url": model.get("output_url"),
        "artifacts": model.get("artifacts", {}),
        "error": model.get("error"),
    }
    if include_timeline:
        result["timeline"] = model["timeline"]
    return result


class RunState:
    def __init__(
        self,
        run_id: str,
        prompt: str,
        base_url: str,
        endpoint_url: str,
        model_ids: dict[str, str],
        max_completion_tokens: int | None,
        order: list[str],
        demo: bool,
    ) -> None:
        self.id = run_id
        self.prompt = prompt
        self.base_url = base_url
        self.endpoint_url = endpoint_url
        self.model_ids = model_ids
        self.max_completion_tokens = max_completion_tokens
        self.order = order
        self.demo = demo
        self.created_at = utc_now()
        self.completed_at: str | None = None
        self.status = "queued"
        self.events: list[dict[str, Any]] = []
        self.closed = False
        self.condition = threading.Condition()
        self.models: dict[str, dict[str, Any]] = {}
        for slug in MODEL_SLOTS:
            model_id = model_ids[slug]
            self.models[slug] = {
                "slug": slug,
                "id": model_id,
                "label": model_id,
                "status": "waiting",
                "reasoning": "",
                "content": "",
                "timeline": [],
                "metrics": {},
                "output_url": None,
                "artifacts": {},
                "error": None,
            }

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.id

    def publish(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, "at": utc_now(), **payload}
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def finish_stream(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def summary(self, include_timeline: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "base_url": self.base_url,
            "endpoint_url": self.endpoint_url,
            "max_completion_tokens": self.max_completion_tokens,
            "order": self.order,
            "demo": self.demo,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "models": {
                slug: public_model(model, include_timeline)
                for slug, model in self.models.items()
            },
        }


def load_default_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "Build a single self-contained HTML simulation. Return only HTML."


def configured_api_key() -> str:
    """Use the generic variable first, retaining the old name as a fallback."""
    return (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("MOONSHOT_API_KEY", "").strip()
    )


def normalize_endpoint_url(base_url: str) -> tuple[str, str]:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("Base URL is required.")
    if len(base_url) > 2_048:
        raise ValueError("Base URL is too long.")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Base URL must be an absolute http:// or https:// URL.")
    endpoint_url = (
        base_url
        if parsed.path.rstrip("/").endswith("/chat/completions")
        else base_url + "/chat/completions"
    )
    normalized_base = (
        base_url[: -len("/chat/completions")]
        if parsed.path.rstrip("/").endswith("/chat/completions")
        else base_url
    ).rstrip("/")
    return normalized_base, endpoint_url


def cached_tokens_from_usage(usage: dict[str, Any]) -> int:
    direct = usage.get("cached_tokens")
    if isinstance(direct, int):
        return direct
    details = usage.get("prompt_tokens_details") or {}
    nested = details.get("cached_tokens")
    return nested if isinstance(nested, int) else 0


def optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def reported_cost_from_usage(usage: dict[str, Any]) -> float | None:
    for key in ("cost", "total_cost", "estimated_cost"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), 8)
    return None


def summarize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = optional_int(usage.get("prompt_tokens"))
    completion_tokens = optional_int(usage.get("completion_tokens"))
    total_tokens = optional_int(usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    cached_tokens = (
        min(cached_tokens_from_usage(usage), prompt_tokens)
        if prompt_tokens is not None
        else None
    )
    uncached_tokens = (
        max(prompt_tokens - (cached_tokens or 0), 0)
        if prompt_tokens is not None
        else None
    )
    reported_cost = reported_cost_from_usage(usage)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "uncached_input_tokens": uncached_tokens,
        "reported_cost_usd": reported_cost,
        "estimated_cost_usd": reported_cost,
        "usage_received": bool(usage),
    }


def extract_html_document(content: str) -> tuple[str, bool]:
    """Extract a complete HTML document and report whether extraction was clean."""
    fenced = re.search(r"```(?:html)?\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else content.strip()
    lower = candidate.lower()
    starts = [pos for pos in (lower.find("<!doctype html"), lower.find("<html")) if pos >= 0]
    if starts:
        candidate = candidate[min(starts) :]
        lower = candidate.lower()
        end = lower.rfind("</html>")
        if end >= 0:
            candidate = candidate[: end + len("</html>")]
        return candidate.strip(), True

    escaped = html.escape(content)
    fallback = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>No HTML extracted</title>
<style>body{{background:#0b0e13;color:#f5f7fa;font:16px/1.5 system-ui;padding:32px}}pre{{white-space:pre-wrap;background:#151a22;padding:20px;border-radius:12px}}</style>
</head><body><h1>No complete HTML document was extracted</h1><pre>{escaped}</pre></body></html>"""
    return fallback, False


class DeltaBuffer:
    """Coalesce token-sized SSE deltas into replay-friendly timed chunks."""

    def __init__(self, state: RunState, slug: str, started: float) -> None:
        self.state = state
        self.slug = slug
        self.started = started
        self.kind: str | None = None
        self.text = ""
        self.last_flush = started

    def add(self, kind: str, text: str) -> None:
        if not text:
            return
        now = time.perf_counter()
        if self.kind is not None and kind != self.kind:
            self.flush(now)
        self.kind = kind
        self.text += text
        if len(self.text) >= 160 or now - self.last_flush >= 0.075:
            self.flush(now)

    def flush(self, now: float | None = None) -> None:
        if not self.text or self.kind is None:
            return
        now = now or time.perf_counter()
        elapsed = round(now - self.started, 3)
        model = self.state.models[self.slug]
        model[self.kind] += self.text
        item = {"t": elapsed, "kind": self.kind, "text": self.text}
        model["timeline"].append(item)
        self.state.publish("delta", model=self.slug, **item)
        self.text = ""
        self.last_flush = now


def parse_usage(chunk: dict[str, Any], choice: dict[str, Any] | None) -> dict[str, Any] | None:
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        return usage
    if choice:
        usage = choice.get("usage")
        if isinstance(usage, dict):
            return usage
    return None


def delta_text(value: Any) -> str:
    """Handle standard string deltas plus common content-part variants."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"]
    return ""


def stream_real_model(state: RunState, slug: str) -> None:
    model = state.models[slug]
    model_dir = state.run_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    model["status"] = "running"
    state.publish("model_started", model=slug, label=model["label"])

    payload: dict[str, Any] = {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state.prompt},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if state.max_completion_tokens is not None:
        # max_tokens is the most widely supported Chat Completions spelling.
        payload["max_tokens"] = state.max_completion_tokens

    (model_dir / "request.json").write_text(
        json.dumps(
            {
                "base_url": state.base_url,
                "endpoint_url": state.endpoint_url,
                "payload": payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    key = configured_api_key()
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "llm-html-arena/1.0",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        state.endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    received_done = False
    buffer = DeltaBuffer(state, slug, started)

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    received_done = True
                    break
                if not data:
                    continue
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                choice = choices[0] if choices else None
                parsed_usage = parse_usage(chunk, choice)
                if parsed_usage:
                    usage = parsed_usage
                if not choice:
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                reasoning = delta_text(
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("thinking")
                )
                content = delta_text(delta.get("content"))
                if (reasoning or content) and first_token_at is None:
                    first_token_at = time.perf_counter()
                buffer.add("reasoning", reasoning)
                buffer.add("content", content)
        buffer.flush()
        if not received_done:
            raise RuntimeError("The stream ended without the final [DONE] marker.")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible API HTTP {exc.code}: {body[:2000]}") from exc

    ended = time.perf_counter()
    total_seconds = ended - started
    ttft_seconds = first_token_at - started if first_token_at else None
    completion_seconds = ended - first_token_at if first_token_at else None
    cost = summarize_usage(usage)
    completion_tokens = cost["completion_tokens"]
    tokens_per_second = (
        completion_tokens / completion_seconds
        if completion_tokens is not None and completion_seconds and completion_seconds > 0
        else None
    )

    output_html, html_extracted = extract_html_document(model["content"])
    (model_dir / "reasoning.txt").write_text(model["reasoning"], encoding="utf-8")
    (model_dir / "content.txt").write_text(model["content"], encoding="utf-8")
    (model_dir / "output.html").write_text(output_html, encoding="utf-8")

    metrics = {
        **cost,
        "total_seconds": round_number(total_seconds, 3),
        "time_to_first_token_seconds": round_number(ttft_seconds, 3),
        "completion_stream_seconds": round_number(completion_seconds, 3),
        "tokens_per_second": round_number(tokens_per_second, 2),
        "reasoning_characters": len(model["reasoning"]),
        "final_characters": len(model["content"]),
        "finish_reason": finish_reason,
        "received_done": received_done,
        "html_extracted": html_extracted,
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model["metrics"] = metrics
    model["status"] = "complete"
    model["output_url"] = f"/runs/{state.id}/{slug}/output.html"
    model["artifacts"] = {
        "reasoning": f"/runs/{state.id}/{slug}/reasoning.txt",
        "content": f"/runs/{state.id}/{slug}/content.txt",
        "metrics": f"/runs/{state.id}/{slug}/metrics.json",
        "request": f"/runs/{state.id}/{slug}/request.json",
        "output": model["output_url"],
    }
    state.publish("model_complete", model=slug, metrics=metrics, output_url=model["output_url"])


def chunk_text(text: str, target: int = 90) -> list[str]:
    words = re.findall(r"\S+\s*", text)
    chunks: list[str] = []
    pending = ""
    for word in words:
        pending += word
        if len(pending) >= target:
            chunks.append(pending)
            pending = ""
    if pending:
        chunks.append(pending)
    return chunks


def demo_html(slug: str) -> str:
    accent = "#9eff6b" if slug == "a" else "#ffb454"
    collision_logic = "0" if slug == "a" else "2"
    label = "Demo Model A" if slug == "a" else "Demo Model B"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gridlock demo — {label}</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#080b10;color:#eef2f7;font-family:system-ui;overflow:hidden}}
header{{height:72px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;background:#111720;border-bottom:1px solid #293442}}
.brand{{font-size:22px;font-weight:800;letter-spacing:.08em;color:{accent}}}.metrics{{display:flex;gap:22px;color:#9aa9ba}} b{{color:#fff}}
canvas{{display:block;width:100vw;height:calc(100vh - 72px)}}
</style></head><body><header><div class="brand">GRIDLOCK / {label}</div><div class="metrics"><span>QUEUE <b id="q">12</b></span><span>COLLISIONS <b>{collision_logic}</b></span><span>PHASE <b id="p">N/S</b></span></div></header>
<canvas id="c"></canvas><script>
const c=document.querySelector('#c'),x=c.getContext('2d'); let cars=[],t=0;
function size(){{c.width=innerWidth*devicePixelRatio;c.height=(innerHeight-72)*devicePixelRatio;x.scale(devicePixelRatio,devicePixelRatio)}} addEventListener('resize',size);size();
for(let i=0;i<28;i++)cars.push({{lane:i%4,p:(i*71)%900,s:1+(i%3)*.25,color:['#71d7ff','{accent}','#ff6b8b','#ddd'][i%4]}});
function draw(){{const w=innerWidth,h=innerHeight-72,road=170,cx=w/2,cy=h/2;t+=.016;x.fillStyle='#081019';x.fillRect(0,0,w,h);x.fillStyle='#1b222c';x.fillRect(0,cy-road/2,w,road);x.fillRect(cx-road/2,0,road,h);x.strokeStyle='#d8be61';x.setLineDash([18,18]);x.lineWidth=2;x.beginPath();x.moveTo(0,cy);x.lineTo(w,cy);x.moveTo(cx,0);x.lineTo(cx,h);x.stroke();x.setLineDash([]);
for(let i=-3;i<=3;i++){{x.fillStyle=i%2?'#e8edf2':'#68788b';x.fillRect(cx-80+i*20,cy-120,12,55);x.fillRect(cx-80+i*20,cy+65,12,55);x.fillRect(cx-120,cy-80+i*20,55,12);x.fillRect(cx+65,cy-80+i*20,55,12)}}
for(const a of cars){{a.p=(a.p+a.s*2)%1000;x.fillStyle=a.color;if(a.lane===0)x.fillRect((a.p/1000)*w,cy-55,26,13);if(a.lane===1)x.fillRect(w-(a.p/1000)*w,cy+40,26,13);if(a.lane===2)x.fillRect(cx+38,(a.p/1000)*h,13,26);if(a.lane===3)x.fillRect(cx-55,h-(a.p/1000)*h,13,26)}}
const green=Math.floor(t/4)%2===0;document.querySelector('#p').textContent=green?'N/S':'E/W';for(const [dx,dy] of [[-105,-105],[86,-105],[-105,86],[86,86]]){{x.fillStyle='#10151c';x.fillRect(cx+dx,cy+dy,19,42);x.fillStyle=green?'{accent}':'#ff4d5f';x.beginPath();x.arc(cx+dx+9,cy+dy+12,5,0,7);x.fill()}}requestAnimationFrame(draw)}}draw();
</script></body></html>"""


def stream_demo_model(state: RunState, slug: str) -> None:
    model = state.models[slug]
    model_dir = state.run_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    model["status"] = "running"
    state.publish("model_started", model=slug, label=model["label"])
    started = time.perf_counter()
    reasoning = (
        f"I will design the intersection state machine, lane paths, traffic-light phases, "
        f"pedestrian requests, and counters before producing the final self-contained HTML for {model['label']}. "
        "The canvas will use a fixed simulation timestep and responsive rendering. "
    ) * 4
    final_html = demo_html(slug)
    buffer = DeltaBuffer(state, slug, started)
    first = None
    for kind, text in (("reasoning", reasoning), ("content", final_html)):
        for part in chunk_text(text):
            if first is None:
                first = time.perf_counter()
            buffer.add(kind, part)
            time.sleep(0.008 if slug == "a" else 0.006)
    buffer.flush()
    ended = time.perf_counter()
    usage = {
        "prompt_tokens": 318,
        "completion_tokens": 8_420 if slug == "a" else 6_130,
        "total_tokens": 8_738 if slug == "a" else 6_448,
        "cached_tokens": 0,
        "cost": 0.127254 if slug == "a" else 0.024822,
    }
    cost = summarize_usage(usage)
    output_html, html_extracted = extract_html_document(model["content"])
    (model_dir / "request.json").write_text(
        json.dumps(
            {
                "demo": True,
                "base_url": state.base_url,
                "endpoint_url": state.endpoint_url,
                "model": model["id"],
                "prompt": state.prompt,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (model_dir / "reasoning.txt").write_text(model["reasoning"], encoding="utf-8")
    (model_dir / "content.txt").write_text(model["content"], encoding="utf-8")
    (model_dir / "output.html").write_text(output_html, encoding="utf-8")
    metrics = {
        **cost,
        "total_seconds": round(ended - started, 3),
        "time_to_first_token_seconds": round((first or started) - started, 3),
        "completion_stream_seconds": round(ended - (first or started), 3),
        "tokens_per_second": round(cost["completion_tokens"] / max(ended - (first or started), 0.01), 2),
        "reasoning_characters": len(model["reasoning"]),
        "final_characters": len(model["content"]),
        "finish_reason": "stop",
        "received_done": True,
        "html_extracted": html_extracted,
        "demo_metrics": True,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model["metrics"] = metrics
    model["status"] = "complete"
    model["output_url"] = f"/runs/{state.id}/{slug}/output.html"
    model["artifacts"] = {
        "reasoning": f"/runs/{state.id}/{slug}/reasoning.txt",
        "content": f"/runs/{state.id}/{slug}/content.txt",
        "metrics": f"/runs/{state.id}/{slug}/metrics.json",
        "request": f"/runs/{state.id}/{slug}/request.json",
        "output": model["output_url"],
    }
    state.publish("model_complete", model=slug, metrics=metrics, output_url=model["output_url"])


def run_pair(state: RunState) -> None:
    state.status = "running"
    state.run_dir.mkdir(parents=True, exist_ok=True)
    state.publish(
        "run_started",
        run_id=state.id,
        order=state.order,
        demo=state.demo,
        base_url=state.base_url,
        models=state.model_ids,
        max_completion_tokens=state.max_completion_tokens,
    )
    try:
        for slug in state.order:
            try:
                if state.demo:
                    stream_demo_model(state, slug)
                else:
                    stream_real_model(state, slug)
            except Exception as exc:  # keep the second model runnable if the first fails
                model = state.models[slug]
                model["status"] = "error"
                model["error"] = str(exc)
                (state.run_dir / slug).mkdir(parents=True, exist_ok=True)
                (state.run_dir / slug / "error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                state.publish("model_error", model=slug, error=str(exc))
        state.status = (
            "complete"
            if all(model["status"] == "complete" for model in state.models.values())
            else "partial"
        )
    except Exception as exc:
        state.status = "error"
        state.publish("run_error", error=str(exc))
    finally:
        state.completed_at = utc_now()
        summary_path = state.run_dir / "run.json"
        summary_path.write_text(
            json.dumps(state.summary(include_timeline=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state.publish(
            "pair_complete",
            run_id=state.id,
            status=state.status,
            summary_url=f"/api/run/{state.id}",
            artifact_url=f"/runs/{state.id}/run.json",
        )
        state.finish_stream()


def new_run(payload: dict[str, Any]) -> RunState:
    prompt = str(payload.get("prompt") or "").strip()
    if len(prompt) < 40:
        raise ValueError("Prompt is too short; provide the complete comparison task.")
    demo = bool(payload.get("demo"))
    base_url, endpoint_url = normalize_endpoint_url(
        str(payload.get("base_url") or DEFAULT_BASE_URL)
    )
    models_value = payload.get("models") or {}
    if not isinstance(models_value, dict):
        raise ValueError("models must be an object with a and b fields.")
    model_ids = {
        "a": str(models_value.get("a") or ("demo/model-a" if demo else "")).strip(),
        "b": str(models_value.get("b") or ("demo/model-b" if demo else "")).strip(),
    }
    for slug, model_id in model_ids.items():
        if not model_id:
            raise ValueError(f"Model {slug.upper()} ID is required.")
        if len(model_id) > 300 or any(char in model_id for char in "\r\n\0"):
            raise ValueError(f"Model {slug.upper()} ID is invalid.")

    raw_max_tokens = payload.get("max_completion_tokens")
    max_tokens = None
    if raw_max_tokens not in (None, "", 0, "0"):
        max_tokens = int(raw_max_tokens)
        if max_tokens < 1 or max_tokens > 1_048_576:
            raise ValueError("max completion tokens must be between 1 and 1,048,576, or empty.")
    order_value = payload.get("order") or ["a", "b"]
    if order_value not in (["a", "b"], ["b", "a"]):
        raise ValueError("Order must contain a and b exactly once.")

    with RUNS_LOCK:
        active = [state for state in RUNS.values() if state.status in {"queued", "running"}]
        if active:
            raise RuntimeError("Another comparison is already running. Wait for it to finish.")
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        state = RunState(
            run_id,
            prompt,
            base_url,
            endpoint_url,
            model_ids,
            max_tokens,
            list(order_value),
            demo,
        )
        RUNS[run_id] = state
    threading.Thread(target=run_pair, args=(state,), daemon=True).start()
    return state


class AppHandler(BaseHTTPRequestHandler):
    server_version = "OpenAICompatibleComparison/2.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        if self.path.startswith("/api/events/"):
            return
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/config":
            self.send_json(
                {
                    "api_key_configured": bool(configured_api_key()),
                    "api_key_source": (
                        "OPENAI_API_KEY"
                        if os.environ.get("OPENAI_API_KEY", "").strip()
                        else "MOONSHOT_API_KEY"
                        if os.environ.get("MOONSHOT_API_KEY", "").strip()
                        else None
                    ),
                    "default_prompt": load_default_prompt(),
                    "default_base_url": DEFAULT_BASE_URL,
                    "default_models": {"a": "", "b": ""},
                    "default_max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
                }
            )
            return
        if path.startswith("/api/events/"):
            self.handle_events(path.rsplit("/", 1)[-1])
            return
        if path.startswith("/api/run/"):
            run_id = path.rsplit("/", 1)[-1]
            state = RUNS.get(run_id)
            if not state:
                self.send_json({"error": "Unknown run id."}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(state.summary(include_timeline=True))
            return
        if path.startswith("/runs/"):
            self.serve_run_artifact(path)
            return
        if path in {"/", "/index.html"}:
            self.serve_static(STATIC_DIR / "index.html")
            return
        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 200_000:
                raise ValueError("Request is too large.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            state = new_run(payload)
            self.send_json(
                {
                    "run_id": state.id,
                    "events_url": f"/api/events/{state.id}",
                    "summary_url": f"/api/run/{state.id}",
                },
                HTTPStatus.ACCEPTED,
            )
        except PermissionError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.UNAUTHORIZED)
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_events(self, run_id: str) -> None:
        state = RUNS.get(run_id)
        if not state:
            self.send_json({"error": "Unknown run id."}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        index = 0
        try:
            while True:
                with state.condition:
                    if index >= len(state.events) and not state.closed:
                        state.condition.wait(timeout=10)
                    events = state.events[index:]
                    index = len(state.events)
                    closed = state.closed
                if not events:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                for event in events:
                    line = "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                if closed and index >= len(state.events):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

    def serve_static(self, path: Path) -> None:
        if not path.is_file():
            self.send_json({"error": "Static file not found."}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_bytes(
            path.read_bytes(),
            content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""),
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "frame-src 'self'; img-src 'self' data:"
                )
            },
        )

    def serve_run_artifact(self, request_path: str) -> None:
        parts = [part for part in request_path.strip("/").split("/") if part]
        if len(parts) < 3 or parts[0] != "runs":
            self.send_json({"error": "Invalid artifact path."}, HTTPStatus.BAD_REQUEST)
            return
        run_id = parts[1]
        if run_id not in RUNS and not (RUNS_DIR / run_id).is_dir():
            self.send_json({"error": "Unknown run id."}, HTTPStatus.NOT_FOUND)
            return
        allowed = {"run.json", "output.html", "reasoning.txt", "content.txt", "metrics.json", "request.json", "error.txt"}
        filename = parts[-1]
        if filename not in allowed or any(part in {"..", "."} for part in parts):
            self.send_json({"error": "Artifact is not allowed."}, HTTPStatus.FORBIDDEN)
            return
        target = RUNS_DIR.joinpath(*parts[1:]).resolve()
        if RUNS_DIR.resolve() not in target.parents:
            self.send_json({"error": "Artifact path escaped the runs directory."}, HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_json({"error": "Artifact not found."}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        headers: dict[str, str] = {}
        if target.name == "output.html":
            headers["Content-Security-Policy"] = (
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "img-src data: blob:; media-src data: blob:; connect-src 'none'; "
                "font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'"
            )
        self.send_bytes(
            target.read_bytes(),
            content_type + ("; charset=utf-8" if content_type.startswith(("text/", "application/json")) else ""),
            headers=headers,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the OpenAI-compatible model comparison recording bench."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="Open the local UI in the default browser.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    server.daemon_threads = True
    url = f"http://{args.host}:{args.port}"
    key_status = "configured" if configured_api_key() else "not set (Demo/no-auth endpoints still work)"
    print(f"OpenAI-compatible comparison bench: {url}")
    print(f"OPENAI_API_KEY: {key_status}")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
