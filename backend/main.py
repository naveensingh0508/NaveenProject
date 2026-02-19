import json
import math
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, HttpUrl
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "notes.db")
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
ENV_PATH = os.path.join(BASE_DIR, ".env")
ROOT_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", ".env"))

# Runtime config (refreshed from .env when requests run).
GEMINI_API_KEY = ""
GOOGLE_API_KEY = ""
GEMINI_MODEL = ""
GEMINI_SUMMARY_MODEL = ""
MAX_TRANSCRIPT_CHARS = 12000
TRANSCRIPT_CHUNK_SIZE = 12000
TRANSCRIPT_CHUNK_OVERLAP = 500
TRANSCRIPT_MAX_CHUNKS = 8
SUMMARY_MODEL = ""
WEBSHARE_PROXY_USERNAME = ""
WEBSHARE_PROXY_PASSWORD = ""
WEBSHARE_PROXY_LOCATIONS = ""
YT_PROXY_HTTP_URL = ""
YT_PROXY_HTTPS_URL = ""
YT_PROXY_RETRIES = 10

api_key = ""
client: Optional[genai.Client] = None


def reload_runtime_config() -> None:
    """Reload .env values so key changes work without manual code changes."""
    global GEMINI_API_KEY
    global GOOGLE_API_KEY
    global GEMINI_MODEL
    global GEMINI_SUMMARY_MODEL
    global MAX_TRANSCRIPT_CHARS
    global TRANSCRIPT_CHUNK_SIZE
    global TRANSCRIPT_CHUNK_OVERLAP
    global TRANSCRIPT_MAX_CHUNKS
    global SUMMARY_MODEL
    global WEBSHARE_PROXY_USERNAME
    global WEBSHARE_PROXY_PASSWORD
    global WEBSHARE_PROXY_LOCATIONS
    global YT_PROXY_HTTP_URL
    global YT_PROXY_HTTPS_URL
    global YT_PROXY_RETRIES
    global api_key
    global client

    # Load root .env first (optional), then backend/.env as authoritative.
    load_dotenv(ROOT_ENV_PATH, override=False)
    load_dotenv(ENV_PATH, override=True)

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()
    GEMINI_SUMMARY_MODEL = os.getenv("GEMINI_SUMMARY_MODEL", "").strip()
    MAX_TRANSCRIPT_CHARS = get_int_env("MAX_TRANSCRIPT_CHARS", 12000)
    TRANSCRIPT_CHUNK_SIZE = get_int_env("TRANSCRIPT_CHUNK_SIZE", 12000)
    TRANSCRIPT_CHUNK_OVERLAP = get_int_env("TRANSCRIPT_CHUNK_OVERLAP", 500)
    TRANSCRIPT_MAX_CHUNKS = get_int_env("TRANSCRIPT_MAX_CHUNKS", 8)
    SUMMARY_MODEL = GEMINI_SUMMARY_MODEL or GEMINI_MODEL
    WEBSHARE_PROXY_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME", "").strip()
    WEBSHARE_PROXY_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD", "").strip()
    WEBSHARE_PROXY_LOCATIONS = os.getenv("WEBSHARE_PROXY_LOCATIONS", "").strip()
    YT_PROXY_HTTP_URL = os.getenv("YT_PROXY_HTTP_URL", "").strip()
    YT_PROXY_HTTPS_URL = os.getenv("YT_PROXY_HTTPS_URL", "").strip()
    YT_PROXY_RETRIES = get_int_env("YT_PROXY_RETRIES", 10)

    new_api_key = GEMINI_API_KEY or GOOGLE_API_KEY
    if new_api_key != api_key:
        api_key = new_api_key
        client = genai.Client(api_key=api_key) if api_key else None


reload_runtime_config()

app = FastAPI(title="YouTube Notes Generator", version="1.0.0")

# Allow local frontend calls during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    url: HttpUrl


YOUTUBE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def ensure_gemini_ready() -> None:
    reload_runtime_config()
    if not client:
        raise HTTPException(
            500,
            f"GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it in {ENV_PATH}",
        )


def _raise_transcript_fetch_error(exc: Exception) -> None:
    text = str(exc).lower()
    if isinstance(exc, (IpBlocked, RequestBlocked)) or "blocking requests from your ip" in text:
        raise HTTPException(
            502,
            (
                "YouTube blocked transcript requests from this server IP "
                "(common on Render/free cloud hosts). "
                "Set WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD "
                "in environment variables, or run backend locally."
            ),
        ) from exc

    raise HTTPException(502, f"Failed to fetch transcript: {exc}") from exc


def build_youtube_api() -> YouTubeTranscriptApi:
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        locations = [
            location.strip()
            for location in WEBSHARE_PROXY_LOCATIONS.split(",")
            if location.strip()
        ]
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=WEBSHARE_PROXY_USERNAME,
                proxy_password=WEBSHARE_PROXY_PASSWORD,
                filter_ip_locations=locations or None,
                retries_when_blocked=max(1, YT_PROXY_RETRIES),
            )
        )

    if YT_PROXY_HTTP_URL or YT_PROXY_HTTPS_URL:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=YT_PROXY_HTTP_URL or None,
                https_url=YT_PROXY_HTTPS_URL or None,
            )
        )

    return YouTubeTranscriptApi()


def _normalize_model_name(name: str) -> str:
    value = (name or "").strip()
    if value.startswith("models/"):
        value = value.split("/", 1)[1]
    return value


def _is_model_not_supported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    checks = [
        "404",
        "not_found",
        "model not found",
        "is not found for api version",
        "not supported for generatecontent",
        "unsupported",
        "call listmodels",
    ]
    return any(token in text for token in checks)


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    checks = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
    ]
    return any(token in text for token in checks)


def _friendly_quota_message(model_name: str, exc: Exception) -> str:
    raw = str(exc)
    retry_match = re.search(r"retry(?:delay)?[^0-9]*([0-9]+(?:\.[0-9]+)?)s", raw, re.I)
    retry_hint = ""
    if retry_match:
        retry_hint = f" Retry after about {retry_match.group(1)} seconds."
    return (
        f"Gemini quota exceeded for model '{model_name}'. "
        "Your current key/project has no available quota for this request."
        + retry_hint
        + " In Google AI Studio, check plan/billing or switch to a model with available quota."
    )


def get_model_candidates(preferred_model: str = "") -> List[str]:
    candidates: List[str] = []

    def _add(name: str) -> None:
        normalized = _normalize_model_name(name)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _add(preferred_model)
    _add(GEMINI_MODEL)
    _add(SUMMARY_MODEL)

    # Stable fallback names across common Gemini API setups.
    for name in [
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
    ]:
        _add(name)

    # If available, append server-reported models that support generation.
    try:
        for model in client.models.list():
            model_name = _normalize_model_name(getattr(model, "name", ""))
            supported = getattr(model, "supported_generation_methods", None)
            if supported:
                methods = [str(m).lower() for m in supported]
                if not any("generatecontent" in m for m in methods):
                    continue
            if "gemini" in model_name:
                _add(model_name)
    except Exception:
        pass

    return candidates


def gemini_generate_with_fallback(
    *,
    preferred_model: str,
    contents: str,
    response_mime_type: str,
    temperature: float,
    error_prefix: str,
):
    ensure_gemini_ready()
    candidates = get_model_candidates(preferred_model)
    last_error: Optional[Exception] = None

    for model_name in candidates:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config={
                    "response_mime_type": response_mime_type,
                    "temperature": temperature,
                },
            )
        except Exception as exc:
            last_error = exc
            if _is_quota_error(exc):
                raise HTTPException(429, _friendly_quota_message(model_name, exc)) from exc
            if _is_model_not_supported_error(exc):
                continue
            raise HTTPException(
                502,
                f"{error_prefix} ({model_name}). Check API key, model access, and network.",
            ) from exc

    if candidates:
        tried = ", ".join(candidates[:6])
        raise HTTPException(
            502,
            f"{error_prefix}: no usable Gemini model found. Tried: {tried}. Last error: {last_error}",
        )

    raise HTTPException(502, f"{error_prefix}: no Gemini models available.")


def chunk_text(text: str, chunk_size: int, overlap: int, max_chunks: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunk_size = max(1, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    if max_chunks and len(text) > chunk_size * max_chunks:
        chunk_size = math.ceil(len(text) / max_chunks)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def summarize_transcript_chunk(chunk: str, index: int, total: int) -> str:
    prompt = (
        "Summarize this transcript chunk into concise key points. "
        "Use plain text bullets only. Keep important technical details.\n"
        f"Chunk {index} of {total}:\n{chunk}"
    )
    try:
        response = gemini_generate_with_fallback(
            preferred_model=SUMMARY_MODEL,
            contents=prompt,
            response_mime_type="text/plain",
            temperature=0.2,
            error_prefix="Gemini API error while summarizing",
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(502, f"Gemini API error while summarizing: {exc}") from exc

    return (response.text or "").strip()


def summarize_long_transcript(transcript: str) -> str:
    chunks = chunk_text(
        transcript,
        TRANSCRIPT_CHUNK_SIZE,
        TRANSCRIPT_CHUNK_OVERLAP,
        TRANSCRIPT_MAX_CHUNKS,
    )
    if not chunks:
        return ""
    summaries = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        summaries.append(summarize_transcript_chunk(chunk, index, total))
    return "\n".join(summary for summary in summaries if summary)


def prepare_transcript(transcript: str) -> str:
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return transcript

    summary = ""
    try:
        summary = summarize_long_transcript(transcript)
    except Exception:
        summary = ""

    if summary:
        return (
            "This is a condensed summary of a long transcript. "
            "Use it to generate complete study notes.\n"
            + summary
        )

    truncated = transcript[:MAX_TRANSCRIPT_CHARS]
    return (
        f"This transcript was truncated to {MAX_TRANSCRIPT_CHARS} characters due to length.\n"
        + truncated
    )


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                video_url TEXT NOT NULL,
                video_title TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def extract_video_id(url: str) -> Optional[str]:
    url = url.strip()
    if YOUTUBE_ID_RE.match(url):
        return url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be", "www.youtu.be"}:
        return parsed.path.lstrip("/")

    if "youtube.com" in host:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/shorts/"):
            parts = parsed.path.split("/")
            return parts[2] if len(parts) > 2 else None
        if parsed.path.startswith("/embed/"):
            parts = parsed.path.split("/")
            return parts[2] if len(parts) > 2 else None

    return None


def fetch_video_title(url: str) -> str:
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=10,
        )
        if response.ok:
            payload = response.json()
            title = payload.get("title")
            if title:
                return title
    except requests.RequestException:
        pass
    return "YouTube Video"


def fetch_transcript(video_id: str) -> str:
    def _join_entries(entries: Any) -> str:
        if hasattr(entries, "to_raw_data"):
            entries = entries.to_raw_data()
        if isinstance(entries, list):
            return " ".join(
                item.get("text", "") for item in entries if isinstance(item, dict)
            )
        return " ".join(getattr(item, "text", "") for item in entries)

    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            try:
                entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            except Exception:
                entries = YouTubeTranscriptApi.get_transcript(video_id)
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
            raise HTTPException(
                400,
                "Transcript not available for this video. Make sure captions are enabled.",
            ) from exc
        except Exception as exc:
            _raise_transcript_fetch_error(exc)
        return _join_entries(entries)

    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
            raise HTTPException(
                400,
                "Transcript not available for this video. Make sure captions are enabled.",
            ) from exc
        except Exception as exc:
            _raise_transcript_fetch_error(exc)

        transcript = None
        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except Exception:
            transcript = next(iter(transcript_list), None)
            if transcript and getattr(transcript, "is_translatable", False):
                try:
                    transcript = transcript.translate("en")
                except Exception:
                    pass

        if not transcript:
            raise HTTPException(
                400,
                "Transcript not available for this video. Try a video with subtitles.",
            )

        try:
            entries = transcript.fetch()
        except Exception as exc:
            _raise_transcript_fetch_error(exc)

        return _join_entries(entries)

    try:
        api = build_youtube_api()
    except Exception as exc:
        raise HTTPException(
            500,
            f"Failed to initialize transcript client. Check proxy environment variables. {exc}",
        ) from exc

    try:
        transcript_list = api.list(video_id)
    except (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    ) as exc:
        raise HTTPException(
            400,
            "Transcript not available for this video. Make sure captions are enabled.",
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        text = str(exc).lower()
        if "blocking requests from your ip" in text:
            _raise_transcript_fetch_error(exc)
        raise HTTPException(
            400,
            "Could not retrieve transcript for this video. Try another video with open captions.",
        ) from exc
    except Exception as exc:
        _raise_transcript_fetch_error(exc)

    transcript = None
    try:
        transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
    except Exception:
        transcript = next(iter(transcript_list), None)
        if transcript and getattr(transcript, "is_translatable", False):
            try:
                transcript = transcript.translate("en")
            except Exception:
                pass

    if not transcript:
        raise HTTPException(
            400,
            "Transcript not available for this video. Try a video with subtitles.",
        )

    try:
        entries = transcript.fetch()
    except Exception as exc:
        _raise_transcript_fetch_error(exc)

    return _join_entries(entries)


def build_prompt(transcript: str) -> str:
    return (
        "Analyze this YouTube transcript and create a complete study pack.\n"
        "Requirements:\n"
        "- Break content into clear topics in learning order.\n"
        "- For each topic create:\n"
        "  1) A full explanation (8-12 sentences, include definitions, context, and at least one example).\n"
        "  2) 6-10 bullet notes with key points and formulas/terms where relevant.\n"
        "  3) At least 2 Mermaid diagrams using diagram types that best fit the topic.\n"
        "     Allowed types: flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, erDiagram, journey, pie, gantt, mindmap, timeline.\n"
        "  4) 5 MCQ quiz questions, each with 4 options and one correct answer text.\n"
        "Return ONLY a valid JSON object with this shape:\n"
        "{\n"
        "  \"table_of_contents\": [\"Topic A\", \"Topic B\"],\n"
        "  \"topics\": [\n"
        "    {\n"
        "      \"title\": \"Topic A\",\n"
        "      \"explanation\": \"...\",\n"
        "      \"bullet_notes\": [\"...\"],\n"
        "      \"diagrams\": [\n"
        "        {\"title\": \"Concept Map\", \"diagram_type\": \"mindmap\", \"mermaid\": \"mindmap\\n  root((Topic))\"},\n"
        "        {\"title\": \"Process\", \"diagram_type\": \"sequenceDiagram\", \"mermaid\": \"sequenceDiagram\\nA->>B: step\"}\n"
        "      ],\n"
        "      \"quiz\": [\n"
        "        {\"question\": \"...\", \"options\": [\"A\",\"B\",\"C\",\"D\"], \"correct_answer\": \"B\"}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Output JSON only, no markdown.\n"
        "- table_of_contents must match topic titles.\n"
        "- Ensure explanations are detailed and teaching-oriented, not short summaries.\n"
        "\nTranscript:\n"
        f"{transcript}"
    )


def call_gemini(transcript: str) -> Dict[str, Any]:
    prompt = build_prompt(transcript)

    try:
        response = gemini_generate_with_fallback(
            preferred_model=GEMINI_MODEL,
            contents=prompt,
            response_mime_type="application/json",
            temperature=0.2,
            error_prefix="Gemini API error",
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(502, f"Gemini API error: {exc}") from exc

    raw = (response.text or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repair_prompt = (
            "The previous output was invalid JSON. "
            "Return corrected JSON only, with no extra text.\n"
            f"Invalid output:\n{raw}"
        )
        try:
            response = gemini_generate_with_fallback(
                preferred_model=GEMINI_MODEL,
                contents=repair_prompt,
                response_mime_type="application/json",
                temperature=0,
                error_prefix="Gemini API error while repairing JSON",
            )
            repaired_raw = (response.text or "").strip()
            return json.loads(repaired_raw)
        except Exception as exc:
            raise HTTPException(502, "Failed to parse JSON from Gemini output.") from exc


def ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    topics_in = data.get("topics") or []
    topics: List[Dict[str, Any]] = []

    for topic in topics_in:
        if not isinstance(topic, dict):
            continue

        title = str(topic.get("title") or "Untitled Topic").strip()
        explanation = str(topic.get("explanation") or "").strip()
        bullet_notes = ensure_list(topic.get("bullet_notes"))

        diagrams: List[Dict[str, str]] = []
        raw_diagrams = topic.get("diagrams")
        if isinstance(raw_diagrams, list):
            for item in raw_diagrams:
                if not isinstance(item, dict):
                    continue
                mermaid = str(item.get("mermaid") or "").strip()
                if not mermaid:
                    continue
                diagrams.append(
                    {
                        "title": str(item.get("title") or "Diagram").strip() or "Diagram",
                        "diagram_type": str(item.get("diagram_type") or "mermaid").strip() or "mermaid",
                        "mermaid": mermaid,
                    }
                )

        fallback_diagram = str(topic.get("diagram") or "").strip()
        if not diagrams and fallback_diagram:
            diagrams.append(
                {
                    "title": "Diagram",
                    "diagram_type": "mermaid",
                    "mermaid": fallback_diagram,
                }
            )

        quiz_items = []
        for quiz in topic.get("quiz") or []:
            if not isinstance(quiz, dict):
                continue

            options = ensure_list(quiz.get("options"))
            while len(options) < 4:
                options.append(f"Option {len(options) + 1}")
            options = options[:4]

            correct = str(quiz.get("correct_answer") or "").strip()
            answer_index = quiz.get("answer_index")
            if not correct and isinstance(answer_index, int) and 0 <= answer_index < len(options):
                correct = options[answer_index]

            quiz_items.append(
                {
                    "question": str(quiz.get("question") or "").strip(),
                    "options": options,
                    "correct_answer": correct,
                }
            )

        topics.append(
            {
                "title": title,
                "explanation": explanation,
                "bullet_notes": bullet_notes,
                "diagram": diagrams[0]["mermaid"] if diagrams else "",
                "diagrams": diagrams,
                "quiz": quiz_items,
            }
        )

    toc = ensure_list(data.get("table_of_contents"))
    if not toc:
        toc = [topic["title"] for topic in topics]

    return {"table_of_contents": toc, "topics": topics}


def save_result(result: Dict[str, Any]) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO notes (created_at, video_url, video_title, result_json) VALUES (?, ?, ?, ?)",
            (
                result["created_at"],
                result["video_url"],
                result["video_title"],
                json.dumps(result),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def load_history(limit: int = 50) -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, video_url, video_title FROM notes ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def load_history_item(note_id: int) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, created_at, video_url, video_title, result_json FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()

    if not row:
        return None

    result = json.loads(row["result_json"])
    result["id"] = row["id"]
    result["created_at"] = row["created_at"]
    result["video_url"] = row["video_url"]
    result["video_title"] = row["video_title"]
    return result


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate")
def generate_notes(request: GenerateRequest) -> Dict[str, Any]:
    reload_runtime_config()
    video_url = str(request.url)
    video_id = extract_video_id(video_url)

    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL.")

    transcript = fetch_transcript(video_id)
    video_title = fetch_video_title(video_url)
    prepared_transcript = prepare_transcript(transcript)

    ai_data = call_gemini(prepared_transcript)
    normalized = normalize_result(ai_data)

    created_at = datetime.utcnow().isoformat() + "Z"
    result = {
        "video_url": video_url,
        "video_title": video_title,
        "created_at": created_at,
        **normalized,
    }

    record_id = save_result(result)
    result["id"] = record_id
    return result


@app.get("/api/history")
def history() -> List[Dict[str, Any]]:
    return load_history()


@app.get("/api/history/{note_id}")
def history_item(note_id: int) -> Dict[str, Any]:
    item = load_history_item(note_id)
    if not item:
        raise HTTPException(404, "History entry not found.")
    return item


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

init_db()
