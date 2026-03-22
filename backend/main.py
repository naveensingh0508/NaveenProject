import json
import hashlib
import hmac
import math
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
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
OTP_EXPIRY_MINUTES = 10
OTP_MAX_ATTEMPTS = 5
OTP_DEBUG_MODE = True
SMS_PROVIDER = "twilio"
OTP_DEFAULT_COUNTRY_CODE = "+91"
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_FROM_NUMBER = ""
TWILIO_MESSAGING_SERVICE_SID = ""

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
    global OTP_EXPIRY_MINUTES
    global OTP_MAX_ATTEMPTS
    global OTP_DEBUG_MODE
    global SMS_PROVIDER
    global OTP_DEFAULT_COUNTRY_CODE
    global TWILIO_ACCOUNT_SID
    global TWILIO_AUTH_TOKEN
    global TWILIO_FROM_NUMBER
    global TWILIO_MESSAGING_SERVICE_SID
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
    OTP_EXPIRY_MINUTES = get_int_env("OTP_EXPIRY_MINUTES", 10)
    OTP_MAX_ATTEMPTS = get_int_env("OTP_MAX_ATTEMPTS", 5)
    OTP_DEBUG_MODE = os.getenv("OTP_DEBUG_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
    SMS_PROVIDER = os.getenv("SMS_PROVIDER", "twilio").strip().lower()
    OTP_DEFAULT_COUNTRY_CODE = os.getenv("OTP_DEFAULT_COUNTRY_CODE", "+91").strip() or "+91"
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()
    TWILIO_MESSAGING_SERVICE_SID = os.getenv("TWILIO_MESSAGING_SERVICE_SID", "").strip()

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


@app.middleware("http")
async def disable_dev_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path.lower()
    if request.method == "GET" and (
        path == "/" or path.endswith(".html") or path.endswith(".js") or path.endswith(".css")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


class GenerateRequest(BaseModel):
    url: HttpUrl


class TranslateNotesRequest(BaseModel):
    notes: Dict[str, Any]
    target_language: str
    target_language_name: Optional[str] = None


class RegisterRequest(BaseModel):
    full_name: str
    mobile_number: str
    password: str


class RequestRegisterOtpRequest(BaseModel):
    full_name: str
    mobile_number: str
    password: str


class VerifyRegisterOtpRequest(BaseModel):
    mobile_number: str
    otp: str


class LoginRequest(BaseModel):
    mobile_number: str
    password: str


YOUTUBE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
MOBILE_RE = re.compile(r"^[0-9]{10,15}$")
SESSION_TTL_HOURS = 24 * 7
OTP_LENGTH = 6


def _clean_text(value: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:max_len]


def _normalize_mobile_number(mobile_number: str) -> str:
    raw = str(mobile_number or "").strip()
    digits = re.sub(r"\D", "", raw)

    if raw and not raw.startswith("+") and len(digits) == 10:
        cc_digits = re.sub(r"\D", "", OTP_DEFAULT_COUNTRY_CODE or "")
        if cc_digits:
            digits = f"{cc_digits}{digits}"

    if not MOBILE_RE.match(digits):
        raise HTTPException(
            400,
            "Invalid mobile number. Use full format like +919876543210 or a 10-digit local number.",
        )
    return f"+{digits}"


def _mobile_to_placeholder_email(mobile_number: str) -> str:
    return f"m{mobile_number.lstrip('+')}@mobile.local"


def _hash_password(password: str, salt_hex: Optional[str] = None) -> Dict[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return {"salt": salt.hex(), "hash": digest.hex()}


def _verify_password(password: str, salt_hex: str, password_hash_hex: str) -> bool:
    computed = _hash_password(password, salt_hex)["hash"]
    return hmac.compare_digest(computed, password_hash_hex)


def _serialize_student(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "full_name": str(row["full_name"]),
        "email": str(row["email"]),
        "mobile_number": str(row["mobile_number"] or ""),
        "student_id": str(row["student_id"]),
        "department": str(row["department"] or ""),
        "year_level": str(row["year_level"] or ""),
        "created_at": str(row["created_at"]),
        "last_login_at": str(row["last_login_at"] or ""),
    }


def _extract_bearer_token(authorization: Optional[str]) -> str:
    header = str(authorization or "").strip()
    if not header:
        raise HTTPException(401, "Missing Authorization header.")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "Invalid Authorization header format.")
    token = header[7:].strip()
    if not token:
        raise HTTPException(401, "Missing bearer token.")
    return token


def _create_session_token(conn: sqlite3.Connection, student_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now_epoch = int(datetime.utcnow().timestamp())
    expires_epoch = int((datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)).timestamp())
    conn.execute(
        "INSERT INTO sessions (token, student_id, created_at_epoch, expires_at_epoch) VALUES (?, ?, ?, ?)",
        (token, student_id, now_epoch, expires_epoch),
    )
    return token


def _student_from_token(authorization: Optional[str] = None) -> Dict[str, Any]:
    # Authentication completely removed. 
    # Mocking a global student to satisfy database interactions safely.
    return {
        "id": 1,
        "full_name": "Global User",
        "email": "global@studykit.local",
        "mobile_number": "1234567890",
        "student_id": "GLOBAL001",
        "department": "General",
        "year_level": "1",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "last_login_at": datetime.utcnow().isoformat() + "Z",
    }


def _generate_student_code() -> str:
    return f"STU{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{secrets.randbelow(1000):03d}"


def _send_otp_via_twilio(mobile_number: str, otp: str) -> None:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise HTTPException(
            500,
            "Twilio is not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in backend/.env.",
        )

    if not TWILIO_MESSAGING_SERVICE_SID and not TWILIO_FROM_NUMBER:
        raise HTTPException(
            500,
            (
                "Twilio sender is missing. Set TWILIO_MESSAGING_SERVICE_SID "
                "or TWILIO_FROM_NUMBER in backend/.env."
            ),
        )

    payload = {
        "To": mobile_number,
        "Body": (
            f"Your StudyKit Pro verification code is {otp}. "
            f"It expires in {max(1, OTP_EXPIRY_MINUTES)} minutes."
        ),
    }
    if TWILIO_MESSAGING_SERVICE_SID:
        payload["MessagingServiceSid"] = TWILIO_MESSAGING_SERVICE_SID
    else:
        payload["From"] = TWILIO_FROM_NUMBER

    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    try:
        response = requests.post(
            url,
            data=payload,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=20,
        )
    except requests.RequestException as exc:
        raise HTTPException(502, f"Failed to connect to Twilio: {exc}") from exc

    if response.status_code >= 400:
        detail = ""
        try:
            body = response.json()
            detail = str(body.get("message") or "").strip()
        except ValueError:
            detail = response.text.strip()
        detail = detail[:220]
        raise HTTPException(
            502,
            f"Twilio rejected SMS request: {detail or 'Unknown provider error.'}",
        )


def _deliver_mobile_otp(mobile_number: str, otp: str) -> Optional[str]:
    # Demo mode shows OTP directly in API response.
    if OTP_DEBUG_MODE:
        return otp

    if SMS_PROVIDER == "twilio":
        _send_otp_via_twilio(mobile_number, otp)
        return None

    raise HTTPException(
        500,
        f"Unsupported SMS_PROVIDER '{SMS_PROVIDER}'. Supported providers: twilio.",
    )


def request_registration_otp(payload: RequestRegisterOtpRequest) -> Dict[str, Any]:
    reload_runtime_config()
    full_name = _clean_text(payload.full_name, 120)
    mobile_number = _normalize_mobile_number(payload.mobile_number)
    password = str(payload.password or "")

    if len(full_name) < 2:
        raise HTTPException(400, "Full name must be at least 2 characters.")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    now_epoch = int(datetime.utcnow().timestamp())
    expires_epoch = int((datetime.utcnow() + timedelta(minutes=max(1, OTP_EXPIRY_MINUTES))).timestamp())
    otp = f"{secrets.randbelow(10**OTP_LENGTH):0{OTP_LENGTH}d}"
    otp_secret = _hash_password(otp)
    password_secret = _hash_password(password)
    otp_preview: Optional[str] = None

    with sqlite3.connect(DB_PATH) as conn:
        existing_student = conn.execute(
            "SELECT id FROM students WHERE mobile_number = ?",
            (mobile_number,),
        ).fetchone()
        if existing_student:
            raise HTTPException(409, "An account with this mobile number already exists.")

        conn.execute("DELETE FROM registration_otps WHERE mobile_number = ?", (mobile_number,))
        conn.execute(
            """
            INSERT INTO registration_otps (
                mobile_number,
                full_name,
                password_salt,
                password_hash,
                otp_salt,
                otp_hash,
                created_at_epoch,
                expires_at_epoch,
                attempts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                mobile_number,
                full_name,
                password_secret["salt"],
                password_secret["hash"],
                otp_secret["salt"],
                otp_secret["hash"],
                now_epoch,
                expires_epoch,
            ),
        )
        otp_preview = _deliver_mobile_otp(mobile_number, otp)
        conn.commit()

    response: Dict[str, Any] = {
        "message": f"OTP sent to {mobile_number}. Please verify to complete registration."
    }
    if otp_preview:
        response["otp_preview"] = otp_preview
    return response


def verify_registration_otp_and_register(payload: VerifyRegisterOtpRequest) -> Dict[str, Any]:
    mobile_number = _normalize_mobile_number(payload.mobile_number)
    otp = re.sub(r"\D", "", str(payload.otp or "").strip())
    if len(otp) != OTP_LENGTH:
        raise HTTPException(400, f"OTP must be {OTP_LENGTH} digits.")

    now_epoch = int(datetime.utcnow().timestamp())
    created_at = datetime.utcnow().isoformat() + "Z"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, mobile_number, full_name, password_salt, password_hash, otp_salt, otp_hash, expires_at_epoch, attempts
            FROM registration_otps
            WHERE mobile_number = ?
            """,
            (mobile_number,),
        ).fetchone()

        if not row:
            raise HTTPException(400, "No pending registration found. Request OTP again.")

        if int(row["expires_at_epoch"]) <= now_epoch:
            conn.execute("DELETE FROM registration_otps WHERE id = ?", (int(row["id"]),))
            conn.commit()
            raise HTTPException(400, "OTP expired. Request a new OTP.")

        if int(row["attempts"]) >= max(1, OTP_MAX_ATTEMPTS):
            conn.execute("DELETE FROM registration_otps WHERE id = ?", (int(row["id"]),))
            conn.commit()
            raise HTTPException(429, "Too many invalid OTP attempts. Request a new OTP.")

        if not _verify_password(otp, str(row["otp_salt"]), str(row["otp_hash"])):
            conn.execute(
                "UPDATE registration_otps SET attempts = attempts + 1 WHERE id = ?",
                (int(row["id"]),),
            )
            conn.commit()
            raise HTTPException(400, "Invalid OTP.")

        existing_student = conn.execute(
            "SELECT id FROM students WHERE mobile_number = ?",
            (mobile_number,),
        ).fetchone()
        if existing_student:
            conn.execute("DELETE FROM registration_otps WHERE id = ?", (int(row["id"]),))
            conn.commit()
            raise HTTPException(409, "An account with this mobile number already exists.")

        student_code = _generate_student_code()
        placeholder_email = _mobile_to_placeholder_email(mobile_number)
        cursor = conn.execute(
            """
            INSERT INTO students (full_name, email, mobile_number, student_id, department, year_level, password_salt, password_hash, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row["full_name"]),
                placeholder_email,
                mobile_number,
                student_code,
                "",
                "",
                str(row["password_salt"]),
                str(row["password_hash"]),
                created_at,
                created_at,
            ),
        )
        student_db_id = int(cursor.lastrowid)

        conn.execute("DELETE FROM registration_otps WHERE id = ?", (int(row["id"]),))
        token = _create_session_token(conn, student_db_id)

        student_row = conn.execute(
            "SELECT id, full_name, email, mobile_number, student_id, department, year_level, created_at, last_login_at FROM students WHERE id = ?",
            (student_db_id,),
        ).fetchone()
        conn.commit()

    if not student_row:
        raise HTTPException(500, "Unable to complete registration.")

    return {"token": token, "student": _serialize_student(student_row)}


def register_student(payload: RegisterRequest) -> Dict[str, Any]:
    full_name = _clean_text(payload.full_name, 120)
    mobile_number = _normalize_mobile_number(payload.mobile_number)
    password = str(payload.password or "")

    if len(full_name) < 2:
        raise HTTPException(400, "Full name must be at least 2 characters.")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    created_at = datetime.utcnow().isoformat() + "Z"
    password_secret = _hash_password(password)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        existing_student = conn.execute(
            "SELECT id FROM students WHERE mobile_number = ?",
            (mobile_number,),
        ).fetchone()
        if existing_student:
            raise HTTPException(409, "An account with this mobile number already exists.")

        student_code = _generate_student_code()
        placeholder_email = _mobile_to_placeholder_email(mobile_number)
        cursor = conn.execute(
            """
            INSERT INTO students (full_name, email, mobile_number, student_id, department, year_level, password_salt, password_hash, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                full_name,
                placeholder_email,
                mobile_number,
                student_code,
                "",
                "",
                password_secret["salt"],
                password_secret["hash"],
                created_at,
                created_at,
            ),
        )
        student_db_id = int(cursor.lastrowid)
        token = _create_session_token(conn, student_db_id)

        student_row = conn.execute(
            "SELECT id, full_name, email, mobile_number, student_id, department, year_level, created_at, last_login_at FROM students WHERE id = ?",
            (student_db_id,),
        ).fetchone()
        conn.commit()

    if not student_row:
        raise HTTPException(500, "Unable to complete registration.")

    return {"token": token, "student": _serialize_student(student_row)}


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
    http_client = requests.Session()
    # Ignore machine-level proxy env vars unless an explicit proxy config is provided.
    http_client.trust_env = False

    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
        locations = [
            location.strip()
            for location in WEBSHARE_PROXY_LOCATIONS.split(",")
            if location.strip()
        ]
        return YouTubeTranscriptApi(
            http_client=http_client,
            proxy_config=WebshareProxyConfig(
                proxy_username=WEBSHARE_PROXY_USERNAME,
                proxy_password=WEBSHARE_PROXY_PASSWORD,
                filter_ip_locations=locations or None,
                retries_when_blocked=max(1, YT_PROXY_RETRIES),
            )
        )

    if YT_PROXY_HTTP_URL or YT_PROXY_HTTPS_URL:
        return YouTubeTranscriptApi(
            http_client=http_client,
            proxy_config=GenericProxyConfig(
                http_url=YT_PROXY_HTTP_URL or None,
                https_url=YT_PROXY_HTTPS_URL or None,
            )
        )

    return YouTubeTranscriptApi(http_client=http_client)


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                mobile_number TEXT,
                student_id TEXT NOT NULL UNIQUE,
                department TEXT,
                year_level TEXT,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                student_id INTEGER NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                expires_at_epoch INTEGER NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS registration_otps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mobile_number TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                otp_salt TEXT NOT NULL,
                otp_hash TEXT NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                expires_at_epoch INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Migration for older DBs created before mobile verification fields.
        student_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(students)").fetchall()
        }
        if "mobile_number" not in student_columns:
            conn.execute("ALTER TABLE students ADD COLUMN mobile_number TEXT")

        otp_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(registration_otps)").fetchall()
        }
        if "mobile_number" not in otp_columns:
            conn.execute("DROP TABLE IF EXISTS registration_otps")
            conn.execute(
                """
                CREATE TABLE registration_otps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mobile_number TEXT NOT NULL UNIQUE,
                    full_name TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    otp_salt TEXT NOT NULL,
                    otp_hash TEXT NOT NULL,
                    created_at_epoch INTEGER NOT NULL,
                    expires_at_epoch INTEGER NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_student_id ON sessions(student_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_students_email ON students(email)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_students_mobile_number ON students(mobile_number)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_registration_otps_mobile_number ON registration_otps(mobile_number)"
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
        with requests.Session() as session:
            # Keep title fetch independent from broken system proxy variables.
            session.trust_env = False
            response = session.get(
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
        "Analyze this YouTube transcript and create a student-friendly study pack.\n"
        "Audience: beginners preparing to understand and revise quickly.\n"
        "Requirements:\n"
        "- Break content into clear topics in learning order.\n"
        "- For each topic create:\n"
        "  1) 2-4 learning objectives (start with action verbs like Explain, Compare, Apply).\n"
        "  2) A full explanation written as short teaching blocks, not one long paragraph.\n"
        "     Use 8-12 short sentences in plain language, define terms, show why it matters, include one concrete example, and separate ideas naturally.\n"
        "  3) 6-10 bullet notes with key points/formulas where relevant.\n"
        "  4) 3-6 key terms with short definitions.\n"
        "  5) 2-4 common mistakes students make and how to avoid them.\n"
        "  6) 2-4 self-check Q&A items with concise answers.\n"
        "  7) A quick recap list (3-5 points) for last-minute revision.\n"
        "  8) At least 2 Mermaid diagrams using diagram types that best fit the topic.\n"
        "     Allowed types: flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, erDiagram, journey, pie, gantt, mindmap, timeline.\n"
        "  9) 5 MCQ quiz questions, each with 4 options and one correct answer text.\n"
        "Return ONLY a valid JSON object with this shape:\n"
        "{\n"
        "  \"table_of_contents\": [\"Topic A\", \"Topic B\"],\n"
        "  \"topics\": [\n"
        "    {\n"
        "      \"title\": \"Topic A\",\n"
        "      \"learning_objectives\": [\"Explain ...\", \"Apply ...\"],\n"
        "      \"explanation\": \"...\",\n"
        "      \"bullet_notes\": [\"...\"],\n"
        "      \"key_terms\": [\n"
        "        {\"term\": \"...\", \"definition\": \"...\"}\n"
        "      ],\n"
        "      \"common_mistakes\": [\"...\"],\n"
        "      \"self_check\": [\n"
        "        {\"question\": \"...\", \"answer\": \"...\"}\n"
        "      ],\n"
        "      \"quick_recap\": [\"...\"],\n"
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
        "- Do not write explanation as one dense paragraph. Keep it naturally broken into short readable parts.\n"
        "- Keep language clear for students (avoid unexplained jargon).\n"
        "- Keep each list item concise and study-ready.\n"
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
        items = value
    elif value is None:
        return []
    else:
        items = [value]

    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def ensure_term_list(value: Any) -> List[Dict[str, str]]:
    terms: List[Dict[str, str]] = []
    if not isinstance(value, list):
        return terms

    for item in value:
        if isinstance(item, dict):
            term = str(item.get("term") or item.get("name") or "").strip()
            definition = str(
                item.get("definition") or item.get("meaning") or item.get("explanation") or ""
            ).strip()
        else:
            raw = str(item).strip()
            if ":" in raw:
                head, tail = raw.split(":", 1)
                term = head.strip()
                definition = tail.strip()
            else:
                term = raw
                definition = ""

        if not term:
            continue

        terms.append({"term": term, "definition": definition})

    return terms


def ensure_qna_list(value: Any) -> List[Dict[str, str]]:
    qna: List[Dict[str, str]] = []
    if not isinstance(value, list):
        return qna

    for item in value:
        if isinstance(item, dict):
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or item.get("expected_answer") or "").strip()
        else:
            question = str(item).strip()
            answer = ""

        if not question:
            continue

        qna.append({"question": question, "answer": answer})

    return qna


def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    topics_in = data.get("topics") or []
    topics: List[Dict[str, Any]] = []

    for topic in topics_in:
        if not isinstance(topic, dict):
            continue

        title = str(topic.get("title") or "Untitled Topic").strip()
        explanation = str(topic.get("explanation") or "").strip()
        bullet_notes = ensure_list(topic.get("bullet_notes"))
        learning_objectives = ensure_list(topic.get("learning_objectives") or topic.get("objectives"))
        key_terms = ensure_term_list(topic.get("key_terms") or topic.get("glossary"))
        common_mistakes = ensure_list(topic.get("common_mistakes"))
        self_check = ensure_qna_list(topic.get("self_check") or topic.get("practice"))
        quick_recap = ensure_list(topic.get("quick_recap") or topic.get("summary_points"))

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
                "learning_objectives": learning_objectives,
                "explanation": explanation,
                "bullet_notes": bullet_notes,
                "key_terms": key_terms,
                "common_mistakes": common_mistakes,
                "self_check": self_check,
                "quick_recap": quick_recap,
                "diagram": diagrams[0]["mermaid"] if diagrams else "",
                "diagrams": diagrams,
                "quiz": quiz_items,
            }
        )

    toc = ensure_list(data.get("table_of_contents"))
    if not toc:
        toc = [topic["title"] for topic in topics]

    return {"table_of_contents": toc, "topics": topics}


def _coerce_language_code(language: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z-]", "", (language or "").strip().lower())
    return cleaned[:16] or "en"


def _coerce_language_name(name: str, fallback_code: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 \-()]", "", (name or "").strip())
    return (cleaned[:60] or fallback_code.upper()).strip()


def _build_translation_prompt(
    notes_json: str, target_language_code: str, target_language_name: str
) -> str:
    return (
        "Translate this study notes JSON into the target language.\n"
        f"Target language: {target_language_name} ({target_language_code}).\n"
        "Rules:\n"
        "- Return JSON only.\n"
        "- Keep all JSON keys exactly the same.\n"
        "- Keep the same general structure.\n"
        "- Translate natural language text fields for student readability.\n"
        "- Do NOT translate Mermaid syntax in diagrams.mermaid.\n"
        "- Do NOT translate URLs.\n"
        "- For each quiz item, ensure correct_answer exactly matches one translated option.\n"
        "- Keep explanations clear and beginner-friendly.\n"
        "- Preserve short explanation breaks when translating.\n"
        "Input JSON:\n"
        f"{notes_json}"
    )


def _preserve_diagrams_from_source(
    source_notes: Dict[str, Any], translated_notes: Dict[str, Any]
) -> None:
    source_topics = source_notes.get("topics") or []
    translated_topics = translated_notes.get("topics") or []
    for index, topic in enumerate(translated_topics):
        if index >= len(source_topics):
            break
        source_topic = source_topics[index] if isinstance(source_topics[index], dict) else {}
        if not isinstance(topic, dict):
            continue
        topic["diagrams"] = source_topic.get("diagrams") or []
        topic["diagram"] = source_topic.get("diagram") or ""


def _align_quiz_answers_with_options(
    source_notes: Dict[str, Any], translated_notes: Dict[str, Any]
) -> None:
    source_topics = source_notes.get("topics") or []
    translated_topics = translated_notes.get("topics") or []

    for topic_index, translated_topic in enumerate(translated_topics):
        if not isinstance(translated_topic, dict):
            continue
        source_topic = (
            source_topics[topic_index]
            if topic_index < len(source_topics) and isinstance(source_topics[topic_index], dict)
            else {}
        )

        source_quiz = source_topic.get("quiz") or []
        translated_quiz = translated_topic.get("quiz") or []
        normalized_quiz: List[Dict[str, Any]] = []

        for quiz_index, translated_item in enumerate(translated_quiz):
            if not isinstance(translated_item, dict):
                continue

            options = ensure_list(translated_item.get("options"))
            while len(options) < 4:
                options.append(f"Option {len(options) + 1}")
            options = options[:4]

            translated_correct = str(translated_item.get("correct_answer") or "").strip()
            if translated_correct in options:
                resolved_correct = translated_correct
            else:
                resolved_correct = ""
                source_item = (
                    source_quiz[quiz_index]
                    if quiz_index < len(source_quiz) and isinstance(source_quiz[quiz_index], dict)
                    else {}
                )
                source_options = ensure_list(source_item.get("options"))
                source_correct = str(source_item.get("correct_answer") or "").strip()
                if source_correct and source_correct in source_options:
                    source_correct_index = source_options.index(source_correct)
                    if source_correct_index < len(options):
                        resolved_correct = options[source_correct_index]

                if not resolved_correct and options:
                    resolved_correct = options[0]

            normalized_quiz.append(
                {
                    "question": str(translated_item.get("question") or "").strip(),
                    "options": options,
                    "correct_answer": resolved_correct,
                }
            )

        translated_topic["quiz"] = normalized_quiz


def translate_notes_payload(
    notes: Dict[str, Any], target_language_code: str, target_language_name: str
) -> Dict[str, Any]:
    normalized_source = normalize_result(notes)
    if target_language_code in {"en", "en-us", "en-gb"}:
        return normalized_source

    prompt = _build_translation_prompt(
        json.dumps(normalized_source, ensure_ascii=False),
        target_language_code,
        target_language_name,
    )

    try:
        response = gemini_generate_with_fallback(
            preferred_model=GEMINI_MODEL,
            contents=prompt,
            response_mime_type="application/json",
            temperature=0.1,
            error_prefix="Gemini API error while translating notes",
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(502, f"Gemini API error while translating notes: {exc}") from exc

    raw = (response.text or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        repair_prompt = (
            "The previous translated output was invalid JSON. "
            "Return corrected JSON only, with no extra text.\n"
            f"Invalid output:\n{raw}"
        )
        try:
            response = gemini_generate_with_fallback(
                preferred_model=GEMINI_MODEL,
                contents=repair_prompt,
                response_mime_type="application/json",
                temperature=0,
                error_prefix="Gemini API error while repairing translated JSON",
            )
            parsed = json.loads((response.text or "").strip())
        except Exception as exc:
            raise HTTPException(502, "Failed to parse translated JSON from Gemini output.") from exc

    translated = normalize_result(parsed)
    _preserve_diagrams_from_source(normalized_source, translated)
    _align_quiz_answers_with_options(normalized_source, translated)
    return translated


def login_student(payload: LoginRequest) -> Dict[str, Any]:
    mobile_number = _normalize_mobile_number(payload.mobile_number)
    password = str(payload.password or "")
    if not password:
        raise HTTPException(400, "Password is required.")

    now_iso = datetime.utcnow().isoformat() + "Z"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT id, full_name, email, mobile_number, student_id, department, year_level, created_at, last_login_at, password_salt, password_hash
            FROM students
            WHERE mobile_number = ?
            """,
            (mobile_number,),
        ).fetchone()

        if not row:
            raise HTTPException(401, "Invalid mobile number or password.")

        if not _verify_password(password, str(row["password_salt"]), str(row["password_hash"])):
            raise HTTPException(401, "Invalid mobile number or password.")

        student_db_id = int(row["id"])
        conn.execute("UPDATE students SET last_login_at = ? WHERE id = ?", (now_iso, student_db_id))
        conn.execute("DELETE FROM sessions WHERE student_id = ?", (student_db_id,))
        token = _create_session_token(conn, student_db_id)

        student_row = conn.execute(
            "SELECT id, full_name, email, mobile_number, student_id, department, year_level, created_at, last_login_at FROM students WHERE id = ?",
            (student_db_id,),
        ).fetchone()

        conn.commit()

    if not student_row:
        raise HTTPException(500, "Unable to load student account.")

    return {"token": token, "student": _serialize_student(student_row)}


def list_students(limit: int = 100) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, full_name, email, mobile_number, student_id, department, year_level, created_at, last_login_at
            FROM students
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [_serialize_student(row) for row in rows]


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


@app.post("/api/auth/register/request-otp")
def request_register_otp(request: RequestRegisterOtpRequest) -> Dict[str, str]:
    return request_registration_otp(request)


@app.post("/api/auth/register/verify-otp")
def verify_register_otp(request: VerifyRegisterOtpRequest) -> Dict[str, Any]:
    return verify_registration_otp_and_register(request)


@app.post("/api/auth/register")
def register_student_account(request: RegisterRequest) -> Dict[str, Any]:
    return register_student(request)


@app.post("/api/auth/login")
def login_student_account(request: LoginRequest) -> Dict[str, Any]:
    return login_student(request)


@app.get("/api/auth/me")
def current_student_profile(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    return _student_from_token(authorization)


@app.get("/api/students")
def students_directory(
    limit: int = 100,
    authorization: Optional[str] = Header(default=None),
) -> List[Dict[str, Any]]:
    _student_from_token(authorization)
    return list_students(limit=limit)


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


@app.post("/api/translate")
def translate_notes(request: TranslateNotesRequest) -> Dict[str, Any]:
    if not isinstance(request.notes, dict):
        raise HTTPException(400, "Invalid notes payload.")

    source_notes = request.notes
    normalized_source = normalize_result(source_notes)
    if not normalized_source.get("topics"):
        raise HTTPException(400, "No notes content found to translate.")

    target_language_code = _coerce_language_code(request.target_language)
    target_language_name = _coerce_language_name(
        request.target_language_name or "", target_language_code
    )

    translated = translate_notes_payload(
        normalized_source, target_language_code, target_language_name
    )

    created_at = str(source_notes.get("created_at") or "").strip()
    if not created_at:
        created_at = datetime.utcnow().isoformat() + "Z"

    result: Dict[str, Any] = {
        "video_url": str(source_notes.get("video_url") or "").strip(),
        "video_title": str(source_notes.get("video_title") or "").strip() or "YouTube Video",
        "created_at": created_at,
        "language": target_language_code,
        "language_name": target_language_name,
        **translated,
    }

    if "id" in source_notes:
        try:
            result["id"] = int(source_notes["id"])
        except (TypeError, ValueError):
            pass

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
