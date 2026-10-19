"""
Input and file validation utilities for ScamShield AI.

Enforces strict bounds on text length, URL formatting, JSON payload structure,
and uploaded image files to prevent malformed inputs, DoS via oversized payloads,
and arbitrary filesystem access.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

# Configuration constants with safe defaults
DEFAULT_MAX_TEXT_LENGTH: int = int(os.environ.get("SCAMSHIELD_MAX_TEXT_LENGTH", "25000"))
DEFAULT_MIN_TEXT_LENGTH: int = int(os.environ.get("SCAMSHIELD_MIN_TEXT_LENGTH", "10"))
DEFAULT_MAX_URL_LENGTH: int = 2048
DEFAULT_MAX_IMAGE_BYTES: int = int(
    os.environ.get("SCAMSHIELD_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))  # 5 MB
)
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
ALLOWED_IMAGE_FORMATS: Set[str] = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
ALLOWED_URL_SCHEMES: Set[str] = {"http", "https"}

# Control character regex (excluding standard whitespace: \t, \n, \r)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ValidationError(Exception):
    """Raised when an API request payload or uploaded file fails validation."""

    def __init__(self, message: str, status_code: int = 400, error_code: str = "VALIDATION_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }


@dataclass(frozen=True)
class ValidatedAnalyzeRequest:
    """Sanitized and validated parameters for text analysis."""

    job_text: str
    source_url: Optional[str]
    char_count: int
    word_count: int


def sanitize_text(text: str) -> str:
    """Remove null bytes and non-printable control characters while normalizing line breaks."""
    cleaned = CONTROL_CHAR_PATTERN.sub(" ", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned.strip()


def validate_job_text(
    text: Any,
    max_length: int = DEFAULT_MAX_TEXT_LENGTH,
    min_length: int = DEFAULT_MIN_TEXT_LENGTH,
) -> Tuple[str, int, int]:
    """
    Validate and sanitize submitted job description text.

    Returns:
        Tuple of (sanitized_text, char_count, word_count)

    Raises:
        ValidationError: If text is missing, not a string, empty, too short, or too long.
    """
    if text is None:
        raise ValidationError(
            "Missing required field 'job_text'. Please provide the job description or offer text to analyze.",
            status_code=400,
            error_code="MISSING_TEXT",
        )

    if not isinstance(text, str):
        raise ValidationError(
            "Field 'job_text' must be a string.",
            status_code=400,
            error_code="INVALID_TEXT_TYPE",
        )

    cleaned = sanitize_text(text)
    if not cleaned:
        raise ValidationError(
            "Submitted job description is empty or contains only whitespace.",
            status_code=400,
            error_code="EMPTY_TEXT",
        )

    char_count = len(cleaned)
    if char_count < min_length:
        raise ValidationError(
            f"Submitted job text is too short ({char_count} characters). "
            f"Please provide at least {min_length} characters of the recruitment listing or message.",
            status_code=400,
            error_code="TEXT_TOO_SHORT",
        )

    if char_count > max_length:
        raise ValidationError(
            f"Submitted job text exceeds the maximum allowed length of {max_length:,} characters "
            f"(received {char_count:,} characters).",
            status_code=413,
            error_code="TEXT_TOO_LONG",
        )

    words = cleaned.split()
    word_count = len(words)
    return cleaned, char_count, word_count


def validate_source_url(url: Any, max_length: int = DEFAULT_MAX_URL_LENGTH) -> Optional[str]:
    """
    Validate optional source URL metadata.

    Never opens or fetches the URL; only validates syntax and safe scheme (`http` or `https`).
    Rejects local file URIs (`file://`), javascript URIs, or path traversal attempts.
    """
    if url is None or url == "":
        return None

    if not isinstance(url, str):
        raise ValidationError(
            "Optional field 'source_url' must be a string if provided.",
            status_code=400,
            error_code="INVALID_URL_TYPE",
        )

    cleaned_url = url.strip()
    if not cleaned_url:
        return None

    if "\r" in cleaned_url or "\n" in cleaned_url or CONTROL_CHAR_PATTERN.search(cleaned_url):
        raise ValidationError(
            "Field 'source_url' must not contain line breaks or non-printable control characters.",
            status_code=400,
            error_code="INVALID_URL_FORMAT",
        )

    if len(cleaned_url) > max_length:
        raise ValidationError(
            f"Field 'source_url' exceeds the maximum allowed length of {max_length} characters.",
            status_code=400,
            error_code="URL_TOO_LONG",
        )

    parsed = urlparse(cleaned_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_URL_SCHEMES or not parsed.netloc:
        raise ValidationError(
            "Field 'source_url' must be a valid HTTP or HTTPS URL (e.g., https://example.com/jobs/123).",
            status_code=400,
            error_code="INVALID_URL_SCHEME",
        )

    return cleaned_url


def validate_analyze_payload(
    payload: Any,
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
) -> ValidatedAnalyzeRequest:
    """
    Validate the JSON payload for POST /api/analyze.

    Accepts `job_text` (primary) or `text` (alias) and optional `source_url` (or `url` alias).
    Rejects unexpected path parameters that could attempt local file loading.
    """
    if not isinstance(payload, dict):
        raise ValidationError(
            "Request body must be a valid JSON object.",
            status_code=400,
            error_code="INVALID_JSON_OBJECT",
        )

    # Guard against attempts to pass file paths instead of text content
    forbidden_keys = {"file_path", "filepath", "local_path", "model_path", "dataset_path"}
    present_forbidden = forbidden_keys.intersection(payload.keys())
    if present_forbidden:
        raise ValidationError(
            f"Unsupported or restricted field(s) in request payload: {', '.join(sorted(present_forbidden))}.",
            status_code=400,
            error_code="RESTRICTED_FIELD",
        )

    raw_text = payload.get("job_text") if "job_text" in payload else payload.get("text")
    raw_url = payload.get("source_url") if "source_url" in payload else payload.get("url")

    job_text, char_count, word_count = validate_job_text(
        raw_text,
        max_length=max_text_length,
        min_length=min_text_length,
    )
    source_url = validate_source_url(raw_url)

    return ValidatedAnalyzeRequest(
        job_text=job_text,
        source_url=source_url,
        char_count=char_count,
        word_count=word_count,
    )


def validate_uploaded_image(
    filename: Optional[str],
    file_bytes: bytes,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> Image.Image:
    """
    Validate an uploaded screenshot image for OCR processing.

    Checks:
    1. Filename is present and has an allowed image extension.
    2. File byte size is non-empty and within `max_bytes`.
    3. Image header and payload can be decoded safely by Pillow and match allowed formats.
    """
    if not filename or not isinstance(filename, str) or not filename.strip():
        raise ValidationError(
            "No image filename provided in upload request.",
            status_code=400,
            error_code="MISSING_FILENAME",
        )

    # Extract only the base filename to prevent directory traversal
    safe_name = os.path.basename(filename.strip())
    _, ext = os.path.splitext(safe_name.lower())
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise ValidationError(
            f"Unsupported file extension '{ext or '(none)'}'. Allowed image types: {allowed_list}.",
            status_code=400,
            error_code="UNSUPPORTED_FILE_TYPE",
        )

    if not file_bytes or len(file_bytes) == 0:
        raise ValidationError(
            "Uploaded image file is empty (0 bytes).",
            status_code=400,
            error_code="EMPTY_IMAGE_FILE",
        )

    if len(file_bytes) > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        raise ValidationError(
            f"Uploaded image exceeds the maximum file size limit of {max_mb:.1f} MB.",
            status_code=413,
            error_code="IMAGE_TOO_LARGE",
        )

    try:
        # First pass: verify image integrity
        verify_stream = io.BytesIO(file_bytes)
        with Image.open(verify_stream) as img_verify:
            img_verify.verify()

        # Second pass: re-open header and inspect format/dimensions BEFORE decompressing pixels
        load_stream = io.BytesIO(file_bytes)
        image = Image.open(load_stream)

        img_format = (image.format or "").upper()
        if img_format not in ALLOWED_IMAGE_FORMATS:
            raise ValidationError(
                f"Unsupported image format '{img_format or 'UNKNOWN'}'. Please upload PNG, JPEG, WEBP, BMP, or TIFF.",
                status_code=400,
                error_code="UNSUPPORTED_IMAGE_FORMAT",
            )

        # Guard against decompression bomb / extreme dimensions before allocating pixel buffer
        width, height = image.size
        if width <= 0 or height <= 0 or (width * height) > 25_000_000:
            raise ValidationError(
                f"Image dimensions ({width}x{height}) are invalid or exceed safe processing limits.",
                status_code=400,
                error_code="INVALID_IMAGE_DIMENSIONS",
            )

        # Decompress pixels only after format and dimensions are verified safe
        image.load()
    except ValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ValidationError(
            "The uploaded file could not be read as a valid image. Please upload a clear PNG, JPG, or WEBP screenshot.",
            status_code=400,
            error_code="UNREADABLE_IMAGE",
        ) from exc

    return image
