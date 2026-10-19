"""
Unit tests for input, URL, payload, and image upload validation (`tests/test_validation.py`).
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "ScamShieldAI"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scamshield.validation import (
    ValidationError,
    validate_analyze_payload,
    validate_job_text,
    validate_source_url,
    validate_uploaded_image,
)


class TestValidation(unittest.TestCase):
    """Tests for strict request validation and security boundaries."""

    def test_empty_and_whitespace_input_rejected(self) -> None:
        """Synthetic case 5: Empty input and whitespace-only input."""
        with self.assertRaises(ValidationError) as ctx_none:
            validate_job_text(None)
        self.assertEqual(ctx_none.exception.error_code, "MISSING_TEXT")
        self.assertEqual(ctx_none.exception.status_code, 400)

        with self.assertRaises(ValidationError) as ctx_empty:
            validate_job_text("")
        self.assertEqual(ctx_empty.exception.error_code, "EMPTY_TEXT")

        with self.assertRaises(ValidationError) as ctx_spaces:
            validate_job_text("   \n\t  ")
        self.assertEqual(ctx_spaces.exception.error_code, "EMPTY_TEXT")

    def test_excessively_long_input_rejected(self) -> None:
        """Synthetic case 6: Excessively long input exceeding max character limit."""
        oversized = "Software Engineer role description. " * 1000  # ~36,000 chars > 25,000 limit
        with self.assertRaises(ValidationError) as ctx:
            validate_job_text(oversized, max_length=25000)
        self.assertEqual(ctx.exception.error_code, "TEXT_TOO_LONG")
        self.assertEqual(ctx.exception.status_code, 413)

    def test_too_short_input_rejected(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            validate_job_text("Hi", min_length=10)
        self.assertEqual(ctx.exception.error_code, "TEXT_TOO_SHORT")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_source_url_validation_blocks_unsafe_schemes(self) -> None:
        self.assertIsNone(validate_source_url(None))
        self.assertIsNone(validate_source_url(""))
        self.assertEqual(
            validate_source_url("https://careers.example.com/jobs/42"),
            "https://careers.example.com/jobs/42",
        )

        for unsafe_url in ("file:///etc/passwd", "javascript:alert(1)", "ftp://files.example.com/a"):
            with self.assertRaises(ValidationError) as ctx:
                validate_source_url(unsafe_url)
            self.assertEqual(ctx.exception.error_code, "INVALID_URL_SCHEME")

    def test_payload_blocks_arbitrary_filepath_fields(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            validate_analyze_payload(
                {
                    "job_text": "Valid job description text for software internship.",
                    "file_path": "/etc/shadow",
                }
            )
        self.assertEqual(ctx.exception.error_code, "RESTRICTED_FIELD")

    def test_image_upload_validation(self) -> None:
        """Synthetic case 8 (validation layer): Unreadable, empty, or unsupported image files."""
        # Unsupported file extension
        with self.assertRaises(ValidationError) as ctx_ext:
            validate_uploaded_image("offer.exe", b"MZ-fake-binary")
        self.assertEqual(ctx_ext.exception.error_code, "UNSUPPORTED_FILE_TYPE")

        # Empty byte stream
        with self.assertRaises(ValidationError) as ctx_empty:
            validate_uploaded_image("screenshot.png", b"")
        self.assertEqual(ctx_empty.exception.error_code, "EMPTY_IMAGE_FILE")

        # Corrupted / unreadable image bytes
        with self.assertRaises(ValidationError) as ctx_corrupt:
            validate_uploaded_image("screenshot.png", b"not-a-valid-png-header-at-all")
        self.assertEqual(ctx_corrupt.exception.error_code, "UNREADABLE_IMAGE")

        # Oversized image file
        with self.assertRaises(ValidationError) as ctx_large:
            validate_uploaded_image("screenshot.png", b"x" * 2048, max_bytes=1024)
        self.assertEqual(ctx_large.exception.error_code, "IMAGE_TOO_LARGE")
        self.assertEqual(ctx_large.exception.status_code, 413)

        # Valid PNG image passes validation
        buf = io.BytesIO()
        img = Image.new("RGB", (120, 60), color=(255, 255, 255))
        img.save(buf, format="PNG")
        validated_img = validate_uploaded_image("offer.png", buf.getvalue())
        self.assertEqual(validated_img.size, (120, 60))


if __name__ == "__main__":
    unittest.main()
