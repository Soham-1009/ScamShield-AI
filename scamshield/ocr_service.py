"""
OCR text extraction service for ScamShield AI (`scamshield/ocr_service.py`).

Uses Tesseract OCR via `pytesseract` to extract recruitment text from uploaded
screenshots. Validates Tesseract binary availability and handles missing executables
or unreadable/blank screenshots with clear, actionable error messages.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

from PIL import Image, ImageOps

from scamshield.validation import sanitize_text

TESSERACT_INSTALL_INSTRUCTIONS = (
    "Tesseract OCR executable was not found on this system. "
    "To enable screenshot analysis:\n"
    "- Windows (winget): winget install --id UB-Mannheim.TesseractOCR\n"
    "- Windows (choco): choco install tesseract\n"
    "- Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
    "- macOS (Homebrew): brew install tesseract\n"
    "If installed in a custom path, set the TESSERACT_CMD environment variable "
    "(e.g., C:\\Program Files\\Tesseract-OCR\\tesseract.exe)."
)


class OCRError(Exception):
    """Raised when OCR extraction cannot be completed due to missing binary or unreadable text."""

    def __init__(
        self,
        message: str,
        status_code: int = 503,
        error_code: str = "OCR_ERROR",
    ) -> None:
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


STANDARD_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
]


class OCRService:
    """
    Wrapper around pytesseract for extracting text from job offer screenshots.
    """

    def __init__(self, tesseract_cmd: Optional[str] = None) -> None:
        self.tesseract_cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD")

    def _configure_pytesseract(self) -> Any:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise OCRError(
                "Python package 'pytesseract' is not installed. Run 'pip install -r requirements.txt'.",
                status_code=503,
                error_code="PYTESSERACT_MISSING",
            ) from exc

        cmd_to_check = self.tesseract_cmd or getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        resolved: Optional[str] = None

        if self.tesseract_cmd:
            # If an explicit path was passed, check that specific path directly
            if os.path.isfile(self.tesseract_cmd) and os.access(self.tesseract_cmd, os.X_OK):
                resolved = self.tesseract_cmd
            else:
                resolved = shutil.which(self.tesseract_cmd)
        else:
            # Check default command via PATH, then check standard install locations
            resolved = shutil.which(cmd_to_check)
            if not resolved:
                for candidate in STANDARD_TESSERACT_PATHS:
                    if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                        resolved = candidate
                        break

        if not resolved:
            raise OCRError(
                TESSERACT_INSTALL_INSTRUCTIONS,
                status_code=503,
                error_code="TESSERACT_NOT_INSTALLED",
            )

        pytesseract.pytesseract.tesseract_cmd = resolved
        return pytesseract

    def get_health_status(self) -> Dict[str, Any]:
        """Return OCR availability and engine metadata for GET /api/health."""
        try:
            pytesseract = self._configure_pytesseract()
            version = str(pytesseract.get_tesseract_version())
            return {
                "available": True,
                "engine": "tesseract",
                "version": version,
                "status_message": f"Tesseract OCR available (v{version}).",
            }
        except OCRError as exc:
            return {
                "available": False,
                "engine": "tesseract",
                "version": None,
                "status_message": exc.message,
            }
        except Exception as exc:
            return {
                "available": False,
                "engine": "tesseract",
                "version": None,
                "status_message": f"Tesseract OCR unavailable: {exc}",
            }

    def extract_text_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Preprocess an image and extract text using Tesseract OCR.

        Raises:
            OCRError: If Tesseract is unavailable, fails, or extracts no readable text.
        """
        pytesseract = self._configure_pytesseract()

        try:
            # Convert to RGB/grayscale and upscale low-resolution screenshots for better character segmentation
            processed = image.convert("RGB")
            grayscale = ImageOps.grayscale(processed)
            width, height = grayscale.size
            if height < 400 or width < 600:
                resample_filter = getattr(Image, "Resampling", Image).BICUBIC
                grayscale = grayscale.resize((width * 2, height * 2), resample=resample_filter)
            raw_text = pytesseract.image_to_string(grayscale)
        except Exception as exc:
            msg = str(exc)
            if "not installed" in msg.lower() or "not in your path" in msg.lower():
                raise OCRError(
                    TESSERACT_INSTALL_INSTRUCTIONS,
                    status_code=503,
                    error_code="TESSERACT_NOT_INSTALLED",
                ) from exc
            raise OCRError(
                f"Failed to run OCR on the uploaded image: {msg}",
                status_code=422,
                error_code="OCR_PROCESSING_FAILED",
            ) from exc

        cleaned_text = sanitize_text(raw_text or "")
        if not cleaned_text or len(cleaned_text) < 10:
            raise OCRError(
                "OCR could not extract sufficient readable text from the uploaded image "
                f"(extracted {len(cleaned_text)} character(s)). Please upload a clearer, higher-resolution "
                "screenshot of the job offer or paste the text manually.",
                status_code=422,
                error_code="OCR_NO_READABLE_TEXT",
            )

        return {
            "extracted_text": cleaned_text,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "engine": "tesseract",
            "confidence_note": (
                "Text was automatically extracted from the uploaded image using Tesseract OCR. "
                "Formatting artifacts or character recognition errors may affect rule matching; "
                "review the extracted text carefully."
            ),
        }
