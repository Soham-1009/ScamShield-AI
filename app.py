"""
Flask Backend API for ScamShield AI (`app.py`).

Endpoints:
- `GET /`                  : Service status and quick API overview page.
- `GET /api/health`        : Returns application status, ML model availability,
                             and OCR availability (with zero LLM dependencies).
- `POST /api/analyze`      : Validates and analyzes recruitment text, returning
                             rule-based risk assessment, ML prediction (when available),
                             experimental triage score, matched phrases, deterministic
                             explanations, and verification steps.
- `POST /api/analyze-image`: Validates an uploaded image screenshot, extracts text
                             via Tesseract OCR, and runs the same analysis pipeline.

Security & Privacy:
- Local processing by default; never stores or logs user-submitted job text or screenshots.
- Strict input validation, file type/size checks, and no arbitrary filesystem access.
- Configurable CORS for the Chrome Extension (`chrome-extension://*` and localhost),
  including full preflight `OPTIONS` and Private Network Access support.
- Sliding-window in-memory rate limiting per client IP.
"""

from __future__ import annotations

import collections
import os
import re
import threading
import time
from typing import Any, Deque, Dict, Optional, Tuple

from flask import Flask, jsonify, make_response, request
from werkzeug.middleware.proxy_fix import ProxyFix

from scamshield import __version__
from scamshield.explanation_service import DeterministicExplanationService
from scamshield.ml_detector import MLDetector
from scamshield.ocr_service import OCRError, OCRService
from scamshield.risk_rules import RiskRuleEngine, compute_experimental_triage_score
from scamshield.validation import (
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_TEXT_LENGTH,
    DEFAULT_MIN_TEXT_LENGTH,
    ValidationError,
    validate_analyze_payload,
    validate_job_text,
    validate_source_url,
    validate_uploaded_image,
)

# Rate-limiting defaults (requests per 60-second sliding window per IP)
DEFAULT_RATE_LIMIT_PER_MINUTE = int(os.environ.get("SCAMSHIELD_RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_WINDOW_SECONDS = 60.0

ALLOWED_ORIGIN_PATTERNS = [
    re.compile(r"^chrome-extension://[a-z]{32}$"),
    re.compile(r"^http://localhost(?::\d+)?$"),
    re.compile(r"^http://127\.0\.0\.1(?::\d+)?$"),
]


class SlidingWindowRateLimiter:
    """Thread-safe, bounded in-memory per-IP sliding-window rate limiter."""

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
        window_seconds: float = 60.0,
        max_clients: int = 10_000,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_clients = max(1, max_clients)
        self._requests: collections.OrderedDict[str, Deque[float]] = collections.OrderedDict()
        self._calls_since_sweep = 0
        self._last_sweep_at = 0.0
        self._lock = threading.Lock()

    def is_allowed(self, client_key: str) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        if self.max_requests <= 0:
            return True, 0

        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # Periodic cleanup avoids scanning every tracked client on each request.
            self._calls_since_sweep += 1
            should_sweep = self._calls_since_sweep >= 256 or (
                len(self._requests) > 1000 and now - self._last_sweep_at >= 1.0
            )
            if should_sweep:
                stale_keys = [k for k, dq in self._requests.items() if not dq or dq[-1] <= cutoff]
                for k in stale_keys:
                    del self._requests[k]
                self._calls_since_sweep = 0
                self._last_sweep_at = now

            dq = self._requests.get(client_key)
            if dq is None:
                # Cap memory when clients rotate source addresses. Evict the least
                # recently seen key when the bounded table is full.
                if len(self._requests) >= self.max_clients:
                    self._requests.popitem(last=False)
                dq = collections.deque()
                self._requests[client_key] = dq
            else:
                self._requests.move_to_end(client_key)

            while dq and dq[0] <= cutoff:
                dq.popleft()

            if len(dq) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - dq[0])))
                return False, retry_after

            dq.append(now)
            return True, 0


def _build_analysis_response(
    job_text: str,
    char_count: int,
    word_count: int,
    source_url: Optional[str],
    rule_engine: RiskRuleEngine,
    ml_detector: MLDetector,
    explanation_service: DeterministicExplanationService,
    ocr_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the core ScamShield AI analysis pipeline on validated text.
    """
    rule_evaluation = rule_engine.evaluate(job_text)
    rule_assessment = rule_evaluation["rule_based_assessment"]
    triggered_rules = rule_evaluation["triggered_rules"]
    exact_phrases = rule_evaluation["exact_matched_phrases"]

    if ml_detector.is_available and word_count < 20:
        ml_prediction = {
            "available": False,
            "insufficient_text": True,
            "model_loaded": True,
            "predicted_label": None,
            "predicted_class": None,
            "fraud_probability_score": None,
            "fraud_score_pct": None,
            "is_calibrated_probability": False,
            "model_version": None,
            "metadata": None,
            "disclaimer": "A short excerpt is not enough context for a useful statistical model signal.",
            "status_message": "ML scoring withheld because fewer than 20 words were submitted.",
        }
    else:
        ml_prediction = ml_detector.predict(job_text)
    experimental_triage = compute_experimental_triage_score(rule_assessment, ml_prediction)

    explanation = explanation_service.generate_explanation(
        rule_evaluation=rule_evaluation,
        ml_prediction=ml_prediction,
        input_word_count=word_count,
    )

    limitations = [
        "Risk assessments are automated estimates and cannot definitively prove whether an offer is fraudulent or legitimate.",
        "The absence of triggered warning rules does not confirm that an employer or job offer is genuine.",
    ]
    if ml_prediction.get("insufficient_text"):
        limitations.append(
            "Text is brief (< 20 words); the ML score was withheld because the excerpt lacks context."
        )
    elif not ml_prediction.get("available"):
        limitations.append(
            "Trained ML model is not currently loaded; analysis reflects rule-based detection only."
        )
    if word_count < 25:
        limitations.append(
            "Submitted text is brief (< 25 words); short snippets may lack sufficient context."
        )
    if ocr_metadata is not None:
        limitations.append(
            "Text was extracted via OCR from an image; recognition errors may affect pattern matching."
        )

    response: Dict[str, Any] = {
        "rule_based_assessment": rule_assessment,
        "ml_prediction": ml_prediction,
        "experimental_triage": experimental_triage,
        "triggered_rules": triggered_rules,
        "exact_matched_phrases": exact_phrases,
        "explanation": explanation,
        "safety_recommendations": explanation["verification_recommendations"],
        "processing_status": {
            "status": "completed",
            "input_char_count": char_count,
            "input_word_count": word_count,
            "source_url": source_url,
            "ml_used": bool(ml_prediction.get("available")),
            "ocr_used": ocr_metadata is not None,
            "limitations": limitations,
            "privacy_notice": (
                "Submitted text and screenshots are processed in memory only and are not stored or logged."
            ),
        },
    }

    if ocr_metadata is not None:
        response["ocr_extraction"] = ocr_metadata

    return response


def create_app(
    model_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    tesseract_cmd: Optional[str] = None,
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> Flask:
    """Application factory for the ScamShield AI Flask backend."""
    app = Flask(__name__)
    # Standard reverse-proxy header handling (trusted 1 hop behind Cloud Run / GFE / Nginx)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    # Cap overall HTTP request body to max_image_bytes + 512 KB overhead
    app.config["MAX_CONTENT_LENGTH"] = max_image_bytes + (512 * 1024)

    rule_engine = RiskRuleEngine()
    ml_detector = MLDetector(model_path=model_path, metadata_path=metadata_path)
    explanation_service = DeterministicExplanationService()
    ocr_service = OCRService(tesseract_cmd=tesseract_cmd)
    rate_limiter = SlidingWindowRateLimiter(max_requests=rate_limit_per_minute)

    def _check_rate_limit() -> Optional[Any]:
        client_ip = request.remote_addr or "127.0.0.1"
        allowed, retry_after = rate_limiter.is_allowed(client_ip)
        if not allowed:
            resp = make_response(
                jsonify(
                    {
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Please wait {retry_after} second(s) before retrying.",
                        "retry_after_seconds": retry_after,
                        "status_code": 429,
                    }
                ),
                429,
            )
            resp.headers["Retry-After"] = str(retry_after)
            return resp
        return None

    @app.after_request
    def apply_cors_and_security_headers(response: Any) -> Any:
        origin = request.headers.get("Origin", "")
        custom_allowed = os.environ.get("SCAMSHIELD_CORS_ORIGIN", "")
        is_local_origin = any(pat.match(origin) for pat in ALLOWED_ORIGIN_PATTERNS[1:])
        is_extension_origin = bool(ALLOWED_ORIGIN_PATTERNS[0].match(origin))
        origin_allowed = bool(origin) and (
            origin == custom_allowed
            or is_local_origin
            or (not custom_allowed and is_extension_origin)
        )
        if origin_allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"

        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
        if origin_allowed and (is_local_origin or is_extension_origin):
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Access-Control-Max-Age"] = "3600"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(413)
    def handle_payload_too_large(_err: Any) -> Any:
        return make_response(
            jsonify(
                {
                    "error": "PAYLOAD_TOO_LARGE",
                    "message": "Request payload exceeds the maximum permitted size.",
                    "status_code": 413,
                }
            ),
            413,
        )

    @app.errorhandler(404)
    def handle_not_found(_err: Any) -> Any:
        return make_response(
            jsonify(
                {
                    "error": "NOT_FOUND",
                    "message": "Requested API endpoint does not exist.",
                    "status_code": 404,
                }
            ),
            404,
        )

    @app.errorhandler(405)
    def handle_method_not_allowed(_err: Any) -> Any:
        return make_response(
            jsonify(
                {
                    "error": "METHOD_NOT_ALLOWED",
                    "message": f"HTTP method '{request.method}' is not allowed for this endpoint.",
                    "status_code": 405,
                }
            ),
            405,
        )

    @app.errorhandler(500)
    def handle_internal_server_error(_err: Any) -> Any:
        # Keep API errors machine-readable without exposing exception details or paths.
        return make_response(
            jsonify(
                {
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "The request could not be completed due to an internal server error.",
                    "status_code": 500,
                }
            ),
            500,
        )

    @app.route("/", methods=["GET", "OPTIONS"])
    def index_root() -> Any:
        if request.method == "OPTIONS":
            return make_response("", 204)
        return jsonify(
            {
                "service": "ScamShield AI Backend",
                "version": __version__,
                "status": "running",
                "endpoints": {
                    "health": "GET /api/health",
                    "analyze_text": "POST /api/analyze",
                    "analyze_image": "POST /api/analyze-image",
                },
                "message": "ScamShield AI API is online. Use the Chrome Extension popup to scan pages or analyze job offers.",
            }
        )

    @app.route("/favicon.ico", methods=["GET"])
    def favicon() -> Any:
        return make_response("", 204)

    @app.route("/api/health", methods=["GET", "OPTIONS"])
    def health_check() -> Any:
        if request.method == "OPTIONS":
            return make_response("", 204)

        rate_err = _check_rate_limit()
        if rate_err is not None:
            return rate_err

        ml_status = ml_detector.get_health_status()
        ocr_status = ocr_service.get_health_status()

        return jsonify(
            {
                "status": "ok",
                "service": "ScamShield AI Backend",
                "version": __version__,
                "ml_model": ml_status,
                "ocr": ocr_status,
                "rule_engine": {
                    "available": True,
                    "rule_count": len(rule_engine.rules),
                },
            }
        )

    @app.route("/api/analyze", methods=["POST", "OPTIONS"])
    def analyze_job_text() -> Any:
        # Handle CORS preflight OPTIONS request before inspecting JSON Content-Type
        if request.method == "OPTIONS":
            return make_response("", 204)

        rate_err = _check_rate_limit()
        if rate_err is not None:
            return rate_err

        if not request.is_json:
            return make_response(
                jsonify(
                    {
                        "error": "INVALID_CONTENT_TYPE",
                        "message": "Request Content-Type must be 'application/json'.",
                        "status_code": 400,
                    }
                ),
                400,
            )

        raw_body = request.get_data()
        if not raw_body or not raw_body.strip():
            return make_response(
                jsonify(
                    {
                        "error": "EMPTY_PAYLOAD",
                        "message": "Request body is empty. Provide a JSON object with 'job_text'.",
                        "status_code": 400,
                    }
                ),
                400,
            )

        payload = request.get_json(silent=True)
        if payload is None:
            return make_response(
                jsonify(
                    {
                        "error": "MALFORMED_JSON",
                        "message": "Request body contains malformed JSON that could not be parsed.",
                        "status_code": 400,
                    }
                ),
                400,
            )

        try:
            validated = validate_analyze_payload(
                payload=payload,
                max_text_length=max_text_length,
                min_text_length=min_text_length,
            )
        except ValidationError as exc:
            return make_response(jsonify(exc.to_dict()), exc.status_code)

        result = _build_analysis_response(
            job_text=validated.job_text,
            char_count=validated.char_count,
            word_count=validated.word_count,
            source_url=validated.source_url,
            rule_engine=rule_engine,
            ml_detector=ml_detector,
            explanation_service=explanation_service,
            ocr_metadata=None,
        )
        return jsonify(result)

    @app.route("/api/analyze-image", methods=["POST", "OPTIONS"])
    def analyze_job_image() -> Any:
        # Handle CORS preflight OPTIONS request immediately
        if request.method == "OPTIONS":
            return make_response("", 204)

        rate_err = _check_rate_limit()
        if rate_err is not None:
            return rate_err

        uploaded = request.files.get("image") or request.files.get("file")
        if uploaded is None:
            return make_response(
                jsonify(
                    {
                        "error": "MISSING_IMAGE_FILE",
                        "message": "No image file uploaded. Submit a multipart/form-data request with field 'image'.",
                        "status_code": 400,
                    }
                ),
                400,
            )

        filename = getattr(uploaded, "filename", None)
        file_bytes = uploaded.read()

        try:
            pil_image = validate_uploaded_image(
                filename=filename,
                file_bytes=file_bytes,
                max_bytes=max_image_bytes,
            )
            raw_url = request.form.get("source_url") or request.form.get("url")
            source_url = validate_source_url(raw_url)
        except ValidationError as exc:
            return make_response(jsonify(exc.to_dict()), exc.status_code)

        try:
            ocr_data = ocr_service.extract_text_from_image(pil_image)
        except OCRError as exc:
            return make_response(jsonify(exc.to_dict()), exc.status_code)
        finally:
            pil_image.close()

        try:
            cleaned_text, char_count, word_count = validate_job_text(
                ocr_data["extracted_text"],
                max_length=max_text_length,
                min_length=min_text_length,
            )
        except ValidationError as exc:
            return make_response(jsonify(exc.to_dict()), exc.status_code)

        result = _build_analysis_response(
            job_text=cleaned_text,
            char_count=char_count,
            word_count=word_count,
            source_url=source_url,
            rule_engine=rule_engine,
            ml_detector=ml_detector,
            explanation_service=explanation_service,
            ocr_metadata=ocr_data,
        )
        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("SCAMSHIELD_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("SCAMSHIELD_PORT", "5000")))
    debug_flag = os.environ.get("SCAMSHIELD_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug_flag)
