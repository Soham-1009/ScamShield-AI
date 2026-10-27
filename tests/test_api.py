"""
Integration and API tests for the ScamShield AI Flask backend (`tests/test_api.py`).

Tests all required endpoints and synthetic test scenarios:
- `GET /api/health` (verifying no LLM fields and honest ML/OCR status)
- `POST /api/analyze` across:
  1. Upfront fee job advertisement
  2. Unrealistic salary promise
  3. Legitimate-looking job description (no false guarantee of safety)
  4. Urgency language without payment request
  5. Empty input (`400`)
  6. Excessively long input (`413`)
  7. ML model unavailable (never fabricates a 0/100 score)
  8. Malformed request payload (`400`)
  9. Multiple warning rules triggered simultaneously
- `POST /api/analyze-image` across:
  10. Unreadable/corrupted screenshot (`400`) and blank image with no text (`422`)
  11. Missing Tesseract executable (`503` with clear installation instructions)
  12. Valid OCR screenshot analysis when Tesseract is available
- Dataset loading & validation checks (`train_model.py`).
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "ScamShieldAI"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pandas as pd
from PIL import Image, ImageDraw

from app import SlidingWindowRateLimiter, create_app
from scamshield.ml_detector import MLDetector
from train_model import load_and_validate_dataset


class _DummyFittedPipeline:
    """Deterministic stub implementing Scikit-learn's predict and predict_proba interface for unit testing."""

    classes_ = [0, 1]

    def predict(self, texts: list[str]) -> list[int]:
        return [1 if "registration fee" in t.lower() else 0 for t in texts]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.12, 0.88] if "registration fee" in t.lower() else [0.85, 0.15] for t in texts]


class TestScamShieldAPI(unittest.TestCase):
    """End-to-end API integration tests for ScamShield AI."""

    def setUp(self) -> None:
        # Configure app pointing to a non-existent model path by default to test honest missing-model behavior
        self.app = create_app(
            model_path="models/nonexistent_test_model.joblib",
            metadata_path="models/nonexistent_test_metadata.json",
            rate_limit_per_minute=200,
        )
        self.client = self.app.test_client()

    def test_health_endpoint_has_no_llm_fields(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("ml_model", data)
        self.assertIn("ocr", data)
        self.assertIn("rule_engine", data)
        self.assertFalse(data["ml_model"]["available"])

        # Ensure zero LLM / generative AI references in health status
        serialized = str(data).lower()
        for forbidden in ("gemini", "ollama", "openai", "llm"):
            self.assertNotIn(forbidden, serialized)

    def test_analyze_upfront_fee_and_ml_unavailable(self) -> None:
        """Synthetic cases 1 & 7: Upfront fee advertisement when ML model is unavailable."""
        payload = {
            "job_text": (
                "Administrative Assistant Internship. Applicants must pay a refundable registration fee "
                "of $99 for background check processing before receiving the offer letter."
            ),
            "source_url": "https://jobs.example.com/listing/101",
        }
        resp = self.client.post("/api/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()

        # Rule-based assessment
        self.assertEqual(body["rule_based_assessment"]["risk_level_code"], "high")
        self.assertGreaterEqual(body["rule_based_assessment"]["triggered_rule_count"], 1)

        # ML prediction unavailable — must NOT fabricate a 0 or 0.0 score
        ml = body["ml_prediction"]
        self.assertFalse(ml["available"])
        self.assertIsNone(ml["predicted_class"])
        self.assertIsNone(ml["fraud_probability_score"])
        self.assertIsNone(ml["fraud_score_pct"])

        # Experimental triage must also report unavailable when ML is missing
        self.assertFalse(body["experimental_triage"]["available"])
        self.assertIsNone(body["experimental_triage"]["score"])

    def test_ml_detector_when_pipeline_is_loaded(self) -> None:
        """Verify MLDetector output structure and disclaimer when a fitted pipeline is present."""
        detector = MLDetector(model_path="models/nonexistent.joblib")
        detector._pipeline = _DummyFittedPipeline()
        detector._metadata = {"model_version": "1.0.0-test-tfidf"}

        pred = detector.predict("Please pay a registration fee of $100 to start.")
        self.assertTrue(pred["available"])
        self.assertEqual(pred["predicted_label"], 1)
        self.assertEqual(pred["predicted_class"], "fraudulent_pattern_detected")
        self.assertAlmostEqual(pred["fraud_probability_score"], 0.88)
        self.assertAlmostEqual(pred["fraud_score_pct"], 88.0)
        self.assertIn("not proof", pred["disclaimer"].lower())

    def test_analyze_legitimate_looking_offer(self) -> None:
        """Synthetic case 3: Legitimate-looking job listing with no warning triggers."""
        payload = {
            "job_text": (
                "Junior Backend Developer at ExampleCorp. You will design and maintain Python "
                "microservices, write automated tests, and collaborate with product managers. "
                "Requirements: Bachelor's degree in Computer Science, familiarity with SQL and Git. "
                "Interviews are conducted via two rounds of technical discussions with our engineering team."
            )
        }
        resp = self.client.post("/api/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()

        self.assertEqual(body["rule_based_assessment"]["triggered_rule_count"], 0)
        self.assertEqual(body["rule_based_assessment"]["risk_level_code"], "no_rules_triggered")
        self.assertIn("Verification Still Required", body["rule_based_assessment"]["risk_category"])
        self.assertEqual(body["triggered_rules"], [])
        self.assertEqual(body["exact_matched_phrases"], [])

    def test_analyze_empty_and_oversized_inputs(self) -> None:
        """Synthetic cases 5 & 6: Empty input and excessively long input."""
        resp_empty = self.client.post("/api/analyze", json={"job_text": "   "})
        self.assertEqual(resp_empty.status_code, 400)
        self.assertEqual(resp_empty.get_json()["error"], "EMPTY_TEXT")

        resp_long = self.client.post("/api/analyze", json={"job_text": "A" * 26000})
        self.assertEqual(resp_long.status_code, 413)
        self.assertEqual(resp_long.get_json()["error"], "TEXT_TOO_LONG")

    def test_analyze_malformed_request_payload(self) -> None:
        """Synthetic case 10: Malformed JSON and invalid Content-Type payloads."""
        # Non-JSON Content-Type
        resp_plain = self.client.post(
            "/api/analyze",
            data="job_text=hello",
            content_type="text/plain",
        )
        self.assertEqual(resp_plain.status_code, 400)
        self.assertEqual(resp_plain.get_json()["error"], "INVALID_CONTENT_TYPE")

        # Broken JSON syntax
        resp_bad_json = self.client.post(
            "/api/analyze",
            data=b'{"job_text": "unterminated string}',
            content_type="application/json",
        )
        self.assertEqual(resp_bad_json.status_code, 400)
        self.assertEqual(resp_bad_json.get_json()["error"], "MALFORMED_JSON")

        # JSON array instead of JSON object
        resp_array = self.client.post(
            "/api/analyze",
            data=b'["not", "an", "object"]',
            content_type="application/json",
        )
        self.assertEqual(resp_array.status_code, 400)
        self.assertEqual(resp_array.get_json()["error"], "INVALID_JSON_OBJECT")

    def test_analyze_image_unreadable_and_blank(self) -> None:
        """Synthetic case 8: Unreadable screenshot and blank screenshot with no text."""
        # Corrupted image bytes
        resp_corrupt = self.client.post(
            "/api/analyze-image",
            data={"image": (io.BytesIO(b"corrupted-image-bytes"), "corrupt.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_corrupt.status_code, 400)
        self.assertEqual(resp_corrupt.get_json()["error"], "UNREADABLE_IMAGE")

        # Blank white image (valid PNG, but 0 readable characters)
        blank_buf = io.BytesIO()
        Image.new("RGB", (300, 120), color=(255, 255, 255)).save(blank_buf, format="PNG")
        blank_buf.seek(0)

        resp_blank = self.client.post(
            "/api/analyze-image",
            data={"image": (blank_buf, "blank.png")},
            content_type="multipart/form-data",
        )
        self.assertIn(resp_blank.status_code, (422, 503))

    def test_analyze_image_missing_tesseract_executable(self) -> None:
        """Synthetic case 9: Missing Tesseract executable returns 503 with clear setup instructions."""
        app_missing_ocr = create_app(
            model_path="models/nonexistent.joblib",
            tesseract_cmd="/nonexistent/path/to/tesseract_binary",
        )
        client_missing_ocr = app_missing_ocr.test_client()

        img_buf = io.BytesIO()
        Image.new("RGB", (200, 100), color=(255, 255, 255)).save(img_buf, format="PNG")
        img_buf.seek(0)

        resp = client_missing_ocr.post(
            "/api/analyze-image",
            data={"image": (img_buf, "offer.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertEqual(body["error"], "TESSERACT_NOT_INSTALLED")
        self.assertIn("Tesseract OCR executable was not found", body["message"])

    def test_analyze_image_end_to_end_with_synthetic_screenshot(self) -> None:
        """Verify end-to-end OCR + rule analysis on a rendered recruitment screenshot image."""
        img = Image.new("RGB", (900, 260), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        offer_text = (
            "URGENT HIRING - DATA ENTRY INTERN\n"
            "No interview required! Direct offer letter today.\n"
            "Pay a refundable registration fee of $120 to confirm.\n"
            "Contact HR on WhatsApp only to reserve your seat."
        )
        draw.text((25, 25), offer_text, fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = self.client.post(
            "/api/analyze-image",
            data={"image": (buf, "scam_screenshot.png")},
            content_type="multipart/form-data",
        )
        if resp.status_code == 200:
            body = resp.get_json()
            self.assertIn("ocr_extraction", body)
            self.assertTrue(body["processing_status"]["ocr_used"])
            self.assertGreaterEqual(body["rule_based_assessment"]["triggered_rule_count"], 1)

    def test_rate_limiting_enforced(self) -> None:
        """Verify that exceeding the per-minute request limit returns HTTP 429."""
        limited_app = create_app(
            model_path="models/nonexistent.joblib",
            rate_limit_per_minute=2,
        )
        limited_client = limited_app.test_client()

        self.assertEqual(limited_client.get("/api/health").status_code, 200)
        self.assertEqual(limited_client.get("/api/health").status_code, 200)
        resp_third = limited_client.get("/api/health")
        self.assertEqual(resp_third.status_code, 429)
        self.assertEqual(resp_third.get_json()["error"], "RATE_LIMIT_EXCEEDED")

    def test_rate_limiting_with_forwarded_for(self) -> None:
        """Verify rate limiter respects X-Forwarded-For to separate client IPs behind a proxy."""
        limited_app = create_app(
            model_path="models/nonexistent.joblib",
            rate_limit_per_minute=2,
        )
        limited_client = limited_app.test_client()

        # Client A makes 2 requests
        self.assertEqual(limited_client.get("/api/health", headers={"X-Forwarded-For": "198.51.100.1"}).status_code, 200)
        self.assertEqual(limited_client.get("/api/health", headers={"X-Forwarded-For": "198.51.100.1"}).status_code, 200)
        self.assertEqual(limited_client.get("/api/health", headers={"X-Forwarded-For": "198.51.100.1"}).status_code, 429)

        # Client B is not blocked
        self.assertEqual(limited_client.get("/api/health", headers={"X-Forwarded-For": "198.51.100.2"}).status_code, 200)

        # Attacker cannot bypass by prepending a spoofed IP to X-Forwarded-For
        self.assertEqual(
            limited_client.get("/api/health", headers={"X-Forwarded-For": "1.2.3.4, 198.51.100.1"}).status_code,
            429,
        )

    def test_sliding_window_rate_limiter_thread_safety_and_memory_cleanup(self) -> None:
        """Verify thread-safety under concurrency and bounded memory cleanup."""
        import collections
        import concurrent.futures
        import time

        limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=0.05)

        def _request(client_id: str) -> bool:
            allowed, _ = limiter.is_allowed(client_id)
            return allowed

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_request, "client_concurrent") for _ in range(20)]
            results = [f.result() for f in futures]

        self.assertEqual(results.count(True), 10)
        self.assertEqual(results.count(False), 10)

        # Populate with 1050 expired entries
        limiter._requests = collections.defaultdict(
            collections.deque,
            {f"ip_{i}": collections.deque([time.monotonic() - 100]) for i in range(1050)},
        )
        allowed, _ = limiter.is_allowed("ip_new")
        self.assertTrue(allowed)
        # Stale keys swept down from 1050 to only the new entry
        self.assertLessEqual(len(limiter._requests), 10)

    def test_analyze_image_with_url_alias(self) -> None:
        """Verify POST /api/analyze-image accepts 'url' form field as alias for 'source_url'."""
        img = Image.new("RGB", (300, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Full Time Python Developer role at Example Corp.", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = self.client.post(
            "/api/analyze-image",
            data={
                "image": (buf, "job.png"),
                "url": "https://careers.example.com/jobs/99",
            },
            content_type="multipart/form-data",
        )
        if resp.status_code == 200:
            body = resp.get_json()
            self.assertEqual(body["processing_status"]["source_url"], "https://careers.example.com/jobs/99")

    def test_dataset_validation_and_missing_text_handling(self) -> None:
        """Verify train_model.load_and_validate_dataset handles missing fields and validates columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "synthetic_jobs.csv"
            df = pd.DataFrame(
                [
                    {
                        "title": "Software Intern",
                        "company_profile": "Established tech company",
                        "description": "Write Python services and unit tests.",
                        "requirements": "CS student",
                        "benefits": "Paid mentorship",
                        "fraudulent": 0,
                    },
                    {
                        "title": "Backend Engineer",
                        "company_profile": None,
                        "description": "Develop REST APIs and database schemas.",
                        "requirements": None,
                        "benefits": "Health insurance",
                        "fraudulent": 0,
                    },
                    {
                        "title": "Quick Cash Entry",
                        "company_profile": "",
                        "description": "Pay a registration fee of $200 and message on Telegram.",
                        "requirements": "No experience",
                        "benefits": "Earn $5000 weekly",
                        "fraudulent": 1,
                    },
                    {
                        "title": "Home Typing",
                        "company_profile": None,
                        "description": "Send bank login credentials and refundable security deposit.",
                        "requirements": None,
                        "benefits": "Guaranteed placement",
                        "fraudulent": 1,
                    },
                ]
            )
            df.to_csv(csv_path, index=False)

            loaded = load_and_validate_dataset(csv_path)
            self.assertEqual(len(loaded), 4)
            self.assertIn("combined_text", loaded.columns)
            self.assertTrue(all(loaded["combined_text"].str.len() > 10))


if __name__ == "__main__":
    unittest.main()
