# Privacy and Operations Checklist

This checklist describes the product's intended handling and the controls required before a public release. It is not a legal opinion or a substitute for reviewing the actual hosting configuration.

## Data flow and user notice

- Page text and pasted text are analyzed locally in the extension. Do not transmit them for telemetry by default.
- Screenshot OCR sends the selected image to the backend URL shown in the extension only after an explicit confirmation. Remote destinations must use HTTPS; local loopback is allowed for development.
- Explain what is sent, the destination, the purpose, and the expected retention before upload. The backend code does not persist the uploaded image or job text; separately review hosting, proxy, crash, and request logs because providers may retain metadata.
- Do not add analytics that collect listing text, screenshots, URLs with tracking parameters, recruiter contact details, or user identifiers without a documented purpose and privacy review.

## Minimum controls before public use

- Keep the Cloud Run service private. For user-facing remote OCR, put an authenticated gateway in front of it and apply service-wide rate limits, request-size limits, abuse monitoring, and an incident response path. CORS is not authentication.
- Require HTTPS for remote traffic, keep dependencies and base images patched, scan deployment artifacts, protect publisher/cloud accounts with MFA, and maintain a tested rollback procedure.
- Configure provider logging to avoid request bodies and uploaded file contents. Set and periodically review retention for operational logs and incident records.
- Provide a way to report false alarms, missed scams, extraction defects, and privacy concerns. Do not collect the submitted listing as part of a report unless the user explicitly chooses to attach it.
- Maintain a model and rule change log: version, data snapshot hash, evaluation report, known limitations, reviewer, release date, and rollback target.

## Legal and privacy review

Before processing real user data at scale, map each data flow, purpose, processor/hosting provider, retention period, user notice, and rights-handling process with qualified counsel. Confirm which provisions and commencement dates apply to the deployment and user population under India's Digital Personal Data Protection Act and Rules, and assess any other applicable jurisdictions. Re-check this mapping when collection, hosting, or product behavior changes.
