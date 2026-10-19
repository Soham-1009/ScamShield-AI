"""
Lightweight WSGI & TestClient fallback (`scamshield/_wsgi_fallback.py`).

Used ONLY when the external `flask` package is not installed in an offline or
minimal Python environment. When `flask` IS installed, `app.py` imports the real
`Flask`, `request`, `jsonify`, and `make_response` directly from `flask`.

Supports:
- `@app.route(path, methods=[...])`
- `@app.after_request`
- `@app.errorhandler(status_code)`
- `request.method`, `request.path`, `request.headers`, `request.remote_addr`,
  `request.content_length`, `request.is_json`, `request.get_json(silent=...)`,
  `request.get_data()`, `request.files`, `request.form`
- `app.test_client()` for synchronous unit/integration testing
- `app.run(host, port)` via Python's built-in `wsgiref.simple_server`
"""

from __future__ import annotations

import io
import json
import threading
from email.parser import BytesParser
from email.policy import default as default_email_policy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server


class UploadedFileStorage:
    """Minimal compatible replacement for Werkzeug FileStorage."""

    def __init__(self, filename: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.filename = filename
        self._data = data
        self.content_type = content_type
        self.stream = io.BytesIO(data)

    def read(self) -> bytes:
        return self._data


class CaseInsensitiveHeaders(dict):
    """Case-insensitive dictionary for HTTP headers, matching Werkzeug/Flask behaviour."""

    def __init__(self, data: Optional[Dict[str, str]] = None) -> None:
        super().__init__()
        self._keys: Dict[str, str] = {}
        if data:
            for k, v in data.items():
                self[k] = v

    def __setitem__(self, key: str, value: str) -> None:
        lower_k = key.lower()
        self._keys[lower_k] = key
        super().__setitem__(lower_k, value)

    def __getitem__(self, key: str) -> str:
        return super().__getitem__(key.lower())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and super().__contains__(key.lower())

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key.lower(), default)

    def items(self) -> Any:
        return [(self._keys.get(k, k), v) for k, v in super().items()]

    def keys(self) -> Any:
        return [self._keys.get(k, k) for k in super().keys()]


class RequestContext:
    """Thread-local HTTP request proxy."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _set(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
        remote_addr: str = "127.0.0.1",
        files: Optional[Dict[str, UploadedFileStorage]] = None,
        form: Optional[Dict[str, str]] = None,
    ) -> None:
        self._local.method = method.upper()
        self._local.path = path
        self._local.headers = CaseInsensitiveHeaders(headers)
        self._local.body = body
        self._local.remote_addr = remote_addr
        self._local.files = files or {}
        self._local.form = form or {}

    @property
    def method(self) -> str:
        return getattr(self._local, "method", "GET")

    @property
    def path(self) -> str:
        return getattr(self._local, "path", "/")

    @property
    def headers(self) -> Dict[str, str]:
        return getattr(self._local, "headers", {})

    @property
    def remote_addr(self) -> str:
        return getattr(self._local, "remote_addr", "127.0.0.1")

    @property
    def content_length(self) -> int:
        return len(getattr(self._local, "body", b""))

    @property
    def content_type(self) -> str:
        for k, v in self.headers.items():
            if k.lower() == "content-type":
                return v
        return ""

    @property
    def is_json(self) -> bool:
        ct = self.content_type.lower()
        return "application/json" in ct or ct.endswith("+json")

    def get_data(self, as_text: bool = False) -> Any:
        raw = getattr(self._local, "body", b"")
        return raw.decode("utf-8", errors="replace") if as_text else raw

    def get_json(self, silent: bool = False, force: bool = False) -> Any:
        if not force and not self.is_json:
            if silent:
                return None
            raise ValueError("Content-Type was not 'application/json'.")
        raw = getattr(self._local, "body", b"")
        if not raw:
            if silent:
                return None
            raise ValueError("Empty JSON body.")
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            if silent:
                return None
            raise

    @property
    def files(self) -> Dict[str, UploadedFileStorage]:
        return getattr(self._local, "files", {})

    @property
    def form(self) -> Dict[str, str]:
        return getattr(self._local, "form", {})


request = RequestContext()


class Response:
    """Minimal HTTP Response object compatible with Flask's Response & TestResponse."""

    def __init__(
        self,
        data: bytes | str = b"",
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        content_type: str = "application/json",
    ) -> None:
        if isinstance(data, str):
            self.data = data.encode("utf-8")
        else:
            self.data = data
        self.status_code = int(status_code)
        self.headers: Dict[str, str] = dict(headers or {})
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = content_type

    def get_data(self, as_text: bool = False) -> Any:
        return self.data.decode("utf-8", errors="replace") if as_text else self.data

    def get_json(self, silent: bool = True) -> Any:
        try:
            return json.loads(self.data.decode("utf-8"))
        except Exception:
            if silent:
                return None
            raise


def jsonify(payload: Any) -> Response:
    body = json.dumps(payload, ensure_ascii=False)
    return Response(data=body, status_code=200, content_type="application/json")


def make_response(rv: Any, status_code: Optional[int] = None) -> Response:
    if isinstance(rv, tuple):
        body = rv[0]
        code = rv[1] if len(rv) > 1 else (status_code or 200)
        hdrs = rv[2] if len(rv) > 2 else None
        if isinstance(body, Response):
            body.status_code = int(code)
            if hdrs:
                body.headers.update(hdrs)
            return body
        if isinstance(body, dict):
            resp = jsonify(body)
            resp.status_code = int(code)
            if hdrs:
                resp.headers.update(hdrs)
            return resp
        return Response(data=body, status_code=int(code), headers=hdrs)

    if isinstance(rv, Response):
        if status_code is not None:
            rv.status_code = int(status_code)
        return rv

    if isinstance(rv, dict):
        resp = jsonify(rv)
        if status_code is not None:
            resp.status_code = int(status_code)
        return resp

    return Response(
        data=str(rv),
        status_code=int(status_code or 200),
        content_type="text/plain; charset=utf-8",
    )


def _parse_multipart(content_type: str, body: bytes) -> Tuple[Dict[str, UploadedFileStorage], Dict[str, str]]:
    files: Dict[str, UploadedFileStorage] = {}
    form: Dict[str, str] = {}
    if not body or "multipart/form-data" not in content_type.lower():
        return files, form

    raw_mime = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    msg = BytesParser(policy=default_email_policy).parsebytes(raw_mime)
    if msg.is_multipart():
        for part in msg.iter_parts():
            cd = part.get("Content-Disposition", "")
            if "form-data" not in cd:
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            part_ct = part.get_content_type() or "application/octet-stream"
            if name:
                if filename is not None:
                    files[name] = UploadedFileStorage(filename=filename, data=payload, content_type=part_ct)
                else:
                    form[name] = payload.decode("utf-8", errors="replace")
    return files, form


class TestClient:
    """Synchronous test client matching Flask's `app.test_client()` interface."""

    def __init__(self, app: "Flask") -> None:
        self.app = app

    def open(
        self,
        path: str,
        method: str = "GET",
        json_data: Any = None,
        data: Any = None,
        content_type: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        remote_addr: str = "127.0.0.1",
    ) -> Response:
        req_headers: Dict[str, str] = dict(headers or {})
        body_bytes = b""
        files: Dict[str, UploadedFileStorage] = {}
        form: Dict[str, str] = {}

        if json_data is not None:
            body_bytes = json.dumps(json_data).encode("utf-8")
            req_headers["Content-Type"] = content_type or "application/json"
        elif isinstance(data, (bytes, bytearray)):
            body_bytes = bytes(data)
            if content_type:
                req_headers["Content-Type"] = content_type
        elif isinstance(data, str):
            body_bytes = data.encode("utf-8")
            if content_type:
                req_headers["Content-Type"] = content_type
        elif isinstance(data, dict):
            # Support Flask test_client multipart syntax: data={"image": (BytesIO(b"..."), "test.png")}
            for k, v in data.items():
                if isinstance(v, tuple) and len(v) >= 2:
                    stream_or_bytes, fname = v[0], v[1]
                    raw_bytes = (
                        stream_or_bytes.read()
                        if hasattr(stream_or_bytes, "read")
                        else bytes(stream_or_bytes)
                    )
                    files[k] = UploadedFileStorage(filename=fname, data=raw_bytes)
                    body_bytes += raw_bytes
                elif hasattr(v, "read"):
                    raw_bytes = v.read()
                    fname = getattr(v, "name", "upload.bin")
                    files[k] = UploadedFileStorage(filename=fname, data=raw_bytes)
                    body_bytes += raw_bytes
                else:
                    form[k] = str(v)
            req_headers["Content-Type"] = content_type or "multipart/form-data"

        if content_type and "Content-Type" not in req_headers:
            req_headers["Content-Type"] = content_type

        return self.app._dispatch_request(
            method=method,
            path=path.split("?")[0],
            headers=req_headers,
            body=body_bytes,
            remote_addr=remote_addr,
            files=files,
            form=form,
        )

    def get(self, path: str, **kwargs: Any) -> Response:
        return self.open(path, method="GET", **kwargs)

    def post(self, path: str, json: Any = None, **kwargs: Any) -> Response:
        return self.open(path, method="POST", json_data=json, **kwargs)

    def options(self, path: str, **kwargs: Any) -> Response:
        return self.open(path, method="OPTIONS", **kwargs)


class Flask:
    """Minimal Flask-compatible WSGI application class."""

    def __init__(self, import_name: str) -> None:
        self.import_name = import_name
        self.config: Dict[str, Any] = {}
        self._routes: Dict[str, Tuple[List[str], Callable[..., Any]]] = {}
        self._after_request_funcs: List[Callable[[Response], Response]] = []
        self._error_handlers: Dict[int, Callable[[Any], Any]] = {}

    def route(self, rule: str, methods: Optional[List[str]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        allowed_methods = [m.upper() for m in (methods or ["GET"])]
        if "GET" in allowed_methods and "HEAD" not in allowed_methods:
            allowed_methods.append("HEAD")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes[rule] = (allowed_methods, func)
            return func

        return decorator

    def after_request(self, func: Callable[[Response], Response]) -> Callable[[Response], Response]:
        self._after_request_funcs.append(func)
        return func

    def errorhandler(self, code: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._error_handlers[int(code)] = func
            return func

        return decorator

    def _finalize_response(self, rv: Any) -> Response:
        resp = make_response(rv)
        for after_fn in self._after_request_funcs:
            resp = after_fn(resp)
        return resp

    def _dispatch_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
        remote_addr: str = "127.0.0.1",
        files: Optional[Dict[str, UploadedFileStorage]] = None,
        form: Optional[Dict[str, str]] = None,
    ) -> Response:
        max_len = self.config.get("MAX_CONTENT_LENGTH")
        if max_len is not None and len(body) > int(max_len):
            if 413 in self._error_handlers:
                request._set(method, path, headers, body, remote_addr=remote_addr)
                return self._finalize_response(self._error_handlers[413]( Exception("Payload Too Large")))
            return self._finalize_response(
                ({"error": "PAYLOAD_TOO_LARGE", "message": "Request payload exceeds maximum allowed size."}, 413)
            )

        if not files and not form and "multipart/form-data" in headers.get("Content-Type", "").lower():
            files, form = _parse_multipart(headers.get("Content-Type", ""), body)

        request._set(
            method=method,
            path=path,
            headers=headers,
            body=body,
            remote_addr=remote_addr,
            files=files,
            form=form,
        )

        if method.upper() == "OPTIONS" and path in self._routes:
            return self._finalize_response(Response(b"", status_code=204))

        if path not in self._routes:
            if 404 in self._error_handlers:
                return self._finalize_response(self._error_handlers[404](Exception("Not Found")))
            return self._finalize_response(({"error": "NOT_FOUND", "message": "Endpoint not found."}, 404))

        allowed_methods, view_func = self._routes[path]
        if method.upper() not in allowed_methods:
            if 405 in self._error_handlers:
                return self._finalize_response(self._error_handlers[405](Exception("Method Not Allowed")))
            return self._finalize_response(
                ({"error": "METHOD_NOT_ALLOWED", "message": f"Method {method} not allowed."}, 405)
            )

        try:
            rv = view_func()
            return self._finalize_response(rv)
        except Exception as exc:
            if 500 in self._error_handlers:
                return self._finalize_response(self._error_handlers[500](exc))
            return self._finalize_response(
                ({"error": "INTERNAL_SERVER_ERROR", "message": "An unexpected internal error occurred."}, 500)
            )

    def wsgi_app(self, environ: Dict[str, Any], start_response: Callable[..., Any]) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")
        remote_addr = environ.get("REMOTE_ADDR", "127.0.0.1")

        try:
            content_length = int(environ.get("CONTENT_LENGTH") or 0)
        except (ValueError, TypeError):
            content_length = 0

        wsgi_input = environ.get("wsgi.input")
        body = wsgi_input.read(content_length) if (wsgi_input and content_length > 0) else b""

        headers: Dict[str, str] = {}
        if environ.get("CONTENT_TYPE"):
            headers["Content-Type"] = environ["CONTENT_TYPE"]
        if environ.get("CONTENT_LENGTH"):
            headers["Content-Length"] = environ["CONTENT_LENGTH"]
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                header_name = key[5:].replace("_", "-").title()
                headers[header_name] = str(value)

        resp = self._dispatch_request(
            method=method,
            path=path,
            headers=headers,
            body=body,
            remote_addr=remote_addr,
        )
        status_reasons = {
            200: "200 OK",
            204: "204 No Content",
            400: "400 Bad Request",
            404: "404 Not Found",
            405: "405 Method Not Allowed",
            413: "413 Payload Too Large",
            422: "422 Unprocessable Entity",
            429: "429 Too Many Requests",
            500: "500 Internal Server Error",
            503: "503 Service Unavailable",
        }
        status_line = status_reasons.get(resp.status_code, f"{resp.status_code} OK")
        response_headers = [(str(k), str(v)) for k, v in resp.headers.items()]
        start_response(status_line, response_headers)
        return [resp.data]

    def __call__(self, environ: Dict[str, Any], start_response: Callable[..., Any]) -> Iterable[bytes]:
        return self.wsgi_app(environ, start_response)

    def test_client(self) -> TestClient:
        return TestClient(self)

    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
        with make_server(host, port, self) as httpd:
            print(f"[ScamShield AI] Serving WSGI application on http://{host}:{port}")
            httpd.serve_forever()
