from __future__ import annotations
import json
import time
from typing import Any, Callable
from starlette.requests import Request
from starlette.responses import Response


def json_logger_middleware() -> Callable:
    """Return a Starlette middleware callable that logs a JSON line per request.

    It captures: method, path, status, latency_ms, and any selected attributes
    from request.state (retrieved_k, chunks_used, low_confidence, selected_intent).
    """

    async def _middleware(request: Request, call_next: Callable[[Request], Any]) -> Response:
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            s = getattr(request, "state", None)
            payload = {
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": latency_ms,
            }
            for k in ("retrieved_k", "chunks_used", "low_confidence", "selected_intent"):
                if s is not None and hasattr(s, k):
                    payload[k] = getattr(s, k)
            try:
                print(json.dumps(payload), flush=True)
            except Exception:
                pass
        return response

    return _middleware
