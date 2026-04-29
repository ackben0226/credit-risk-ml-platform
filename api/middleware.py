import time
import uuid
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.middleware")


# =====================================================
# SAFE TIME FUNCTION (MONOTONIC CLOCK)
# =====================================================

def now():
    return time.perf_counter()


# =====================================================
# MIDDLEWARE
# =====================================================

async def request_logging_middleware(request: Request, call_next):

    # -------------------------------------------------
    # 1. TRACE / REQUEST ID HANDLING
    # -------------------------------------------------
    incoming_request_id = request.headers.get("X-Request-ID")

    request_id = incoming_request_id if incoming_request_id else str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = now()

    try:
        response = await call_next(request)

        latency_ms = (now() - start_time) * 1000

        # -------------------------------------------------
        # 2. STRUCTURED LOGGING (PRODUCTION-READY)
        # -------------------------------------------------
        logger.info(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2)
            }
        )

        # -------------------------------------------------
        # 3. TRACE HEADERS (DISTRIBUTED SYSTEMS READY)
        # -------------------------------------------------
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = str(round(latency_ms, 2))

        return response

    except Exception as e:

        latency_ms = (now() - start_time) * 1000

        logger.exception(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "latency_ms": round(latency_ms, 2)
            }
        )

        # -------------------------------------------------
        # 4. STRUCTURED ERROR RESPONSE
        # -------------------------------------------------
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": request_id
            },
            headers={
                "X-Request-ID": request_id
            }
        )