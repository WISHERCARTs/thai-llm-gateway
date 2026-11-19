"""Errors in the OpenAI error shape, so OpenAI-compatible clients surface them properly."""

from fastapi.responses import JSONResponse


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, *, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}

    def response(self, request_id: str) -> JSONResponse:
        return JSONResponse(
            status_code=self.status,
            content={"error": {"message": self.message, "type": _type(self.status), "code": self.code}},
            headers={**self.headers, "x-request-id": request_id},
        )


def _type(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 429:
        return "rate_limit_error"
    if status >= 500:
        return "api_error"
    return "invalid_request_error"
