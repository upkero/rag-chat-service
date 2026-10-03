import hmac
from typing import Annotated

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from src.app.core.settings.security import SecuritySettings, get_security_settings
from src.app.exceptions.auth import UnauthorizedError

# auto_error=False: we raise our own typed UnauthorizedError so the response uses
# the app's uniform error envelope (not FastAPI's default 403 shape). Declaring it
# via APIKeyHeader still advertises the scheme in the OpenAPI docs.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    provided: Annotated[str | None, Security(_api_key_header)],
    settings: Annotated[SecuritySettings, Depends(get_security_settings)],
) -> None:
    # hmac.compare_digest → constant-time compare, avoids leaking the key via
    # timing. Encode to bytes so non-ASCII keys don't raise.
    expected = settings.api_key.get_secret_value()
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise UnauthorizedError()
