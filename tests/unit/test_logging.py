import logging
from collections.abc import Iterator

import pytest

from src.app.core.logging import setup_logging
from src.app.core.settings.logging import LoggingSettings


@pytest.fixture
def restore_logging() -> Iterator[None]:
    names = ("", "uvicorn", "uvicorn.error", "uvicorn.access")
    saved = [(n, logging.getLogger(n).handlers[:], logging.getLogger(n).propagate) for n in names]
    yield
    for name, handlers, propagate in saved:
        logging.getLogger(name).handlers = handlers
        logging.getLogger(name).propagate = propagate


@pytest.mark.usefixtures("restore_logging")
def test_the_access_log_stays_off_and_the_error_log_goes_to_the_root_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging(LoggingSettings(_env_file=None, format="text"))

    logging.getLogger("uvicorn.access").info("GET /health/live 200")
    logging.getLogger("uvicorn.error").warning("boom")

    output = capsys.readouterr().out
    assert "/health/live" not in output
    assert "boom" in output
