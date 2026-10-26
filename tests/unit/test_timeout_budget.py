from src.app.core.settings.llm import LLMSettings
from src.app.core.settings.ops_core import OpsCoreSettings
from src.app.core.settings.rerank import RerankSettings

# How long the site's bridge waits for /ask before it gives up on the visitor's behalf.
_BRIDGE_TIMEOUT_SECONDS = 90.0


def _default(model: type, field: str) -> float:
    default = model.model_fields[field].default  # type: ignore[attr-defined]
    assert isinstance(default, int | float)
    return float(default)


def test_the_llm_timeout_times_its_attempts_fits_under_the_bridge() -> None:
    attempts = _default(LLMSettings, "max_retries") + 1
    assert _default(LLMSettings, "timeout_seconds") * attempts < _BRIDGE_TIMEOUT_SECONDS


def test_one_slow_search_and_a_dead_reranker_still_leave_the_model_its_whole_budget() -> None:
    """A search that needs a second attempt and a rerank that times out are both survivable;
    the model call after them must still finish before the bridge gives up."""
    llm = _default(LLMSettings, "timeout_seconds") * (_default(LLMSettings, "max_retries") + 1)
    search = 2 * _default(OpsCoreSettings, "timeout_seconds")
    rerank = _default(RerankSettings, "timeout_seconds")
    assert search + rerank + llm < _BRIDGE_TIMEOUT_SECONDS
