import pytest

from src.app.messages import MESSAGES, get_message
from src.app.prompts import Prompt, get_prompt


def test_answer_prompt_is_registered_with_its_placeholders() -> None:
    prompt = get_prompt("rag.answer")

    assert prompt.placeholders == {"reply_language", "context", "question"}
    assert prompt.id.startswith("rag.answer@")


def test_answer_prompt_is_english() -> None:
    """The rule the whole migration exists for: no Cyrillic in a prompt file.

    Localized text the user reads verbatim lives in `messages/`; a prompt is read
    by the model, so the reply language is a placeholder instead.
    """
    text = get_prompt("rag.answer").text

    assert not any("Ѐ" <= character <= "ӿ" for character in text)
    assert "{reply_language}" in text


def test_render_fills_every_placeholder() -> None:
    rendered = Prompt("greet", "Say hello to {name} in {reply_language}.").render(
        name="Anna",
        reply_language="English",
    )

    assert rendered == "Say hello to Anna in English."


def test_render_names_all_missing_placeholders_at_once() -> None:
    prompt = Prompt("greet", "Say hello to {name} in {reply_language}.")

    with pytest.raises(KeyError) as error:
        prompt.render(name="Anna")

    # Both the prompt id and the missing field, so the failure needs no bisect.
    assert "greet@" in str(error.value)
    assert "reply_language" in str(error.value)


def test_unknown_prompt_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="rag.answer"):
        get_prompt("nope")


def test_message_falls_back_to_english_for_an_unknown_language() -> None:
    assert get_message("low_confidence", "de") == MESSAGES["en"]["low_confidence"]
    assert get_message("low_confidence", "ru") == MESSAGES["ru"]["low_confidence"]
