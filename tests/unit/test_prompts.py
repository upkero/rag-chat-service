import pytest

from src.app.messages import MESSAGES, get_message
from src.app.prompts import Prompt, get_prompt


def test_answer_prompts_are_registered_with_their_placeholders() -> None:
    system, user = get_prompt("rag.answer.system"), get_prompt("rag.answer.user")

    # Rules and wording on the system side, retrieved text and the question on the user side.
    assert system.placeholders == {"reply_language", "no_answer_marker"}
    assert user.placeholders == {"context", "question"}
    assert system.id.startswith("rag.answer.system@")


def test_answer_prompts_are_english() -> None:
    """The rule the whole migration exists for: no Cyrillic in a prompt file.

    Localized text the user reads verbatim lives in `messages/`; a prompt is read
    by the model, so the reply language is a placeholder instead.
    """
    for name in ("rag.answer.system", "rag.answer.user"):
        text = get_prompt(name).text

        assert not any("Ѐ" <= character <= "ӿ" for character in text)
    assert "{reply_language}" in get_prompt("rag.answer.system").text


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
    with pytest.raises(KeyError, match="rag.answer.system"):
        get_prompt("nope")


def test_message_falls_back_to_english_for_an_unknown_language() -> None:
    assert get_message("low_confidence", "de") == MESSAGES["en"]["low_confidence"]
    assert get_message("low_confidence", "ru") == MESSAGES["ru"]["low_confidence"]


def test_user_facing_messages_use_no_long_dash() -> None:
    """The site shows these strings word for word, and the copy rule bans the long dash."""
    for language, messages in MESSAGES.items():
        for key, text in messages.items():
            assert "—" not in text and "–" not in text, f"{language}.{key}"
