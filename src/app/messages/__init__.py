"""Text the user receives verbatim, by language.

Not prompts. A prompt is written for a model, so it is English and the reply
language is a placeholder inside it; these strings are read by a human exactly as
written, so a translation is the whole point rather than a fork. Keeping the two
apart is what lets `prompts/` stay English-only without the rule breaking on its
first day.
"""

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "low_confidence": "I can't answer that precisely. Please check with an administrator.",
    },
    "ru": {
        "low_confidence": "Не могу ответить точно, уточните у администратора.",
    },
}

_DEFAULT_LANGUAGE = "en"


def get_message(key: str, language: str) -> str:
    """Return one message, falling back to English for an unknown language.

    A missing *key* raises: that is a typo in this repository and should fail
    loudly. A missing *language* does not: it comes from configuration, and
    answering in the wrong language beats answering with a stack trace.
    """
    return MESSAGES.get(language, MESSAGES[_DEFAULT_LANGUAGE])[key]
