from src.app.core.settings.app import AppSettings


def test_cors_origins_are_parsed_from_a_comma_separated_string() -> None:
    # The env always hands over a string; NoDecode + the validator is what turns
    # it into a list without demanding JSON quoting in a shell.
    settings = AppSettings(_env_file=None, cors_allowed_origins="http://a.test, http://b.test ,")

    assert settings.cors_allowed_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_default_to_closed() -> None:
    assert AppSettings(_env_file=None).cors_allowed_origins == []
