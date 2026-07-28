from src.app.core.settings.app import AppSettings


def test_cors_origins_are_parsed_from_a_comma_separated_string() -> None:
    # The env always hands over a string; NoDecode + the validator is what turns
    # it into a list without demanding JSON quoting in a shell.
    # The suppression below is part of the assertion: mypy sees the declared
    # list[str] and not the NoDecode validator, which is the exact mismatch this
    # test exists to pin down.
    settings = AppSettings(cors_allowed_origins="http://a.test, http://b.test ,")  # type: ignore[arg-type]

    assert settings.cors_allowed_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_default_to_closed() -> None:
    assert AppSettings().cors_allowed_origins == []
