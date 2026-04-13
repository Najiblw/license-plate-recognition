from src.postprocess import clean_plate_text, is_plausible_plate


def test_clean_plate_text_removes_symbols() -> None:
    assert clean_plate_text(" ubA 123a! ") == "UBA123A"


def test_is_plausible_plate_accepts_mixed_alphanumeric_text() -> None:
    assert is_plausible_plate("UBA123A") is True


def test_is_plausible_plate_rejects_short_text() -> None:
    assert is_plausible_plate("AB1") is False


def test_is_plausible_plate_rejects_letters_only_text() -> None:
    assert is_plausible_plate("ABCDEFG") is False


def test_is_plausible_plate_rejects_repetitive_noise() -> None:
    assert is_plausible_plate("AAAAAAA") is False
