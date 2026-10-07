from pydantic import BaseModel, ValidationError

from tracemeet.errors import summarize_error, write_error_log


class Model(BaseModel):
    x: int


def test_validation_errors_are_summarised_without_input_values():
    try:
        Model.model_validate({"x": "secret transcript text"})
    except ValidationError as exc:
        message = summarize_error(exc)
    assert message.startswith("1 validation error for Model")
    assert "secret" not in message


def test_many_errors_stay_short():
    try:
        Model.model_validate({})
        raise AssertionError("expected a validation error")
    except ValidationError as exc:
        assert len(summarize_error(exc)) <= 400


def test_long_messages_are_truncated():
    assert len(summarize_error(RuntimeError("x" * 5000))) == 400


def test_plain_errors_name_their_type():
    assert summarize_error(ValueError("bad value")) == "ValueError: bad value"


def test_error_log_contains_the_traceback(tmp_path):
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        write_error_log(tmp_path / "error.log", exc)
    assert "RuntimeError: boom" in (tmp_path / "error.log").read_text(encoding="utf-8")