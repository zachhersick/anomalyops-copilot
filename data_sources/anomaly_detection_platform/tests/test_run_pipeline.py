import pytest

from run_pipeline import run_pipeline, run_step


def test_run_step_calls_step_function():
    calls = []

    def fake_step():
        calls.append("called")

    run_step("Fake step", fake_step)

    assert calls == ["called"]


def test_run_pipeline_runs_steps_in_order():
    calls = []

    def step_one():
        calls.append("one")

    def step_two():
        calls.append("two")

    fake_steps = [
        ("Step one", step_one),
        ("Step two", step_two),
    ]

    run_pipeline(fake_steps)

    assert calls == ["one", "two"]


def test_run_pipeline_stops_when_step_fails():
    calls = []

    def step_one():
        calls.append("one")

    def failing_step():
        calls.append("fail")
        raise RuntimeError("fake failure")

    def step_three():
        calls.append("three")

    fake_steps = [
        ("Step one", step_one),
        ("Failing step", failing_step),
        ("Step three", step_three),
    ]

    with pytest.raises(RuntimeError, match="fake failure"):
        run_pipeline(fake_steps)

    assert calls == ["one", "fail"]