"""Interactive session metadata collection for BrainCo datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable


INTERACTION_CLASSES = ("none", "handshake", "fist_bump", "hug")
SEX_VALUES = ("f", "m")
MIN_AGE = 1
MAX_AGE = 120
MIN_HEIGHT_CM = 120
MAX_HEIGHT_CM = 230
MIN_WEIGHT_KG = 40
MAX_WEIGHT_KG = 120
MISSING_VALUE = "none"


@dataclass(frozen=True)
class SessionProfile:
    interaction_class: str
    sex: str
    age: int | str
    height_cm: int | str
    weight_kg: int | str

    def to_dict(self) -> dict:
        return asdict(self)

    def dataset_name(self, timestamp: datetime | None = None) -> str:
        timestamp = datetime.now() if timestamp is None else timestamp
        return (
            f"{self.interaction_class}-{self.sex}-age{self.age}-"
            f"height{self.height_cm}-weight{self.weight_kg}-"
            f"{timestamp.strftime('%Y%m%d-%H%M%S')}"
        )


def _choice_prompt(
    label: str,
    allowed: tuple[str, ...],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    allowed_text = ", ".join(allowed)
    while True:
        value = input_fn(f"{label} [{allowed_text}]: ").strip().lower()
        if value in allowed:
            return value
        output_fn(f"Неверное значение. Допустимо: {allowed_text}.")


def _integer_or_none_prompt(
    label: str,
    minimum: int,
    maximum: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> int | str:
    while True:
        raw = input_fn(f"{label} [{minimum}..{maximum} или n]: ").strip().lower()
        if raw == "n":
            return MISSING_VALUE
        try:
            value = int(raw)
        except ValueError:
            output_fn("Введите целое число или n, если значение неизвестно.")
            continue
        if minimum <= value <= maximum:
            return value
        output_fn(f"Значение должно быть от {minimum} до {maximum}.")


def _read_all_fields(
    input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> dict:
    return {
        "interaction_class": _choice_prompt(
            "Тип задачи", INTERACTION_CLASSES, input_fn, output_fn
        ),
        "sex": _choice_prompt("Пол", SEX_VALUES, input_fn, output_fn),
        "age": _integer_or_none_prompt(
            "Возраст", MIN_AGE, MAX_AGE, input_fn, output_fn
        ),
        "height_cm": _integer_or_none_prompt(
            "Рост, см", MIN_HEIGHT_CM, MAX_HEIGHT_CM, input_fn, output_fn
        ),
        "weight_kg": _integer_or_none_prompt(
            "Вес, кг", MIN_WEIGHT_KG, MAX_WEIGHT_KG, input_fn, output_fn
        ),
    }


def _show_fields(values: dict, output_fn: Callable[[str], None]) -> None:
    output_fn("\nВведённые данные:")
    output_fn(f"  task:   {values['interaction_class']}")
    output_fn(f"  sex:    {values['sex']}")
    output_fn(f"  age:    {values['age']}")
    height = values["height_cm"]
    weight = values["weight_kg"]
    output_fn(f"  height: {height if height == MISSING_VALUE else f'{height} cm'}")
    output_fn(f"  weight: {weight if weight == MISSING_VALUE else f'{weight} kg'}")


def collect_session_profile(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> SessionProfile:
    """Collect, validate, review, and confirm participant/session metadata."""
    output_fn("=== BrainCo dataset session metadata ===")
    values = _read_all_fields(input_fn, output_fn)
    edit_actions = (
        "ok", "task", "sex", "age", "height", "weight", "all"
    )
    while True:
        _show_fields(values, output_fn)
        action = _choice_prompt(
            "Подтвердить или изменить поле",
            edit_actions,
            input_fn,
            output_fn,
        )
        if action == "ok":
            return SessionProfile(**values)
        if action == "all":
            values = _read_all_fields(input_fn, output_fn)
        elif action == "task":
            values["interaction_class"] = _choice_prompt(
                "Тип задачи", INTERACTION_CLASSES, input_fn, output_fn
            )
        elif action == "sex":
            values["sex"] = _choice_prompt("Пол", SEX_VALUES, input_fn, output_fn)
        elif action == "age":
            values["age"] = _integer_or_none_prompt(
                "Возраст", MIN_AGE, MAX_AGE, input_fn, output_fn
            )
        elif action == "height":
            values["height_cm"] = _integer_or_none_prompt(
                "Рост, см", MIN_HEIGHT_CM, MAX_HEIGHT_CM, input_fn, output_fn
            )
        elif action == "weight":
            values["weight_kg"] = _integer_or_none_prompt(
                "Вес, кг", MIN_WEIGHT_KG, MAX_WEIGHT_KG, input_fn, output_fn
            )
