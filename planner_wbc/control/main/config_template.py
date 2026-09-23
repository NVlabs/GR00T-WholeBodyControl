from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ArgsConfig:
    """Args Config for running the data collection loop."""

    def update(
        self,
        config_dict: dict,
        strict: bool = False,
        skip_keys: list[str] = [],
        allowed_keys: list[str] | None = None,
    ):
        for k, v in config_dict.items():
            if k in skip_keys:
                continue
            if allowed_keys is not None and k not in allowed_keys:
                continue
            if strict and not hasattr(self, k):
                raise ValueError(f"Config {k} not found in {self.__class__.__name__}")
            if not strict and not hasattr(self, k):
                continue
            setattr(self, k, v)

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
        strict: bool = False,
        skip_keys: list[str] = [],
        allowed_keys: list[str] | None = None,
    ):
        instance = cls()
        instance.update(
            config_dict=config_dict, strict=strict, skip_keys=skip_keys, allowed_keys=allowed_keys
        )
        return instance

    def to_dict(self):
        return asdict(self)

    def get(self, key: str, default: Any = None):
        return getattr(self, key) if hasattr(self, key) else default
