"""Miscellaneous training utilities: W&B helpers, dynamic imports, OmegaConf tools, and timers."""

import wandb
import importlib
import os
import time
from omegaconf import OmegaConf, DictConfig, ListConfig


def wandb_run_exists():
    return isinstance(wandb.run, wandb.sdk.wandb_run.Run)


def import_type_from_str(s):
    module_name, type_name = s.rsplit(".", 1)
    module = importlib.import_module(module_name)
    type_to_import = getattr(module, type_name)
    return type_to_import


def recursive_set_struct(cfg, struct_value: bool):
    OmegaConf.set_struct(cfg, struct_value)
    if isinstance(cfg, DictConfig):
        for key in cfg.keys():
            try:
                value = cfg[key]
                if isinstance(value, (DictConfig, ListConfig)):
                    recursive_set_struct(value, struct_value)
            except Exception as e:
                # print(e)
                pass
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                recursive_set_struct(item, struct_value)


def get_filtered_state_dict(state_dict, state_dict_key):
    """
    Filter state_dict keys that start with the given prefix and remove the prefix.

    Args:
        state_dict: Dictionary of state dict keys and values
        state_dict_key: Prefix string to filter by

    Returns:
        Filtered dictionary with prefix removed from keys
    """
    filtered_dict = {}
    for key, value in state_dict.items():
        if key.startswith(state_dict_key):
            # Remove the prefix from the key
            new_key = key[len(state_dict_key) :].lstrip(".")
            filtered_dict[new_key] = value
    return filtered_dict


def custom_instantiate(d, _resolve=True, _recursive=False, **add_kwargs):
    """
    Recursively instantiate nested configs with _target_ fields.
    """

    def _recursive_instantiate(obj):
        # If it's a dict and has a _target_, instantiate it
        if isinstance(obj, dict) and "_target_" in obj:
            if obj.get("_recursive_", None) == True:
                assert False, "recursive is not supported"
            obj = obj.copy()
            obj.pop("_recursive_", None)
            obj.pop("_convert_", None)
            obj.pop("_partial_", None)
            _type = import_type_from_str(obj.pop("_target_"))
            # Recursively instantiate all dict/list values
            for k, v in list(obj.items()):
                if isinstance(v, (dict, DictConfig)):
                    obj[k] = _recursive_instantiate(v)
                elif isinstance(v, (list, ListConfig)):
                    obj[k] = [_recursive_instantiate(i) for i in v]
            return _type(**obj)
        # If it's a dict, recursively instantiate its values
        elif isinstance(obj, dict):
            return {k: _recursive_instantiate(v) for k, v in obj.items()}
        # If it's a list, recursively instantiate its items
        elif isinstance(obj, list):
            return [_recursive_instantiate(i) for i in obj]
        else:
            return obj

    # Top-level: allow add_kwargs to override
    d = d.copy()
    if isinstance(d, DictConfig):
        if _resolve:
            d = OmegaConf.to_container(d, resolve=_resolve)
        else:
            recursive_set_struct(d, False)
    if d.get("_recursive_", None) == True:
        assert False, "recursive is not supported"
    d.pop("_recursive_", None)
    d.pop("_convert_", None)
    d.pop("_partial_", None)
    _type = import_type_from_str(d.pop("_target_"))
    if _recursive:
        # Recursively instantiate all dict/list values
        for k, v in list(d.items()):
            if isinstance(v, (dict, DictConfig)):
                d[k] = _recursive_instantiate(v)
            elif isinstance(v, (list, ListConfig)):
                d[k] = [_recursive_instantiate(i) for i in v]
    return _type(**d, **add_kwargs)


# Global variable for timing indentation level
timer_indent_level = 0


# Context manager for timing
class Timer:
    def __init__(self, name="", instance_enabled=True):
        self.name = name
        self.start_time = None
        self.enabled = instance_enabled and os.environ.get("TIMER_ENABLED", "0") == "1"
        if "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
        else:
            self.rank = 0
        self.show_rank = os.environ.get("TIMER_SHOW_RANK", "0") == "1"
        self.rank_zero_only = os.environ.get("TIMER_RANK_ZERO_ONLY", "0") == "1"

    def __enter__(self):
        if (not self.enabled) or (self.rank_zero_only and self.rank != 0):
            return self
        global timer_indent_level
        self.start_time = time.perf_counter()
        self.current_indent = timer_indent_level  # Capture current indent level
        timer_indent_level += 1  # Increment global indent level for next call
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False  # Re-raise the exception
        if (not self.enabled) or (self.rank_zero_only and self.rank != 0):
            return self
        global timer_indent_level
        elapsed_time = time.perf_counter() - self.start_time
        indent = "    " * self.current_indent  # 4 spaces per indent level
        rank_str = f"[rank{self.rank}] " if self.show_rank else ""
        print(f"{indent}{rank_str}[{self.name}] time: {elapsed_time:.4f} seconds")
        timer_indent_level -= 1  # Decrement global indent level after finishing
