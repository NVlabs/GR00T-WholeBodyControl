"""Utilities for scheduled parameter updates and learning-rate scheduling.

Includes object-path navigation for dynamically accessing/mutating nested
config attributes, a WarmupCosineScheduler for LR with linear warm-up and
cosine decay, and helpers for managing parameter change schedules.
"""

import numpy
import torch
import math
import re
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf.dictconfig import DictConfig


def _navigate_object_path(obj, path, split_char="@"):
    """
    Navigate through a complex object path that may include:
    - Attribute access: obj.attr
    - Function calls: obj.method('param')
    - Dictionary/array access: obj['key'][0]
    - Mixed combinations: obj.method('param')['key'][0].attr
    """
    current_obj = obj

    # Split the path by the split_char and process each segment
    segments = path.split(split_char)

    for segment in segments:
        current_obj = _process_path_segment(current_obj, segment)

    return current_obj


def _process_path_segment(obj, segment):
    """
    Process a single path segment that may contain:
    - Simple attribute: attr
    - Function call: method('param')
    - Bracket access: ['key'][0]
    - Combined: method('param')['key']
    """
    current_obj = obj

    # Parse the segment to identify different access patterns
    i = 0
    while i < len(segment):
        if segment[i] == "[":
            # Handle bracket access
            bracket_end = _find_matching_bracket(segment, i)
            bracket_content = segment[i + 1 : bracket_end]

            # Evaluate the bracket content
            if bracket_content.startswith("'") and bracket_content.endswith("'"):
                # String key
                key = bracket_content[1:-1]
                current_obj = current_obj[key]
            elif bracket_content.startswith('"') and bracket_content.endswith('"'):
                # String key with double quotes
                key = bracket_content[1:-1]
                current_obj = current_obj[key]
            elif bracket_content.lstrip("-").isdigit():
                # Numeric index
                index = int(bracket_content)
                current_obj = current_obj[index]
            else:
                # Try to evaluate as expression (for complex keys)
                try:
                    key = eval(bracket_content)
                    current_obj = current_obj[key]
                except:
                    # Fallback to string key
                    current_obj = current_obj[bracket_content]

            i = bracket_end + 1

        else:
            # Handle attribute access or function call
            attr_start = i
            # Find the end of the identifier (attribute or method name)
            while i < len(segment) and (segment[i].isalnum() or segment[i] == "_"):
                i += 1

            if attr_start < i:
                attr_name = segment[attr_start:i]

                # Check if this is followed by parentheses (function call)
                if i < len(segment) and segment[i] == "(":
                    # This is a function call
                    paren_end = _find_matching_paren(segment, i)
                    args_str = segment[i + 1 : paren_end]

                    # Parse and evaluate arguments
                    args = _parse_function_args(args_str)

                    # Call the method
                    method = getattr(current_obj, attr_name)
                    current_obj = method(*args)

                    i = paren_end + 1
                else:
                    # This is a simple attribute access
                    if attr_name.lstrip("-").isdigit():
                        # Numeric index for direct access
                        current_obj = current_obj[int(attr_name)]
                    else:
                        # Attribute access
                        current_obj = getattr(current_obj, attr_name)
            else:
                # Skip non-alphanumeric characters that aren't brackets or parentheses
                i += 1

    return current_obj


def _find_matching_bracket(s, start):
    """Find the matching closing bracket for an opening bracket at position start."""
    count = 1
    i = start + 1
    while i < len(s) and count > 0:
        if s[i] == "[":
            count += 1
        elif s[i] == "]":
            count -= 1
        i += 1
    return i - 1


def _find_matching_paren(s, start):
    """Find the matching closing parenthesis for an opening parenthesis at position start."""
    count = 1
    i = start + 1
    while i < len(s) and count > 0:
        if s[i] == "(":
            count += 1
        elif s[i] == ")":
            count -= 1
        i += 1
    return i - 1


def _parse_function_args(args_str):
    """Parse function arguments from a string."""
    if not args_str.strip():
        return []

    args = []
    current_arg = ""
    paren_count = 0
    bracket_count = 0
    in_quotes = False
    quote_char = None

    for char in args_str:
        if char in ['"', "'"] and not in_quotes:
            in_quotes = True
            quote_char = char
            current_arg += char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current_arg += char
        elif not in_quotes:
            if char == "(":
                paren_count += 1
                current_arg += char
            elif char == ")":
                paren_count -= 1
                current_arg += char
            elif char == "[":
                bracket_count += 1
                current_arg += char
            elif char == "]":
                bracket_count -= 1
                current_arg += char
            elif char == "," and paren_count == 0 and bracket_count == 0:
                args.append(_evaluate_arg(current_arg.strip()))
                current_arg = ""
            else:
                current_arg += char
        else:
            current_arg += char

    if current_arg.strip():
        args.append(_evaluate_arg(current_arg.strip()))

    return args


def _evaluate_arg(arg_str):
    """Evaluate a function argument string to its proper type."""
    arg_str = arg_str.strip()

    # String literals
    if (arg_str.startswith("'") and arg_str.endswith("'")) or (
        arg_str.startswith('"') and arg_str.endswith('"')
    ):
        return arg_str[1:-1]

    # Numeric literals
    if arg_str.lstrip("-").replace(".", "").isdigit():
        if "." in arg_str:
            return float(arg_str)
        else:
            return int(arg_str)

    # Boolean literals
    if arg_str.lower() == "true":
        return True
    elif arg_str.lower() == "false":
        return False
    elif arg_str.lower() == "none":
        return None

    # For complex expressions, try eval (be careful in production)
    try:
        return eval(arg_str)
    except:
        # Fallback to string
        return arg_str


def _get_final_target(obj, target_attr):
    """Get the final target object for reading, handling complex paths."""
    if _is_complex_path(target_attr):
        return _process_path_segment(obj, target_attr)
    else:
        # Simple attribute or numeric index
        if target_attr.lstrip("-").isdigit():
            return obj[int(target_attr)]
        else:
            return getattr(obj, target_attr)


def _set_final_target(obj, target_attr, value):
    """Set the final target value, handling complex paths."""
    if _is_complex_path(target_attr):
        # For complex paths, we need to navigate to the parent and set the final element
        _set_complex_path_value(obj, target_attr, value)
    else:
        # Simple attribute or numeric index
        if target_attr.lstrip("-").isdigit():
            obj[int(target_attr)] = value
        else:
            setattr(obj, target_attr, value)


def _is_complex_path(path):
    """Check if a path contains complex access patterns (brackets or parentheses)."""
    return "[" in path or "(" in path


def _set_complex_path_value(obj, path, value):
    """Set a value using a complex path by navigating to the parent and setting the final element."""
    # Parse the path to find the parent path and final accessor
    parent_obj = obj

    # Find the last bracket or the final attribute
    last_bracket = path.rfind("[")
    last_paren = path.rfind("(")

    if last_bracket > last_paren:
        # Last accessor is a bracket
        bracket_end = _find_matching_bracket(path, last_bracket)
        parent_path = path[:last_bracket]
        bracket_content = path[last_bracket + 1 : bracket_end]

        if parent_path:
            parent_obj = _process_path_segment(obj, parent_path)

        # Set the value using bracket access
        if bracket_content.startswith("'") and bracket_content.endswith("'"):
            key = bracket_content[1:-1]
            parent_obj[key] = value
        elif bracket_content.startswith('"') and bracket_content.endswith('"'):
            key = bracket_content[1:-1]
            parent_obj[key] = value
        elif bracket_content.lstrip("-").isdigit():
            index = int(bracket_content)
            parent_obj[index] = value
        else:
            try:
                key = eval(bracket_content)
                parent_obj[key] = value
            except:
                parent_obj[bracket_content] = value
    else:
        # No brackets, treat as simple attribute
        if path.lstrip("-").isdigit():
            obj[int(path)] = value
        else:
            setattr(obj, path, value)


def update_scheduled_params(obj, scheduler_dict, step, split_char="@"):
    scheduled_params_dict = {}
    for target, cfg in scheduler_dict.items():
        sch_type = cfg["type"]
        val_type = cfg.get("val_type", "float")
        target_attr = target
        target_obj = obj
        if split_char in target:
            target_obj_str, target_attr = target.rsplit(split_char, 1)
            target_obj = _navigate_object_path(obj, target_obj_str, split_char)
        if sch_type == "linear":
            i = len(cfg["seg_vals"]) - 1
            while step < cfg["seg_steps"][i]:
                i -= 1
            if i == len(cfg["seg_vals"]) - 1:
                val = cfg["seg_vals"][i]
            else:
                t = (step - cfg["seg_steps"][i]) / (cfg["seg_steps"][i + 1] - cfg["seg_steps"][i])
                t = max(0.0, min(1.0, t))
                val = (1.0 - t) * cfg["seg_vals"][i] + t * cfg["seg_vals"][i + 1]
        elif sch_type == "segment":
            i = len(cfg["seg_vals"]) - 1
            while step < cfg["seg_steps"][i]:
                i -= 1
            val = cfg["seg_vals"][i]

        val = eval(val_type)(val)

        if type(val) is DictConfig or type(val) is dict:
            # Handle complex path for dict/config access
            tmp_obj = _get_final_target(target_obj, target_attr)

            if cfg.get("overwrite_dict", False):
                _set_final_target(target_obj, target_attr, val)
            else:
                for k, v in val.items():
                    if type(tmp_obj) is dict:
                        tmp_obj[k] = v
                    else:
                        setattr(tmp_obj, k, v)
        else:
            # Handle complex path for direct value assignment
            _set_final_target(target_obj, target_attr, val)

        scheduled_params_dict[target] = val

        if "trigger_func" in cfg and step == cfg["seg_steps"][i]:
            target_func = cfg["trigger_func"]
            print(f"Triggering function: {target_func}")
            if split_char in target_func:
                target_obj_str, target_func_name = target_func.rsplit(split_char, 1)
                target_obj = _navigate_object_path(obj, target_obj_str, split_char)
            else:
                target_obj = obj
                target_func_name = target_func
            getattr(target_obj, target_func_name)()

    return scheduled_params_dict


class WarmupCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        final_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.final_lr = final_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.num_warmup_steps:
            return [
                base_lr * float(current_step) / float(max(1, self.num_warmup_steps))
                for base_lr in self.base_lrs
            ]
        else:
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return [
                self.final_lr + (base_lr - self.final_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


if __name__ == "__main__":
    # Test the complex path navigation
    class MockEventManager:
        def __init__(self):
            self.configs = {
                "push_robot": {"params": {"velocity_range": {"x": [1.0, 2.0], "y": [0.5, 1.5]}}}
            }

        def get_term_cfg(self, term_name):
            return self.configs[term_name]

    class MockEnv:
        def __init__(self):
            self.event_manager = MockEventManager()

    class MockSimulator:
        def __init__(self):
            self.env = MockEnv()

    # Test complex path navigation
    mock_obj = MockSimulator()

    # Test the path: env@event_manager@get_term_cfg('push_robot')@params@velocity_range@x@0
    test_path = "env@event_manager@get_term_cfg('push_robot')['params']['velocity_range']['x'][0]"

    # Create a simple scheduler config to test
    scheduler_config = {
        test_path: {"type": "linear", "seg_steps": [0, 100], "seg_vals": [5.0, 10.0]}
    }

    # Test the function
    print("Testing complex path navigation...")
    print(
        f"Original value: {mock_obj.env.event_manager.get_term_cfg('push_robot')['params']['velocity_range']['x'][0]}"
    )

    result = update_scheduled_params(mock_obj, scheduler_config, 50)
    print(
        f"Updated value: {mock_obj.env.event_manager.get_term_cfg('push_robot')['params']['velocity_range']['x'][0]}"
    )
    print(f"Scheduler result: {result}")

    # Test with step that triggers second segment
    result2 = update_scheduled_params(mock_obj, scheduler_config, 150)
    print(
        f"Updated value (step 150): {mock_obj.env.event_manager.get_term_cfg('push_robot')['params']['velocity_range']['x'][0]}"
    )
    print(f"Scheduler result: {result2}")

    print("\nOriginal learning rate scheduler test:")

    class YourModel(torch.nn.Module):
        def __init__(self):
            super(YourModel, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = YourModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    num_warmup_steps = 1000
    num_training_steps = 10000
    final_lr = 0.0001

    scheduler = WarmupCosineScheduler(optimizer, num_warmup_steps, num_training_steps, final_lr)

    lrs = []
    for step in range(num_training_steps):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])

    # Plotting the learning rate vs training steps
    import matplotlib.pyplot as plt

    plt.plot(range(num_training_steps), lrs)
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs Training Steps")
    # plt.show()
    plt.savefig("out/lr_vs_steps.png")
