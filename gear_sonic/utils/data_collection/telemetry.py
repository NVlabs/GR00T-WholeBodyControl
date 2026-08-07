"""Timing and performance monitoring for data collection loops."""

from collections import deque
import time


class Telemetry:
    """Handles timing and performance monitoring for different code sections."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._current_timers = {}
        self._last_results = {}
        self._history = {}

    def start_timer(self, name: str):
        self._current_timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        if name not in self._current_timers:
            print(f"Warning: Telemetry Timer '{name}' stopped without being started.")
            return 0.0

        start_time = self._current_timers.pop(name)
        duration = time.perf_counter() - start_time
        self.record_value(name, duration)
        return duration

    class Timer:
        """Context manager for timing operations."""

        def __init__(self, telemetry, name: str):
            self.telemetry = telemetry
            self.name = name

        def __enter__(self):
            self.telemetry.start_timer(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            del exc_type, exc_val, exc_tb
            self.telemetry.stop_timer(self.name)
            return False

    def timer(self, name: str):
        """Returns a context manager for timing an operation."""
        return self.Timer(self, name)

    def record_value(self, name: str, value: float):
        self._last_results[name] = value
        if name not in self._history:
            self._history[name] = deque(maxlen=self.window_size)
        self._history[name].append(value)

    def get_last_timing(self) -> dict[str, float]:
        return self._last_results.copy()

    def clear_last_timing(self):
        self._last_results.clear()

    def get_average(self, name: str) -> float | None:
        if name not in self._history or not self._history[name]:
            return None
        return sum(self._history[name]) / len(self._history[name])

    def log_timing_info(
        self,
        context: str = "",
        threshold: float = 0.001,
        log_averages: bool = True,
    ):
        current_iteration_data = self.get_last_timing()
        significant_timings = {k: v for k, v in current_iteration_data.items() if v > threshold}

        should_log = bool(significant_timings)

        avg_data = {}
        if log_averages:
            for name in self._history:
                avg = self.get_average(name)
                if avg is not None:
                    avg_data[f"{name}_avg"] = avg
            if avg_data:
                should_log = True

        if not should_log:
            self.clear_last_timing()
            return

        log_lines = [f"\n{context} Timing breakdown:" if context else "\nTiming breakdown:"]

        if significant_timings:
            log_lines.append(f"  Current Iteration (> {threshold*1000:.1f}ms):")
            for name, duration in sorted(significant_timings.items()):
                log_lines.append(f"    {name}: {duration * 1000:.2f}ms")
        elif current_iteration_data:
            total_key = next((k for k in current_iteration_data if "total" in k), None)
            if total_key:
                log_lines.append(
                    f"  Current Iteration Total: {current_iteration_data[total_key] * 1000:.2f}ms"
                )

        if log_averages and avg_data:
            log_lines.append(f"  Moving Averages (last {self.window_size} iters):")
            for name, avg in sorted(avg_data.items()):
                log_lines.append(f"    {name}: {avg * 1000:.2f}ms")

        print("\n".join(log_lines))
        self.clear_last_timing()
