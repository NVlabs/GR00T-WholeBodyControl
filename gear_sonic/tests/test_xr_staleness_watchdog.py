import unittest

from gear_sonic.scripts.pico_manager_thread_server import XRStalenessWatchdog


class XRStalenessWatchdogTest(unittest.TestCase):
    def test_does_not_estop_before_first_sample(self):
        watchdog = XRStalenessWatchdog(warn_ms=50.0, estop_ms=200.0)

        self.assertEqual(watchdog.poll(None, 1_000_000_000, active=True), "ok")

    def test_reports_warn_and_estop_after_valid_sample(self):
        watchdog = XRStalenessWatchdog(warn_ms=50.0, estop_ms=200.0)
        sample_ns = 1_000_000_000

        self.assertEqual(watchdog.poll(sample_ns, sample_ns + 10_000_000, active=True), "ok")
        self.assertEqual(
            watchdog.poll(sample_ns, sample_ns + 60_000_000, active=True), "warn"
        )
        self.assertEqual(
            watchdog.poll(sample_ns, sample_ns + 250_000_000, active=True), "estop"
        )

    def test_reset_when_inactive(self):
        watchdog = XRStalenessWatchdog(warn_ms=50.0, estop_ms=200.0)
        sample_ns = 1_000_000_000

        self.assertEqual(
            watchdog.poll(sample_ns, sample_ns + 250_000_000, active=True), "estop"
        )
        self.assertEqual(
            watchdog.poll(sample_ns, sample_ns + 250_000_000, active=False), "idle"
        )
        self.assertEqual(watchdog.poll(None, sample_ns + 300_000_000, active=True), "ok")


if __name__ == "__main__":
    unittest.main()
