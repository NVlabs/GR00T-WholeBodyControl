"""Real C++ gateway + ZMQ client against local PTYs, never robot devices."""

import os
import select
import socket
import struct
import subprocess
import threading
import time

import numpy as np
import pytest
import yaml

from gear_sonic.utils.hand_control.inspire.client import InspireRealBackend
from gear_sonic.utils.hand_control.inspire.config import DEFAULT_MAPPING_PATH, driver_command


class SerialEmulator:
    def __init__(self, hand_id):
        self.master, slave = os.openpty()
        self.path = os.ttyname(slave)
        os.close(slave)
        self.hand_id = hand_id
        self.positions = [1000] * 6
        self.speed = [400] * 6
        self.read_times, self.writes = [], []
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        pending = bytearray()
        try:
            while not self.stopped.is_set():
                if not select.select([self.master], [], [], 0.02)[0]:
                    continue
                try:
                    pending.extend(os.read(self.master, 4096))
                except OSError:
                    self.stopped.wait(0.005)  # Slave not yet opened, or already closed.
                    continue
                while len(pending) >= 4 and len(pending) >= pending[3] + 5:
                    size = pending[3] + 5
                    request = bytes(pending[:size])
                    del pending[:size]
                    assert request[:2] == b"\xeb\x90" and request[2] == self.hand_id
                    assert request[-1] == sum(request[2:-1]) % 256
                    address = int.from_bytes(request[5:7], "little")
                    if request[4] == 0x12:
                        if address == 0x05CE:
                            self.positions = list(struct.unpack("<6H", request[7:19]))
                            self.writes.append((time.monotonic(), self.positions.copy()))
                        elif address == 0x05F2:
                            self.speed = list(struct.unpack("<6H", request[7:19]))
                        else:
                            assert address == 0x03EC
                        response = bytearray(
                            [0x90, 0xEB, self.hand_id, 4, 0x12, request[5], request[6], 1, 0]
                        )
                    else:
                        assert request[4] == 0x11
                        if address == 0x060A:
                            self.read_times.append(time.monotonic())
                            values = self.positions
                        else:
                            assert address == 0x05F2
                            values = self.speed
                        response = bytearray(
                            [0x90, 0xEB, self.hand_id, 15, 0x11, request[5], request[6]]
                        )
                        response.extend(struct.pack("<6H", *values))
                        response.append(0)
                    response[-1] = sum(response[2:-1]) % 256
                    os.write(self.master, response)
        except BaseException as exc:
            self.error = exc

    def close(self):
        self.stopped.set()
        self.thread.join(1)
        os.close(self.master)
        if self.error:
            raise self.error


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


def test_native_50hz_session_feedback_and_disarm(tmp_path):
    binary = os.environ.get("INSPIRE_TEST_GATEWAY")
    if not binary:
        pytest.skip("set INSPIRE_TEST_GATEWAY to a compiled native driver for the PTY test")
    left, right = SerialEmulator(2), SerialEmulator(3)
    cfg = yaml.safe_load(DEFAULT_MAPPING_PATH.read_text())
    cfg["hardware"].update(
        left_device=left.path,
        right_device=right.path,
        left_id=2,
        right_id=3,
        command_endpoint=endpoint(),
        state_endpoint=endpoint(),
        lock_path=str(tmp_path / "writer.lock"),
    )
    assert cfg["hardware"]["command_rate_hz"] == 50
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log = tmp_path / "gateway.log"
    hand = InspireRealBackend(path)
    process = None
    try:
        with log.open("w") as output:
            process = subprocess.Popen(
                driver_command(binary, path),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, G1_RL_SESSION_CONTROL="1"),
            )
            deadline = time.monotonic() + 5
            while (
                time.monotonic() < deadline and "DIRECT_SERIAL_STATE_READY" not in log.read_text()
            ):
                assert process.poll() is None, log.read_text()
                time.sleep(0.02)
            assert "DIRECT_SERIAL_STATE_READY" in log.read_text(), log.read_text()
            hand.connect()
            assert hand.read_state().error is None
            assert not left.writes and not right.writes  # connect never submits a position.
            hand.set_target([0] * 6, [0] * 6)
            deadline = time.monotonic()
            for i in range(1, 31):
                deadline += 0.02
                time.sleep(max(0, deadline - time.monotonic()))
                hand.set_target([i * 0.004] * 6, [i * 0.002] * 6)
            time.sleep(0.08)
            feedback = hand.read_state()
            assert feedback.error is None
            np.testing.assert_allclose(feedback.left_position, [0.12] * 6, atol=0.002)
            np.testing.assert_allclose(feedback.right_position, [0.06] * 6, atol=0.002)
            assert hand._session.rows["hand"][0]["accepted_command_sequence"] is not None
            for emulator in (left, right):
                times = emulator.read_times[-30:]
                assert len(times) == 30
                measured_hz = 1 / float(np.median(np.diff(times)))
                assert 40 < measured_hz < 60, measured_hz
                # With a 50 Hz producer, serial writes must also exceed the old 20 Hz.
                writes = [t for t, _ in emulator.writes]
                assert len(writes) >= 22, len(writes)
                assert (len(writes) - 1) / (writes[-1] - writes[0]) > 30
            hand.stop()
            hand.stop()
            assert hand._session.rows["hand"][0]["status"] == "DISARMED"
            counts = len(left.writes), len(right.writes)
            time.sleep(0.08)
            assert counts == (len(left.writes), len(right.writes))
            with pytest.raises(RuntimeError, match="stopped"):
                hand.set_target([0] * 6, [0] * 6)
            hand.close()
            hand.close()
            # Explicit reconnect starts a new owner; no hidden RESET or legacy fallback.
            hand.connect()
            assert hand.read_state().error is None
            hand.close()
            process.terminate()
            assert process.wait(5) == 0, log.read_text()
        assert left.speed == right.speed == [400] * 6  # source driver restores speed on exit
        assert "serial_rate_hz=50" in log.read_text()
        assert "command_watchdog=disabled" in log.read_text()
    finally:
        try:
            hand.close()
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(5)
            left.close()
            right.close()
