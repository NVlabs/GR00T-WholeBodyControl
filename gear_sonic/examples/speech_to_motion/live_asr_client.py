# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream live microphone audio to a Riva ASR NIM. On each finalized
transcript, send it to motion_controller.py, which generates the motion via
KIMODO, safety-checks it, and streams it to GEAR-SONIC.

Requires a running Riva ASR NIM (see the "Riva" prerequisite admonition in
the tutorial) and motion_controller.py already running and reachable.

Examples:
    # everything on localhost (single-machine setup)
    python live_asr_client.py

    # Riva and the motion controller on a remote machine
    python live_asr_client.py --riva-server 10.0.0.5:50051 --controller-host 10.0.0.5
"""
import argparse
import json
import os
import queue
import re
import socket
import sys
import time

import riva.client
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100ms chunks
MIN_CHARS = 3  # ignore trivial/noise finals like "mm", "uh"

# Robot-command vocabulary to bias recognition toward. Boosting helps but
# does not guarantee correct recognition of every command word, e.g. "squat".
BOOSTED_WORDS = [
    "walk", "step", "steps", "forward", "backward", "turn", "left", "right",
    "squat", "wave", "stop", "crouch", "approach", "times",
    "one", "two", "three", "four", "five",
]
BOOSTED_SCORE = 8.0

# Known, consistent ASR substitution error seen across repeated tests on the
# same phrase: "squat" -> "spot" happened identically in separate recordings,
# so it's corrected here rather than chasing it with more boost-score tuning.
KNOWN_CORRECTIONS = {
    "spot": "squat",
}


def apply_known_corrections(transcript: str) -> str:
    def _replace(match: "re.Match") -> str:
        word = match.group(0)
        corrected = KNOWN_CORRECTIONS[word.lower()]
        return corrected.capitalize() if word[0].isupper() else corrected

    pattern = r"\b(" + "|".join(re.escape(w) for w in KNOWN_CORRECTIONS) + r")\b"
    return re.sub(pattern, _replace, transcript, flags=re.IGNORECASE)


audio_q: "queue.Queue[bytes]" = queue.Queue()


def _callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(bytes(indata))


def audio_chunk_generator():
    while True:
        yield audio_q.get()


def trigger_motion_controller(prompt: str, controller_addr: tuple) -> None:
    """Send the transcript to motion_controller.py. The controller replies
    immediately with {"status": "processing"} and does generation/safety-gate/
    streaming asynchronously -- the actual outcome (streamed / rejected by
    the safety gate / superseded by a newer command) only shows up in the
    controller's own log, not in this response."""
    print(f'\n>>> Sending to motion controller: "{prompt}"')
    t0 = time.time()
    try:
        s = socket.create_connection(controller_addr, timeout=30)
        s.sendall((json.dumps({"prompt": prompt, "duration": 5}) + "\n").encode("utf-8"))
        resp = s.recv(65536)
        s.close()
        result = json.loads(resp.decode("utf-8"))
        elapsed = time.time() - t0
        if result.get("status") in ("processing", "ok"):
            print(f">>> Accepted in {elapsed:.2f}s -- generation/safety-gate/streaming running async, check motion_controller's log for outcome\n")
        else:
            print(f">>> REJECTED at handoff: {result.get('error')}\n")
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f">>> Could not reach motion controller at {controller_addr}: {e}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--riva-server", default=os.environ.get("RIVA_SERVER", "localhost:50051"), help="Riva ASR NIM gRPC endpoint (default: $RIVA_SERVER or localhost:50051)")
    parser.add_argument("--controller-host", default=os.environ.get("CONTROLLER_HOST", "localhost"), help="motion_controller.py host (default: $CONTROLLER_HOST or localhost)")
    parser.add_argument("--controller-port", type=int, default=8767, help="motion_controller.py port (default: 8767)")
    args = parser.parse_args()
    controller_addr = (args.controller_host, args.controller_port)

    auth = riva.client.Auth(uri=args.riva_server, use_ssl=False)
    asr_service = riva.client.ASRService(auth)

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    streaming_config = riva.client.StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    riva.client.add_word_boosting_to_config(streaming_config, BOOSTED_WORDS, BOOSTED_SCORE)

    print(f"Connecting to Riva ASR at {args.riva_server} ...")
    print(f"Motion controller: {controller_addr[0]}:{controller_addr[1]}")
    print("Speak a command, e.g. 'walk forward and wave' (Ctrl+C to stop)...\n")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SAMPLES,
        dtype="int16",
        channels=1,
        callback=_callback,
    ):
        responses = asr_service.streaming_response_generator(
            audio_chunks=audio_chunk_generator(),
            streaming_config=streaming_config,
        )
        for response in responses:
            for result in response.results:
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript.strip()
                if not transcript:
                    continue
                if result.is_final:
                    print(f"[FINAL] {transcript}")
                    corrected = apply_known_corrections(transcript)
                    if corrected != transcript:
                        print(f"[corrected] {corrected}")
                    if len(corrected) >= MIN_CHARS:
                        trigger_motion_controller(corrected, controller_addr)
                else:
                    print(f"  ...{transcript}", end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
