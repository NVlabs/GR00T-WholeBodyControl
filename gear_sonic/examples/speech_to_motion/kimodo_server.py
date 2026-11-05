# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent KIMODO inference server (speech-to-motion path).

Loads a KIMODO text-to-motion model once and keeps it resident on the GPU,
then serves generation requests over a simple newline-delimited JSON TCP
protocol. This avoids the ~1 minute per-call model-reload cost of spawning a
fresh `kimodo_gen` CLI process for every voice command.

Requires KIMODO (https://github.com/nv-tlabs/kimodo) installed separately in
its own environment. Run this script inside that environment -- it does not
need $KIMODO_ROOT on sys.path, `pip install -e .` from the KIMODO repo makes
the `kimodo` package importable directly.

Examples:
    # defaults: 0.0.0.0:8766, kimodo-g1-rp, CSVs written to ./outputs
    python kimodo_server.py

    # custom model / output location
    python kimodo_server.py --model kimodo-g1-rp --output-dir /data/kimodo_outputs
"""
import argparse
import json
import os
import re
import socket
import socketserver
import threading
import time

import torch

from kimodo import load_model
from kimodo.exports.mujoco import MujocoQposConverter

# Splits a compound command into separate action clauses so each gets its own
# entry in KIMODO's multi-prompt/transition sequence, instead of the whole
# sentence being generated as one undivided clip (which is what happens if we
# only split on periods -- real ASR transcripts only punctuate at the very
# end of the utterance, not between clauses, so periods alone never split a
# compound command). Found via a real test: "Walk two steps forward, then
# turn right and do one squat and wave two times and stop." generated as one
# blended, mushy motion (no real squat) when treated as a single clip; this
# splits it into 5 separate staged clauses instead.
#
# Also split on bare commas: Riva sometimes punctuates a clause boundary with
# "," instead of "and"/"then" (e.g. "...forward, then turn left, wave two
# times and stop" -- observed real transcript), and splitting on connectives
# alone silently merges two distinct actions ("turn left" + "wave two times")
# into one clause, confusing KIMODO with a blended text it was never meant to
# generate as a single continuous motion.
_CONNECTIVES = re.compile(r"\bthen\b|\band\b|,", flags=re.IGNORECASE)
PER_CLAUSE_SECONDS = 2.0
MIN_CLAUSE_SECONDS = 1.0


def segment_command(prompt: str) -> list:
    text = prompt.strip().rstrip(".")
    parts = _CONNECTIVES.split(text)
    clauses = [p.strip(" ,") for p in parts if p.strip(" ,")]
    return clauses or [text]


class KimodoServer:
    def __init__(self, model_name: str, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Loading {model_name} on {self._device} ...")
        t0 = time.time()
        self.model, _ = load_model(
            model_name, device=self._device, default_family="Kimodo", return_resolved_name=True
        )
        self.converter = MujocoQposConverter(self.model.skeleton)
        print(f"Model loaded in {time.time() - t0:.1f}s. fps={self.model.fps}. Ready.")

    def generate(self, prompt: str, duration: float = 5.0, diffusion_steps: int = 100) -> dict:
        clauses = segment_command(prompt)
        texts = [c + "." for c in clauses]
        if len(texts) > 1:
            per_clause = max(MIN_CLAUSE_SECONDS, PER_CLAUSE_SECONDS)
            num_frames = [int(per_clause * self.model.fps)] * len(texts)
        else:
            num_frames = [int(duration * self.model.fps)]

        t0 = time.time()
        with self._lock:
            output = self.model(
                texts,
                num_frames,
                constraint_lst=[],
                num_denoising_steps=diffusion_steps,
                num_samples=1,
                multi_prompt=True,
                num_transition_frames=5,
                post_processing=False,
                return_numpy=True,
            )
            qpos = self.converter.dict_to_qpos(output, self._device)
        gen_time = time.time() - t0

        stem = "gen_" + time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, stem + ".csv")
        self.converter.save_csv(qpos, csv_path)

        return {
            "status": "ok",
            "prompt": prompt,
            "clauses": clauses,
            "output_csv": csv_path,
            "num_frames": sum(num_frames),
            "fps": self.model.fps,
            "generation_seconds": round(gen_time, 2),
        }


def make_handler(server: KimodoServer):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                line = self.rfile.readline()
                if not line:
                    return
                req = json.loads(line.decode("utf-8"))
                prompt = req.get("prompt", "")
                duration = float(req.get("duration", 5.0))
                diffusion_steps = int(req.get("diffusion_steps", 100))
                print(f"[request] prompt={prompt!r} duration={duration}")
                try:
                    result = server.generate(prompt, duration, diffusion_steps)
                    if result.get("status") == "ok":
                        print(f"[clauses] {result['clauses']}")
                except Exception as e:  # noqa: BLE001
                    result = {"status": "error", "error": str(e)}
                self.wfile.write((json.dumps(result) + "\n").encode("utf-8"))
            except Exception as e:  # noqa: BLE001
                print(f"[handler error] {e}")

    return Handler


class TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8766, help="TCP port (default: 8766)")
    parser.add_argument("--model", default="kimodo-g1-rp", help="KIMODO model name (default: kimodo-g1-rp)")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("KIMODO_OUTPUT_DIR", os.path.join(os.getcwd(), "outputs")),
        help="where generated qpos CSVs are written (default: $KIMODO_OUTPUT_DIR or ./outputs)",
    )
    args = parser.parse_args()

    server = KimodoServer(args.model, args.output_dir)
    with TCPServer((args.host, args.port), make_handler(server)) as tcp_server:
        print(f"KIMODO server listening on {args.host}:{args.port}")
        tcp_server.serve_forever()


if __name__ == "__main__":
    main()
