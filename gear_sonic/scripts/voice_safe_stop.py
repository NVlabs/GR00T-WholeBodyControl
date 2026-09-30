#!/usr/bin/env python3
"""Voice safe stop: say "stop" -> the robot's safe stop (same as pressing k in deploy).

Listens to a microphone on this PC, recognises a few command words offline
(Vosk, small English model, restricted vocabulary) and publishes the safe-stop
command on its own ZMQ socket. deploy (--input-type zmq_manager) connects to it
on <--zmq-host>:5570, topic "safety" (see SAFE_STOP.md).

    say "stop" (or "robot stop", "freeze")      -> safe stop
    release: press u in the deploy terminal (default). Voice release only with
    --allow-release: say "release" / "continue".

Setup (once, on the PC with the microphone; teleop venv):
    uv pip install vosk sounddevice          # or: pip install vosk sounddevice
    sudo apt install libportaudio2           # if sounddevice cannot find PortAudio
    cd ~/yara_sonic && wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip \\
        && unzip vosk-model-small-en-us-0.15.zip

Run (from the repo root):
    python gear_sonic/scripts/voice_safe_stop.py --list-devices      # find the mic
    python gear_sonic/scripts/voice_safe_stop.py --device 3          # listen
    python gear_sonic/scripts/voice_safe_stop.py --typed             # no mic: type words (tests the chain)
    python gear_sonic/scripts/voice_safe_stop.py --dry-run           # recognise only, send nothing

Later the robot's own microphone can replace the PC mic: only the audio source
(read_audio) changes; the recogniser and the ZMQ message stay the same.
"""

import argparse
import json
import os
import queue
import struct
import sys
import time

import zmq

HEADER_SIZE = 1280  # must match zmq_packed_message_subscriber.hpp
SAMPLE_RATE = 16000


def safety_message(field):
    """Packed ZMQ message: topic 'safety', one u8 field (safe_stop / safe_release) = 1."""
    header = json.dumps({"v": 1, "endian": "le", "count": 1,
                         "fields": [{"name": field, "dtype": "u8", "shape": [1]}]},
                        separators=(",", ":")).encode()
    return b"safety" + header.ljust(HEADER_SIZE, b"\x00") + struct.pack("B", 1)


class Sender:
    def __init__(self, port, dry_run, cooldown):
        self.dry_run = dry_run
        self.cooldown = cooldown
        self.last = {}
        self.sock = None
        if not dry_run:
            self.sock = zmq.Context.instance().socket(zmq.PUB)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.bind(f"tcp://*:{port}")
            print(f"[voice] publishing on tcp://*:{port} topic 'safety' "
                  f"(deploy: --zmq-host <this PC>, --safe-stop-voice-port {port})")

    def send(self, field, heard, t_heard):
        now = time.time()
        if now - self.last.get(field, 0.0) < self.cooldown:
            return
        self.last[field] = now
        tag = "STOP" if field == "safe_stop" else "RELEASE"
        if self.dry_run:
            print(f"\a[voice] heard '{heard}' -> {tag} (dry run, nothing sent)")
            return
        # A few copies: PUB/SUB drops messages sent before the subscriber is connected.
        for _ in range(3):
            self.sock.send(safety_message(field))
            time.sleep(0.01)
        print(f"\a[voice] heard '{heard}' -> {tag} sent ({(now - t_heard) * 1000:.0f} ms after the audio block)")


def match(text, phrases):
    text = " " + " ".join(text.split()) + " "
    return next((p for p in phrases if f" {p} " in text), None)


def run_typed(sender, stop_words, release_words):
    print("[voice] typed mode: type a phrase + Enter (Ctrl+D to quit)")
    for line in sys.stdin:
        heard = line.strip().lower()
        t = time.time()
        if match(heard, stop_words):
            sender.send("safe_stop", heard, t)
        elif release_words and match(heard, release_words):
            sender.send("safe_release", heard, t)
        else:
            print(f"[voice] '{heard}': no command")


def run_mic(args, sender, stop_words, release_words):
    try:
        import sounddevice as sd
        import vosk
    except ImportError as e:
        sys.exit(f"missing package ({e}); install with: uv pip install vosk sounddevice")
    model_path = os.path.expanduser(args.model)
    if not os.path.isdir(model_path):
        sys.exit(f"Vosk model not found at {model_path} (see the setup lines at the top of this file)")
    vosk.SetLogLevel(-1)
    model = vosk.Model(model_path)
    # Restricted vocabulary: everything else decodes to [unk] -> far fewer false stops.
    vocab = sorted(set(" ".join(stop_words + release_words).split()))
    grammar = json.dumps(vocab + ["[unk]"])
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE, grammar)
    rec.SetWords(True)

    blocks = queue.Queue()

    def on_audio(indata, frames, t, status):  # audio thread
        if status:
            print(f"[voice] audio: {status}", file=sys.stderr)
        blocks.put((bytes(indata), time.time()))

    print(f"[voice] listening (device {args.device if args.device is not None else 'default'}); "
          f"stop words: {', '.join(stop_words)}"
          + (f"; release words: {', '.join(release_words)}" if release_words else "; release = u in deploy"))
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * args.block),
                           device=args.device, dtype="int16", channels=1, callback=on_audio):
        while True:
            data, t = blocks.get()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                words = [w for w in res.get("result", []) if w.get("conf", 0.0) >= args.min_conf]
                heard = " ".join(w["word"] for w in words)
                if args.verbose and res.get("text"):
                    print(f"[voice] final: '{res['text']}' (kept: '{heard}')")
                if heard and match(heard, stop_words):
                    sender.send("safe_stop", heard, t)
                elif heard and release_words and match(heard, release_words):
                    sender.send("safe_release", heard, t)
            elif args.fast:
                # Partial result: a stop word already recognised mid-utterance -> act now.
                part = json.loads(rec.PartialResult()).get("partial", "")
                if part and match(part, stop_words):
                    sender.send("safe_stop", part + " (partial)", t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5570, help="PUB port (deploy --safe-stop-voice-port)")
    ap.add_argument("--model", default="~/yara_sonic/vosk-model-small-en-us-0.15", help="Vosk model folder")
    ap.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--stop-words", default="stop,robot stop,freeze", help="comma-separated phrases")
    ap.add_argument("--allow-release", action="store_true", help="also release by voice (off by default)")
    ap.add_argument("--release-words", default="release,continue")
    ap.add_argument("--min-conf", type=float, default=0.6, help="min word confidence (final results)")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="act only on final results (slower, fewer false stops)")
    ap.add_argument("--block", type=float, default=0.1, help="audio block length (s)")
    ap.add_argument("--cooldown", type=float, default=1.0, help="ignore repeats for this long (s)")
    ap.add_argument("--typed", action="store_true", help="no microphone: type phrases")
    ap.add_argument("--dry-run", action="store_true", help="recognise only, send nothing")
    ap.add_argument("--verbose", action="store_true", help="print every recognised phrase")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    stop_words = [w.strip().lower() for w in args.stop_words.split(",") if w.strip()]
    release_words = ([w.strip().lower() for w in args.release_words.split(",") if w.strip()]
                     if args.allow_release else [])
    sender = Sender(args.port, args.dry_run, args.cooldown)
    try:
        if args.typed:
            run_typed(sender, stop_words, release_words)
        else:
            run_mic(args, sender, stop_words, release_words)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
