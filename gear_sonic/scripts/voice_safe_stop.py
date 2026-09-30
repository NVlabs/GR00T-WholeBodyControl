#!/usr/bin/env python3
"""Voice safe stop: say "stop" -> the robot's safe stop (same as pressing k in deploy).

Listens to a microphone on this PC, recognises a few command words offline
(Vosk, small English model, restricted vocabulary) and publishes the safe-stop
command on its own ZMQ socket. deploy (--input-type zmq_manager) connects to it
on <--zmq-host>:5570, topic "safety" (see SAFE_STOP.md).

    say "hold on" or "stop"                      -> safe stop
    (chosen on the G1's own mic: the mic is muffled, and words carried by s/t/f
    sounds get lost; "hold on" 10/10 + "stop" 10/10 at close range, 0 false
    triggers in normal conversation. "let go" is NOT used: "hello" was heard as it.)
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

On the robot, use the G1's built-in microphone array instead of a sound card:
    python gear_sonic/scripts/voice_safe_stop.py --g1-mic --dry-run --verbose
The G1's voice service streams the mic as UDP multicast 239.168.123.161:5555
(16 kHz, s16le, mono, 5120-byte packets) on the robot network (192.168.123.x);
it only streams while the voice service's mic mode is on.
"""

import argparse
import json
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
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


G1_MIC_GROUP = "239.168.123.161"
G1_MIC_PORT = 5555


def robot_ip():
    """This machine's address on the robot network (192.168.123.x)."""
    out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    return next((ip for ip in out if ip.startswith("192.168.123.")), None)


def start_g1_mic(blocks, iface_ip):
    """Receive the G1 microphone multicast into `blocks` (same format as the sound card path)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", G1_MIC_PORT))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                    socket.inet_aton(G1_MIC_GROUP) + socket.inet_aton(iface_ip))
    sock.settimeout(1.0)

    def loop():
        silent = 0
        while True:
            try:
                data, _ = sock.recvfrom(65536)
                silent = 0
                blocks.put((data[:len(data) // 2 * 2], time.time()))
            except socket.timeout:
                silent += 1
                if silent == 3:
                    print("[voice] no G1 mic packets for 3 s (is the voice service's mic mode on?)")

    threading.Thread(target=loop, daemon=True).start()


def run_mic(args, sender, stop_words, release_words):
    try:
        import vosk
    except ImportError as e:
        sys.exit(f"missing package ({e}); install with: uv pip install vosk")
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
    words_msg = (f"stop words: {', '.join(stop_words)}"
                 + (f"; release words: {', '.join(release_words)}" if release_words else "; release = u in deploy"))

    if args.g1_mic:
        iface_ip = args.g1_iface_ip or robot_ip()
        if not iface_ip:
            sys.exit("no 192.168.123.x address found; run this on the robot or pass --g1-iface-ip")
        start_g1_mic(blocks, iface_ip)
        print(f"[voice] listening to the G1 microphone ({G1_MIC_GROUP}:{G1_MIC_PORT} via {iface_ip}); {words_msg}")
        stream = None
    else:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            sys.exit(f"sound card input unavailable ({e}); install: uv pip install sounddevice "
                     "+ sudo apt install libportaudio2, or use --g1-mic on the robot")

        def on_audio(indata, frames, t, status):  # audio thread
            if status:
                print(f"[voice] audio: {status}", file=sys.stderr)
            blocks.put((bytes(indata), time.time()))

        print(f"[voice] listening (device {args.device if args.device is not None else 'default'}); {words_msg}")
        stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * args.block),
                                   device=args.device, dtype="int16", channels=1, callback=on_audio)
        stream.start()
    try:
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
    finally:
        if stream is not None:
            stream.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5570, help="PUB port (deploy --safe-stop-voice-port)")
    ap.add_argument("--model", default=None,
                    help="Vosk model folder (default: ~/vosk-model-small-en-us-0.15 or ~/yara_sonic/...)")
    ap.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--g1-mic", action="store_true",
                    help="use the G1's built-in microphone (UDP multicast, run on the robot)")
    ap.add_argument("--g1-iface-ip", default=None, help="this machine's 192.168.123.x address (auto)")
    ap.add_argument("--stop-words", default="hold on,stop", help="comma-separated phrases")
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

    if args.model is None:
        candidates = ["~/vosk-model-small-en-us-0.15", "~/yara_sonic/vosk-model-small-en-us-0.15"]
        args.model = next((c for c in candidates if os.path.isdir(os.path.expanduser(c))), candidates[0])
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
