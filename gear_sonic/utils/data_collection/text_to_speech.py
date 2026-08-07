"""Optional text-to-speech feedback for data collection."""

import threading


class TextToSpeech:
    def __init__(self, rate: int = 150, volume: float = 1.0):
        try:
            import pyttsx3

            self.engine = pyttsx3.init(driverName="espeak")
            self.engine.setProperty("rate", rate)
            self.engine.setProperty("volume", volume)
        except Exception as e:
            print(f"[Text To Speech] Initialization failed: {e}")
            self.engine = None
        self._speech_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def say(self, message: str, blocking: bool = False):
        if self.engine:
            if blocking:
                self._say_blocking(message)
            else:
                thread = threading.Thread(target=self._say_blocking, args=(message,), daemon=True)
                thread.start()
                self._speech_thread = thread

    def _say_blocking(self, message: str):
        with self._lock:
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except RuntimeError:
                pass

    def wait_for_completion(self):
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join()

    def print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        print(message)
        if say and self.engine is not None:
            self.say(message, blocking=blocking)
