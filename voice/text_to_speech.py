# voice/text_to_speech.py
# Uses macOS `say` command via subprocess — avoids pyttsx3 threading issues
# with FastAPI/uvicorn where runAndWait() can deadlock off the main thread.

import subprocess
import threading
import shutil


class TextToSpeech:

    def __init__(self):
        self._proc   = None          # current `say` subprocess
        self._lock   = threading.Lock()
        self._thread = None

        # Check say is available
        self._use_say = shutil.which("say") is not None
        if not self._use_say:
            print("[TTS] WARNING: macOS `say` not found. TTS disabled.")
        else:
            print("Text-to-Speech ready (macOS say).")

    def stop(self):
        """Kill the running `say` process immediately."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=1)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            self._proc = None

    def speak(self, text: str):
        """Blocking speak using macOS `say`."""
        if not text or not text.strip() or not self._use_say:
            return

        # Stop any currently running speech
        self.stop()

        # Clean text — remove markdown symbols that sound bad when spoken
        import re
        clean = re.sub(r'[#*`_~>]', '', text)
        clean = re.sub(r'https?://\S+', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        with self._lock:
            try:
                self._proc = subprocess.Popen(
                    ["say", "-v", "Samantha", "-r", "175", clean],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"[TTS] speak failed: {e}")
                self._proc = None
                return

        # Wait for process outside lock so stop() can interrupt
        if self._proc:
            try:
                self._proc.wait()
            except Exception:
                pass

    def speak_async(self, text: str):
        """Non-blocking speak — runs in background thread."""
        if not text or not text.strip():
            return

        # Stop current speech
        if self._thread and self._thread.is_alive():
            self.stop()
            self._thread.join(timeout=0.5)

        self._thread = threading.Thread(
            target=self.speak, args=(text,), daemon=True
        )
        self._thread.start()


if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Hello. Your AI assistant is ready.")