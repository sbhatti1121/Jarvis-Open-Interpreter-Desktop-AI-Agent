import os
import sys
import time
import asyncio
import tempfile
import threading
import subprocess
import shutil
from pathlib import Path

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import edge_tts
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QLineEdit, QComboBox, QCheckBox, QMessageBox,
    QFrame, QProgressBar
)

APP_NAME = "Voice Interpreter"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms at 16 kHz
WAKE_THRESHOLD = 0.55
WAKE_MODEL = "hey_jarvis"

def find_interpreter():
    """Find the installed Open Interpreter CLI without assuming a Python API."""
    candidates = []
    found = shutil.which("interpreter")
    if found:
        candidates.append(found)
    # Common Windows install locations; PATH is preferred.
    local = os.environ.get("LOCALAPPDATA", "")
    user = os.environ.get("USERPROFILE", "")
    for root in (local, user):
        if root:
            candidates += [
                str(Path(root) / "Programs" / "interpreter" / "interpreter.exe"),
                str(Path(root) / ".local" / "bin" / "interpreter.exe"),
                str(Path(root) / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "Scripts" / "interpreter.exe"),
            ]
    for item in candidates:
        if item and (Path(item).exists() or shutil.which(item)):
            return item
    return None

class AgentWorker(QThread):
    output = Signal(str)
    finished_text = Signal(str, int)

    def __init__(self, prompt, working_dir):
        super().__init__()
        self.prompt = prompt
        self.working_dir = working_dir

    def run(self):
        exe = find_interpreter()
        if not exe:
            self.finished_text.emit(
                "Could not find the Open Interpreter CLI. In PowerShell, run "
                "'interpreter --version'. If it works there, add its install folder to PATH "
                "and restart this app.", 127
            )
            return
        # Keep CLI diagnostics out of the user-facing conversation panel.
        try:
            # Current Open Interpreter CLI documents `interpreter exec "<prompt>"`.
            # Do not use the legacy Python `interpreter.chat()` API.
            jarvis_instructions = (
                "You are Jarvis, a capable Windows desktop assistant. Use Open Interpreter's available "
                "tools and PowerShell when useful to complete the user's request, including opening apps, "
                "browser tasks, downloading files, and managing files when explicitly requested. Use "
                "available tools only to carry out the user's intended "
                "task, and do not perform unrelated actions. For destructive or irreversible actions, "
                "confirm with the user first. Never claim success unless execution confirms it. The UI "
                "should receive only your final natural-language answer: do not include command output, "
                "tool traces, startup banners, token counts, session IDs, or internal reasoning in that answer. "
                "Keep normal conversational replies concise.\n\nUser request: "
            )
            # This installed CLI does not support --ask-for-approval; that flag causes
            # an argument-parsing error. Keep the app-level Yes/No task confirmation.
            # Preserve the sandbox option and let the installed CLI apply its own approval
            # configuration for individual commands.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with tempfile.TemporaryDirectory(prefix="voice-interpreter-") as output_dir:
                output_path = Path(output_dir) / "final-response.txt"
                cmd = [
                    exe, "exec", "--sandbox", "danger-full-access",
                    "--output-last-message", str(output_path),
                    jarvis_instructions + self.prompt
                ]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    cwd=self.working_dir, creationflags=creationflags,
                    bufsize=1
                )
                for _ in iter(proc.stdout.readline, ""):
                    pass
                code = proc.wait()
                result = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
            if not result:
                result = f"Open Interpreter did not produce a final response (exit code {code})."
            self.finished_text.emit(result, code)
        except Exception as exc:
            self.finished_text.emit(f"Could not start Open Interpreter: {exc}", 1)

class SpeechWorker(QThread):
    status = Signal(str)
    heard = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self._model = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            import openwakeword
            from openwakeword.model import Model
            self.status.emit("Checking/downloading wake-word models (first run may take a moment)…")
            # Some openWakeWord pip installs do not include pretrained model files.
            model_dir = Path(openwakeword.__file__).parent / "resources" / "models"
            expected = model_dir / "hey_jarvis_v0.1.onnx"
            if not expected.exists():
                openwakeword.utils.download_models()
            if not expected.exists():
                available = ", ".join(p.name for p in model_dir.glob("*jarvis*")) or "none"
                raise FileNotFoundError(
                    f"Hey Jarvis ONNX model was not installed at {expected}. "
                    f"Jarvis model files found: {available}. Reinstall/download openWakeWord models."
                )
            self._model = Model(wakeword_models=[str(expected)], inference_framework="onnx")
            self.status.emit('Listening for “Hey Jarvis”…')
            while not self._stop_event.is_set():
                triggered = False
                # Close the wake-word input stream before SpeechRecognition opens
                # the microphone; many Windows audio drivers allow only one stream.
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                    blocksize=FRAME_SAMPLES) as stream:
                    while not self._stop_event.is_set():
                        audio, overflowed = stream.read(FRAME_SAMPLES)
                        frame = np.asarray(audio).reshape(-1).astype(np.int16)
                        scores = self._model.predict(frame)
                        score = max((float(v) for v in scores.values()), default=0.0)
                        if score >= WAKE_THRESHOLD:
                            triggered = True
                            break
                if triggered and not self._stop_event.is_set():
                    self.status.emit("Wake word detected. Listening to your command…")
                    if hasattr(self._model, "reset"):
                        self._model.reset()
                    text = self._record_command()
                    if text:
                        self.heard.emit(text)
                    if not self._stop_event.is_set():
                        self.status.emit('Listening for “Hey Jarvis”…')
        except Exception as exc:
            self.error.emit(
                f"Voice listener error: {exc}\n\nCheck microphone permissions and install "
                "audio dependencies. You can still type commands in the app."
            )
            self.status.emit("Voice listener stopped.")

    def _record_command(self):
        """Collect a complete command until the user says 'done Jarvis'.

        SpeechRecognition records one utterance at a time; we transcribe each
        segment and keep listening across short pauses. A longer pause after
        the last segment submits the command automatically.
        """
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.0
        recognizer.non_speaking_duration = 0.4
        recognizer.dynamic_energy_threshold = True
        segments = []
        end_phrases = ("done jarvis", "that's all jarvis", "that is all jarvis")
        max_segments = 8
        try:
            with sr.Microphone(sample_rate=16000) as source:
                self.status.emit("Speak your full command. Say 'Done Jarvis' when finished.")
                recognizer.adjust_for_ambient_noise(source, duration=0.35)
                for _ in range(max_segments):
                    if self._stop_event.is_set():
                        break
                    try:
                        timeout = 5 if not segments else 1.5
                        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=12)
                    except sr.WaitTimeoutError:
                        if segments:
                            break
                        self.status.emit("No speech detected. Listening again…")
                        return ""
                    self.status.emit("Understanding your words…")
                    try:
                        part = recognizer.recognize_google(audio, language="en-US").strip()
                    except sr.UnknownValueError:
                        # Ignore a noisy segment and continue listening.
                        continue
                    if not part:
                        continue
                    segments.append(part)
                    lowered = " ".join(segments).lower().rstrip(" .,!?")
                    if any(lowered.endswith(phrase) for phrase in end_phrases):
                        break
                combined = " ".join(segments).strip()
                # Remove the spoken end phrase before sending the command to the AI.
                import re
                combined = re.sub(
                    r"\s*(?:done\s+jarvis|that'?s\s+all\s+jarvis|that\s+is\s+all\s+jarvis)[.!?]*\s*$",
                    "", combined, flags=re.IGNORECASE
                ).strip(" .,!?")
                if combined:
                    self.status.emit("Command captured. Processing…")
                return combined
        except sr.RequestError as exc:
            self.error.emit(f"Online speech recognition failed: {exc}")
        except Exception as exc:
            self.error.emit(f"Microphone/transcription error: {exc}")
        return ""

class VoiceOutputWorker(QThread):
    error = Signal(str)

    def __init__(self, text, voice="en-US-AriaNeural"):
        super().__init__()
        self.text = text[:3500]
        self.voice = voice

    def run(self):
        try:
            asyncio.run(self._speak())
        except Exception as exc:
            self.error.emit(f"Online text-to-speech failed: {exc}")

    async def _speak(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        try:
            communicate = edge_tts.Communicate(self.text, self.voice)
            await communicate.save(path)
            # pygame reliably plays the generated MP3 without launching an external player.
            # Playback runs in this worker, not the UI thread.
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            pygame.mixer.quit()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jarvis AI Assistant")
        self.resize(1080, 720)
        self.setMinimumSize(820, 560)
        self.worker = None
        self.agent_worker = None
        self.tts_worker = None
        self.listening = False
        self.workspace = str(Path.home() / "VoiceInterpreterWorkspace")
        Path(self.workspace).mkdir(parents=True, exist_ok=True)

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        top = QHBoxLayout()
        brand = QLabel("◉  JARVIS AI ASSISTANT")
        brand.setObjectName("brand")
        top.addWidget(brand)
        top.addStretch()
        self.status_badge = QLabel("● READY")
        self.status_badge.setObjectName("status")
        # Keep the header minimal; detailed system status stays out of the UI.
        top.addWidget(self.status_badge)
        root.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(10)
        # Center hologram / voice control panel
        hero = QFrame()
        hero.setObjectName("panel")
        hl = QVBoxLayout(hero)
        hl.setContentsMargins(14, 10, 14, 12)
        hl.setSpacing(8)
        self.face = QLabel()
        self.face.setObjectName("face")
        self.face.setAlignment(Qt.AlignCenter)
        face_path = Path(__file__).with_name("jarvis_face.png")
        if face_path.exists():
            pix = QPixmap(str(face_path))
            self.face.setPixmap(pix.scaled(520, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.face.setText("◉\nJ A R V I S")
        hl.addWidget(self.face, 1)
        self.mic_btn = QPushButton("🎙   START LISTENING")
        self.mic_btn.setObjectName("primary")
        self.mic_btn.clicked.connect(self.toggle_listening)
        hl.addWidget(self.mic_btn)
        self.voice_hint = QLabel("Say “Hey Jarvis” to begin")
        self.voice_hint.setObjectName("hint")
        self.voice_hint.setAlignment(Qt.AlignCenter)
        hl.addWidget(self.voice_hint)
        body.addWidget(hero, 5)

        # Conversation and command panel
        chat = QFrame()
        chat.setObjectName("panel")
        chat.setMinimumWidth(310)
        cl = QVBoxLayout(chat)
        cl.setContentsMargins(12, 12, 12, 12)
        ch = QHBoxLayout()
        ch_title = QLabel("CONVERSATION")
        ch_title.setObjectName("sectionTitle")
        ch.addWidget(ch_title)
        ch.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.log.clear())
        ch.addWidget(clear_btn)
        cl.addLayout(ch)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Jarvis replies will appear here…")
        self.log.setObjectName("conversation")
        cl.addWidget(self.log, 1)
        self.command = QLineEdit()
        self.command.setPlaceholderText("Type a command…")
        self.command.returnPressed.connect(self.send_command)
        cl.addWidget(self.command)
        self.send_btn = QPushButton("➤  SEND COMMAND")
        self.send_btn.setObjectName("send")
        self.send_btn.clicked.connect(self.send_command)
        cl.addWidget(self.send_btn)
        settings = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Aria · US", "en-US-AriaNeural")
        self.voice_combo.addItem("Jenny · US", "en-US-JennyNeural")
        self.voice_combo.addItem("Guy · US", "en-US-GuyNeural")
        self.voice_combo.addItem("Sonia · UK", "en-GB-SoniaNeural")
        settings.addWidget(self.voice_combo, 1)
        self.speak_check = QCheckBox("Speak replies")
        self.speak_check.setChecked(True)
        settings.addWidget(self.speak_check)
        cl.addLayout(settings)
        self.listen_btn = QPushButton("Start wake-word listening")
        self.listen_btn.setVisible(False)  # center microphone button is the primary control
        self.listen_btn.clicked.connect(self.toggle_listening)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_agent)
        settings.addWidget(self.stop_btn)
        body.addWidget(chat, 4)
        root.addLayout(body, 1)

        self.setStyleSheet("""
            QWidget#central { background:#020817; color:#d9f4ff; }
            QFrame#panel { background:#041225; border:1px solid #07528a; border-radius:16px; }
            QLabel#brand { color:#8beaff; font-size:18px; font-weight:800; letter-spacing:2px; padding:6px; }
            QLabel#status { background:#052c30; color:#44ffd1; border:1px solid #087f88; border-radius:10px; padding:8px 12px; font-weight:700; }
            QLabel#sectionTitle { color:#76e7ff; font-size:14px; font-weight:800; letter-spacing:2px; }
            QLabel#face { background:#020817; border:1px solid #043b71; border-radius:18px; }
            QLabel#wave { color:#00d9ff; font-size:25px; font-weight:800; padding:4px; }
            QLabel#hint { color:#65b7e8; font-size:11px; }
            QTextEdit, QLineEdit, QComboBox { background:#020b19; border:1px solid #07518a; border-radius:9px; padding:9px; color:#dff8ff; selection-background-color:#007ebc; }
            QTextEdit#conversation { font-family:'Segoe UI'; font-size:12px; }
            QPushButton { background:#072744; border:1px solid #0876ae; border-radius:9px; padding:10px; color:#bdefff; font-weight:700; }
            QPushButton:hover { background:#09416b; }
            QPushButton#primary { background:#003b69; border:2px solid #00c8ff; color:#baf7ff; font-size:15px; padding:14px; }
            QPushButton#primary:hover { background:#00528a; }
            QPushButton#send { background:#04375d; color:#7deaff; }
            QCheckBox { color:#8bdcff; spacing:6px; }
        """)
        self.ensure_workspace_repo()
        self.check_cli()

    def make_divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#07528a; background:#07528a; max-height:1px;")
        return line

    def animate_wave(self):
        patterns = ["▁ ▂ ▄ ▆ ▇ ▅ ▃ ▁ ▂ ▄ ▆ ▇ ▅ ▃ ▁", "▂ ▄ ▆ ▇ ▅ ▃ ▁ ▂ ▄ ▆ ▇ ▅ ▃ ▁ ▂", "▁ ▃ ▅ ▇ ▆ ▄ ▂ ▁ ▃ ▅ ▇ ▆ ▄ ▂ ▁"]
        self.wave.setText(patterns[self.wave_step % len(patterns)])
        self.wave_step += 1

    def ensure_workspace_repo(self):
        git = shutil.which("git")
        if not git:
            self.append("WARNING", "Git was not found. Install Git for Windows or use a workspace already inside a Git repository.")
            return
        if not (Path(self.workspace) / ".git").exists():
            try:
                subprocess.run([git, "init", "-q"], cwd=self.workspace, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, timeout=15)
                pass
            except Exception as exc:
                pass

    def check_cli(self):
        exe = find_interpreter()
        if exe:
            self.status_badge.setText("● READY")
        else:
            self.set_status("Interpreter not found")

    def append(self, who, text):
        # Keep the conversation panel user-facing: never display internal logs/status.
        if who not in ("YOU", "YOU (VOICE)", "JARVIS"):
            return
        safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        color = "#78dfff" if who == "JARVIS" else "#9bb6d4"
        label = "JARVIS" if who == "JARVIS" else "YOU"
        self.log.append(f'<span style="color:{color};font-weight:700">{label}</span><br>{safe}')
        self.log.ensureCursorVisible()

    def set_status(self, text):
        self.status_badge.setText("●  " + text)

    def toggle_listening(self):
        if self.listening:
            self.stop_listener()
        else:
            self.start_listener()

    def start_listener(self):
        if self.worker and self.worker.isRunning():
            return
        self.worker = SpeechWorker()
        self.worker.status.connect(self.on_voice_status)
        self.worker.heard.connect(self.on_heard)
        self.worker.error.connect(lambda msg: self.set_status("Voice error"))
        self.worker.start()
        self.listening = True
        self.listen_btn.setText("■  Stop wake-word listening")
        self.mic_btn.setText("■   STOP LISTENING")
        self.voice_hint.setText("Jarvis is listening for the wake word…")
        self.set_status("Starting voice")

    def stop_listener(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2500)
        self.listening = False
        self.listen_btn.setText("▶  Start wake-word listening")
        self.mic_btn.setText("🎙   START LISTENING")
        self.voice_hint.setText("Say “Hey Jarvis” to talk to your assistant")
        self.set_status("SYSTEM READY")

    def on_voice_status(self, text):
        self.set_status(text[:30])
        self.voice_hint.setText(text)

    def on_heard(self, text):
        self.command.setText(text)
        # send_command() adds the user message once; don't duplicate voice input.
        self.send_command()

    def send_command(self):
        prompt = self.command.text().strip()
        if not prompt:
            return
        approved = QMessageBox.question(
            self, "Approve Jarvis task?",
            "Allow Open Interpreter to process this request?\n\n"
            f"Request: {prompt}\n\n"
            "This assistant can run commands and modify files on your PC. "
            "Approve only requests you trust.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if approved != QMessageBox.StandardButton.Yes:
            self.set_status("Request cancelled")
            return
        if self.agent_worker and self.agent_worker.isRunning():
            QMessageBox.information(self, "Agent busy", "Wait for the current task to finish.")
            return
        self.command.clear()
        self.append("YOU", prompt)
        self._last_prompt = prompt
        self.set_status("Agent working")
        self.send_btn.setEnabled(False)
        self.agent_worker = AgentWorker(prompt, self.workspace)
        self.agent_worker.finished_text.connect(self.on_agent_finished)
        self.agent_worker.start()

    def on_agent_finished(self, result, code):
        self.send_btn.setEnabled(True)
        self.set_status("Ready" if code == 0 else "Needs attention")
        # Show only the concise assistant response, never raw CLI output or diagnostics.
        reply = self.make_spoken_reply(result)
        if not reply:
            reply = "I couldn't extract a response. Please try that command again."
        self.append("JARVIS", reply)
        if self.speak_check.isChecked():
            self.tts_worker = VoiceOutputWorker(reply, self.voice_combo.currentData())
            self.tts_worker.error.connect(lambda msg: self.set_status("Speech playback error"))
            self.tts_worker.start()

    def make_spoken_reply(self, result):
        """Clean terminal formatting while preserving the CLI's final response."""
        import re
        text = str(result or "")
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        return text[:500].strip()

    def stop_agent(self):
        # CLI child process is owned by the worker; termination is intentionally not forceful.
        # Stop button provides a clear warning if the task is still running.
        if self.agent_worker and self.agent_worker.isRunning():
            QMessageBox.warning(
                self, "Stop task",
                "The current CLI task is running in a worker. For safety this version does not force-kill child processes. "
                "Use Ctrl+C in a terminal for a manually launched task, or close the task through Open Interpreter."
            )
        else:
            self.append("SYSTEM", "No active agent task.")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
