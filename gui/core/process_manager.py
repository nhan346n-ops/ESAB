"""Subprocess manager for executing pyat backend commands via QProcess.

Manages async, non-blocking execution of: python -m pyat <config.json>
Parses JSON-Lines progress protocol from stdout.
"""
import json
import os
from enum import Enum
from typing import Optional

from PySide6.QtCore import QProcess, QObject, Signal


class ProcessState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    CANCELLED = "cancelled"


class ProcessManager(QObject):
    """Wraps QProcess for running python -m pyat <config.json>."""

    state_changed = Signal(object)  # ProcessState
    progress_changed = Signal(int, str)  # percent, message
    log_received = Signal(str, str)  # level, message
    finished = Signal(int, str)  # exit_code, output_file
    error_occurred = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._state = ProcessState.IDLE
        self._stdout_buffer = ""

    @property
    def state(self) -> ProcessState:
        return self._state

    def run(self, config_json_path: str, working_dir: Optional[str] = None) -> None:
        """Execute: python -m pyat <config_json_path>

        Args:
            config_json_path: Path to the arguments JSON file.
            working_dir: Working directory for subprocess. Defaults to project root.
        """
        if self._state == ProcessState.RUNNING:
            self.log_received.emit("WARNING", "A process is already running.")
            return

        if working_dir is None:
            working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # gui/core/ -> gui/ -> project root
            working_dir = os.path.dirname(os.path.dirname(working_dir))

        self._process = QProcess(self)
        self._process.setWorkingDirectory(working_dir)
        self._process.setProcessChannelMode(QProcess.SeparateChannels)

        # Ensure PYTHONPATH includes src/ for pyat imports
        env = QProcess.systemEnvironment()
        src_path = os.path.join(working_dir, "src")
        if src_path not in os.environ.get("PYTHONPATH", ""):
            env.append(f"PYTHONPATH={src_path}")
        self._process.setEnvironment(env)

        # Connect signals
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

        # Start
        self._state = ProcessState.RUNNING
        self._stdout_buffer = ""
        self.state_changed.emit(self._state)
        self.log_received.emit("INFO", f"Running: python -m pyat {config_json_path}")
        self._process.start("python", ["-m", "pyat", config_json_path])

    def cancel(self) -> None:
        if self._process and self._state == ProcessState.RUNNING:
            self._process.kill()
            self._state = ProcessState.CANCELLED
            self.state_changed.emit(self._state)
            self.log_received.emit("INFO", "Process cancelled by user.")

    def is_running(self) -> bool:
        return self._state == ProcessState.RUNNING

    def _on_stdout(self) -> None:
        """Parse stdout for JSON-Lines progress protocol."""
        if not self._process:
            return
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._stdout_buffer += data

        # Process complete lines
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            # Try JSON-Lines protocol
            try:
                msg = json.loads(line)
                msg_type = msg.get("type", "")
                if msg_type == "progress":
                    pct = msg.get("percent", 0)
                    txt = msg.get("message", "")
                    self.progress_changed.emit(pct, txt)
                elif msg_type == "log":
                    lvl = msg.get("level", "INFO")
                    txt = msg.get("message", "")
                    self.log_received.emit(lvl, txt)
                elif msg_type == "complete":
                    self.log_received.emit("OK", f"Output: {msg.get('output_file', '')}")
                elif msg_type == "error":
                    self.log_received.emit("ERROR", msg.get("message", str(msg)))
                else:
                    self.log_received.emit("INFO", line)
            except json.JSONDecodeError:
                # Plain text line
                self.log_received.emit("INFO", line)

    def _on_stderr(self) -> None:
        if not self._process:
            return
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            line = line.strip()
            if line:
                self.log_received.emit("ERROR", line)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if exit_status == QProcess.CrashExit:
            self._state = ProcessState.ERROR
            self.state_changed.emit(self._state)
            self.error_occurred.emit(f"Process crashed with exit code {exit_code}")
            return

        if exit_code == 0:
            self._state = ProcessState.FINISHED
            self.log_received.emit("OK", f"Process completed successfully (exit 0).")
        else:
            self._state = ProcessState.ERROR
            self.log_received.emit("ERROR", f"Process exited with code {exit_code}")

        self.state_changed.emit(self._state)
        self.finished.emit(exit_code, "")

    def _on_error(self, error: QProcess.ProcessError) -> None:
        err_map = {
            QProcess.FailedToStart: "Failed to start process. Is 'python' on PATH?",
            QProcess.Crashed: "Process crashed.",
            QProcess.Timedout: "Process timed out.",
            QProcess.WriteError: "Write error.",
            QProcess.ReadError: "Read error.",
            QProcess.UnknownError: "Unknown error.",
        }
        msg = err_map.get(error, f"Process error: {error}")
        self._state = ProcessState.ERROR
        self.state_changed.emit(self._state)
        self.error_occurred.emit(msg)
        self.log_received.emit("ERROR", msg)
