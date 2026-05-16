"""Task lifecycle manager with QUEUED→RUNNING→COMPLETED/FAILED state machine."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

from PySide6.QtCore import QObject, Signal

from ..utils.config import LOGS_DIR, get_timestamp, ensure_dirs


class TaskStatus(Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Task:
    """A single processing task/job."""
    task_id: str
    tool_name: str
    config_json_path: str
    status: TaskStatus = TaskStatus.QUEUED
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log_file: str = ""
    exit_code: Optional[int] = None
    error_message: str = ""
    output_file: str = ""
    progress: int = 0
    progress_msg: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = get_timestamp()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "config_json_path": self.config_json_path,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_file": self.log_file,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "output_file": self.output_file,
        }


class TaskManager(QObject):
    """Manages the lifecycle of processing tasks."""

    # Signals
    task_added = Signal(Task)
    task_status_changed = Signal(Task)
    task_progress = Signal(str, int, str)  # task_id, percent, message
    console_log = Signal(str, str, str)  # task_id, level, message
    all_tasks_done = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._tasks: Dict[str, Task] = {}
        self._counter = 0
        ensure_dirs()

    def create_task(self, tool_name: str, config_json_path: str) -> Task:
        """Create and register a new task."""
        self._counter += 1
        timestamp = get_timestamp()
        task_id = f"{timestamp}_{self._counter:03d}"

        log_file = str(LOGS_DIR / f"{task_id}_{tool_name.replace(' ', '_')}.log")

        task = Task(
            task_id=task_id,
            tool_name=tool_name,
            config_json_path=config_json_path,
            log_file=log_file,
        )

        self._tasks[task_id] = task
        self.task_added.emit(task)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status and emit signal."""
        task = self._tasks.get(task_id)
        if task:
            task.status = status
            if status == TaskStatus.RUNNING:
                task.started_at = get_timestamp()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.finished_at = get_timestamp()
            self.task_status_changed.emit(task)

    def update_progress(self, task_id: str, percent: int, message: str) -> None:
        """Update task progress."""
        task = self._tasks.get(task_id)
        if task:
            task.progress = percent
            task.progress_msg = message
        self.task_progress.emit(task_id, percent, message)

    def append_log(self, task_id: str, level: str, message: str) -> None:
        """Append a log line for a task."""
        task = self._tasks.get(task_id)
        if task and task.log_file:
            try:
                with open(task.log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{level}] {message}\n")
            except Exception:
                pass
        self.console_log.emit(task_id, level, message)

    def set_completed(self, task_id: str, output_file: str = "") -> None:
        """Mark task as completed."""
        task = self._tasks.get(task_id)
        if task:
            task.output_file = output_file
        self.update_status(task_id, TaskStatus.COMPLETED)
        self._check_all_done()

    def set_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed with error message."""
        task = self._tasks.get(task_id)
        if task:
            task.error_message = error
        self.update_status(task_id, TaskStatus.FAILED)
        self.append_log(task_id, "ERROR", error)
        self._check_all_done()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get all task summaries for history display."""
        return [t.to_dict() for t in self._tasks.values()]

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a given status."""
        return [t for t in self._tasks.values() if t.status == status]

    def _check_all_done(self) -> None:
        """Check if all running tasks are complete."""
        running = any(
            t.status == TaskStatus.RUNNING for t in self._tasks.values()
        )
        if not running and self._tasks:
            self.all_tasks_done.emit()

    def clear_completed(self) -> None:
        """Remove completed/cancelled tasks from the list."""
        to_remove = [
            tid for tid, t in self._tasks.items()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
        ]
        for tid in to_remove:
            del self._tasks[tid]
