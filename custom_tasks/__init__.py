from .generation import generate
from .longmemeval import LongMemEval

TASKS = {task.name: task for task in (LongMemEval,)}

__all__ = ["LongMemEval", "TASKS", "generate"]
