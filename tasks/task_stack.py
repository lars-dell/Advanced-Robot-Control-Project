"""
Task Stack manager for prioritized multi-task hierarchy ordering and execution.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from tasks.base_task import BaseTask


class TaskStack:
    """
    Manages a ordered list of BaseTask instances sorted strictly by priority.
    """

    def __init__(self, tasks: Optional[List[BaseTask]] = None) -> None:
        """
        Initialize the task stack.

        Args:
            tasks: Optional initial list of BaseTask objects.
        """
        self.tasks: List[BaseTask] = []
        if tasks is not None:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task: BaseTask) -> None:
        """
        Adds a task to the stack and maintains strict priority ordering (0 = highest priority).

        Args:
            task: BaseTask instance to append.
        """
        self.tasks.append(task)
        # Sort by priority index (ascending)
        self.tasks.sort(key=lambda t: t.priority)

    def evaluate_all(self, state: Dict[str, np.ndarray], t: float = 0.0) -> List[Tuple[BaseTask, np.ndarray, np.ndarray]]:
        """
        Evaluates all tasks in the stack for the current state and time.

        Returns:
            List[Tuple[BaseTask, np.ndarray, np.ndarray]]: List of (task, J_i, f_i) ordered by priority.
        """
        return [(task, *task.compute(state, t)) for task in self.tasks]

    def get_task_errors(self, state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Computes error scalar for each task in the stack.

        Returns:
            Dict[str, float]: Mapping of task name -> tracking error norm.
        """
        return {task.name: task.compute_error(state) for task in self.tasks}
