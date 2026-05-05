"""
Batch code execution on CPU while GPU continues inference.

Instead of blocking on each solver step's code execution,
collect code snippets and execute them in parallel across CPU cores.
"""

import subprocess
import multiprocessing as mp
from typing import List, Tuple, Optional
from dataclasses import dataclass
from src.env.python_tool import ToolResult


@dataclass
class CodeTask:
    """A single code execution task."""
    code: str
    task_id: int


def _execute_code_task(task: CodeTask, timeout: float = 5.0) -> Tuple[int, ToolResult]:
    """Execute a single code task. Returns (task_id, result)."""
    try:
        # Wrapper script that auto-prints last expression
        wrapper = f"""
import sys
code_str = {repr(task.code)}
try:
    import ast
    tree = ast.parse(code_str)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Expr(value=ast.Call(
            func=ast.Name(id='print', ctx=ast.Load()),
            args=[tree.body[-1].value],
            keywords=[]
        ))
    code_str = ast.unparse(tree)
except:
    pass
exec(code_str)
"""
        process = subprocess.Popen(
            ["python3", "-c", wrapper],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        stdout, stderr = process.communicate(timeout=timeout)
        output = stdout if not stderr else stderr
        is_error = bool(stderr) or process.returncode != 0

        return task.task_id, ToolResult(
            output=output.strip()[:256] if output else "No output",
            is_error=is_error,
            wall_time_ms=0
        )

    except subprocess.TimeoutExpired:
        return task.task_id, ToolResult(
            output="TimeoutExpired",
            is_error=True,
            wall_time_ms=int(timeout * 1000)
        )
    except Exception as e:
        return task.task_id, ToolResult(
            output=f"Error: {str(e)}",
            is_error=True,
            wall_time_ms=0
        )


class CodeBatcher:
    """Batch executor for code snippets across CPU cores."""

    def __init__(self, num_workers: int = 4, timeout: float = 5.0):
        """
        Initialize the batch executor.

        Args:
            num_workers: Number of CPU cores to use for parallel execution
            timeout: Timeout per code execution in seconds
        """
        self.num_workers = num_workers
        self.timeout = timeout
        self.task_queue: List[CodeTask] = []
        self.task_counter = 0

    def queue_code(self, code: str) -> int:
        """
        Queue code for batch execution.

        Args:
            code: Python code string to execute

        Returns:
            task_id: Unique identifier for this task
        """
        task_id = self.task_counter
        self.task_queue.append(CodeTask(code=code, task_id=task_id))
        self.task_counter += 1
        return task_id

    def execute_batch(self) -> dict:
        """
        Execute all queued code in parallel using multiprocessing pool.

        Returns:
            dict: {task_id -> ToolResult} mapping for all executed tasks
        """
        if not self.task_queue:
            return {}

        results_map = {}

        try:
            # Use multiprocessing pool for true parallelism across CPU cores
            with mp.Pool(processes=self.num_workers) as pool:
                # Execute all tasks in parallel
                results = pool.starmap(
                    _execute_code_task,
                    [(task, self.timeout) for task in self.task_queue]
                )

            # Collect results in order by task_id
            for task_id, result in results:
                results_map[task_id] = result

        except Exception as e:
            # Fallback: execute sequentially on error
            print(f"Batch execution failed ({e}), falling back to sequential")
            for task in self.task_queue:
                _, result = _execute_code_task(task, self.timeout)
                results_map[task.task_id] = result

        # Clear queue after execution
        self.task_queue = []
        self.task_counter = 0

        return results_map

    def clear(self):
        """Clear the task queue."""
        self.task_queue = []
        self.task_counter = 0
