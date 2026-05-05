import sys
import time
import multiprocessing as mp
import ast
from dataclasses import dataclass

@dataclass
class ToolResult:
    output: str
    is_error: bool
    wall_time_ms: float

def _worker_execute(code: str) -> tuple[str, str]:
    import sys
    import io
    import traceback
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = out = io.StringIO()
    sys.stderr = err = io.StringIO()
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body[-1].value
            is_print = (isinstance(last_expr, ast.Call) and isinstance(last_expr.func, ast.Name) and last_expr.func.id == 'print')
            if not is_print:
                new_node = ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()), args=[last_expr], keywords=[]))
                tree.body[-1] = new_node
        ast.fix_missing_locations(tree)
        compiled = compile(tree, filename="<string>", mode="exec")
        exec(compiled, {})
    except Exception as e:
        print(f"{type(e).__name__}: {str(e)}", file=sys.stderr)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    return out.getvalue(), err.getvalue()

_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = mp.get_context()
        _pool = ctx.Pool(processes=64)
    return _pool

def run_python(code: str, timeout: float = 5.0) -> ToolResult:
    """Runs python code in a persistent worker pool and returns the result."""
    start_time = time.time()
    pool = _get_pool()
    try:
        res_future = pool.apply_async(_worker_execute, (code,))
        out, err = res_future.get(timeout=timeout)
        wall_time_ms = (time.time() - start_time) * 1000
        
        err = err.strip()
        out = out.strip()
        if err:
            output = err or out
            is_err = True
        else:
            output = out
            is_err = False
            
        if len(output) > 256:
            output = output[:253] + "..."
            
        return ToolResult(output=output, is_error=is_err, wall_time_ms=wall_time_ms)
        
    except mp.TimeoutError:
        # Recreate the pool on timeout to clear the stuck worker
        global _pool
        _pool.terminate()
        _pool = None
        return ToolResult(output="TimeoutExpired", is_error=True, wall_time_ms=timeout*1000)
    except Exception as e:
        return ToolResult(output=f"Internal Error: {str(e)}", is_error=True, wall_time_ms=0)

def looks_sensible(output: str) -> bool:
    """Checks if the output is non-empty and reasonably sized."""
    if not output or not output.strip():
        return False
    if len(output) > 256: # Truncated by script usually
        return False
    return True
