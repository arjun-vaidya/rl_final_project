import subprocess
import sys
from dataclasses import dataclass

@dataclass
class ToolResult:
    output: str
    is_error: bool
    wall_time_ms: float

def run_python(code: str, timeout: float = 5.0) -> ToolResult:
    """Runs python code in a sandboxed subprocess and returns the result."""
    # Wrapper that uses AST to print the last expression if it exists
    wrapper_script = f"""
import ast
import sys

code = {repr(code)}
try:
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1].value
        # If it's already a print call, don't wrap it
        is_print = (
            isinstance(last_expr, ast.Call) and 
            isinstance(last_expr.func, ast.Name) and 
            last_expr.func.id == 'print'
        )
        if not is_print:
            new_node = ast.Expr(value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[last_expr],
                keywords=[]
            ))
            tree.body[-1] = new_node
    
    ast.fix_missing_locations(tree)
    compiled = compile(tree, filename="<string>", mode="exec")
    exec(compiled, {{}})
except Exception as e:
    print(f"{{type(e).__name__}}: {{str(e)}}", file=sys.stderr)
    sys.exit(1)
"""

    import time
    start_time = time.time()
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", wrapper_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=timeout)
        end_time = time.time()
        
        wall_time_ms = (end_time - start_time) * 1000
        
        if process.returncode != 0:
            return ToolResult(
                output=stderr.strip() or stdout.strip(),
                is_error=True,
                wall_time_ms=wall_time_ms
            )
        
        output = stdout.strip()
        if len(output) > 256:
            output = output[:253] + "..."
            
        return ToolResult(
            output=output,
            is_error=False,
            wall_time_ms=wall_time_ms
        )
        
    except subprocess.TimeoutExpired:
        process.kill()
        return ToolResult(
            output="TimeoutExpired",
            is_error=True,
            wall_time_ms=timeout * 1000
        )
    except Exception as e:
        return ToolResult(
            output=f"Internal Error: {str(e)}",
            is_error=True,
            wall_time_ms=0
        )

def looks_sensible(output: str) -> bool:
    """Checks if the output is non-empty and reasonably sized."""
    if not output or not output.strip():
        return False
    if len(output) > 256: # Truncated by script usually
        return False
    return True
