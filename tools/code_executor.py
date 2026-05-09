import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional
import signal
from contextlib import contextmanager
import threading
import time

class ExecutionTimeout(Exception):
    """Raised when code execution exceeds time limit."""
    pass

class CodeExecutor:
    """
    Safe sandboxed Python code executor with timeout and security controls.
    
    Features:
    - Execution timeout (default 10 seconds)
    - Dangerous import/operation detection
    - Output/error capture
    - Clear error reporting
    - Context variables isolation
    
    Usage:
        executor = CodeExecutor(timeout=10)
        result = executor.execute("print(1 + 2)")
        print(result['output'])  # "3"
    """
    
    # Dangerous imports that can harm the system
    BLOCKED_IMPORTS = {
        'os', 'sys', 'subprocess', 'socket', 'requests',
        'shutil', 'tempfile', 'importlib', '__import__',
        'eval', 'exec', 'compile', '__builtins__'
    }
    
    # Dangerous operations/keywords
    BLOCKED_KEYWORDS = {
        '__import__', 'eval', 'exec', 'compile', 'globals',
        'locals', 'vars', '__dict__', '__code__', '__class__',
        'open', 'input', 'file'
    }
    
    def __init__(self, timeout: int = 15):
        """
        Initialize code executor.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.execution_globals = {
            '__builtins__': {
                # Allow safe builtins only
                'print': print,
                'len': len,
                'range': range,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'abs': abs,
                'sum': sum,
                'max': max,
                'min': min,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'all': all,
                'any': any,
                'round': round,
                'pow': pow,
                'divmod': divmod,
                'isinstance': isinstance,
                'type': type,
            }
        }
    
    def _check_dangerous_code(self, code: str) -> Optional[str]:
        """
        Check if code contains dangerous patterns.
        
        Returns:
            Error message if dangerous code detected, None otherwise
        """
        code_lines = code.split('\n')
        
        for line_num, line in enumerate(code_lines, 1):
            # Remove comments
            clean_line = line.split('#')[0]
            
            # Check for blocked keywords
            for keyword in self.BLOCKED_KEYWORDS:
                if keyword in clean_line:
                    return f"Line {line_num}: Blocked operation '{keyword}' detected"
            
            # Check for blocked imports
            if clean_line.strip().startswith('import ') or clean_line.strip().startswith('from '):
                for blocked in self.BLOCKED_IMPORTS:
                    if blocked in clean_line:
                        return f"Line {line_num}: Blocked import '{blocked}' detected"
        
        return None
    
    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code safely with timeout.
        
        Args:
            code: Python code to execute
            context: Optional dict of variables to inject into execution context
        
        Returns:
            Dict with keys:
                - success: bool
                - output: captured stdout
                - error: error message (if any)
                - error_type: type of error (if any)
                - execution_time: time in seconds
        """
        start_time = time.time()
        
        # Check for dangerous code
        security_error = self._check_dangerous_code(code)
        if security_error:
            return {
                'success': False,
                'output': '',
                'error': security_error,
                'error_type': 'SecurityError',
                'execution_time': time.time() - start_time
            }
        
        # Prepare execution environment
        exec_globals = dict(self.execution_globals)
        if context:
            exec_globals.update(context)
        
        exec_locals = {}
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            # Redirect stdout/stderr to buffers
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                # Execute with timeout using threading
                result_container = {'result': None, 'error': None}
                
                def run_code():
                    try:
                        exec(code, exec_globals, exec_locals)
                        result_container['result'] = 'ok'
                    except Exception as e:
                        result_container['error'] = e
                
                thread = threading.Thread(target=run_code)
                thread.daemon = True
                thread.start()
                thread.join(timeout=self.timeout)
                
                if thread.is_alive():
                    return {
                        'success': False,
                        'output': output_buffer.getvalue(),
                        'error': f'Code execution exceeded {self.timeout}s timeout',
                        'error_type': 'TimeoutError',
                        'execution_time': time.time() - start_time
                    }
                
                if result_container['error']:
                    raise result_container['error']
            
            # Capture output
            stdout_text = output_buffer.getvalue()
            stderr_text = error_buffer.getvalue()
            full_output = stdout_text + stderr_text if stderr_text else stdout_text
            
            # Merge local variables back to context
            if context:
                context.update({
                    k: v for k, v in exec_locals.items() 
                    if not k.startswith('_')
                })
            
            return {
                'success': True,
                'output': full_output,
                'error': None,
                'error_type': None,
                'execution_time': time.time() - start_time,
                'locals': exec_locals
            }
        
        except Exception as e:
            stderr_text = error_buffer.getvalue()
            error_msg = f"{type(e).__name__}: {str(e)}"
            if stderr_text:
                error_msg = stderr_text + "\n" + error_msg
            
            return {
                'success': False,
                'output': output_buffer.getvalue(),
                'error': error_msg,
                'error_type': type(e).__name__,
                'execution_time': time.time() - start_time,
                'traceback': traceback.format_exc()
            }


# Convenience function for single use
def execute_code(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute Python code safely.
    
    Args:
        code: Python code string
        timeout: Execution timeout in seconds
    
    Returns:
        Execution result dict
        
    Example:
        result = execute_code("print('hello'); x = [1,2,3]; print(sum(x))")
        if result['success']:
            print(result['output'])
        else:
            print(result['error'])
    """
    executor = CodeExecutor(timeout=timeout)
    return executor.execute(code)