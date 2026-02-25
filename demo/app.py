"""numba WASM demo — compile Python to WebAssembly, run in browser.

Usage: PYTHONPATH=. python demo/app.py
"""
import base64
import sys
import textwrap

from flask import Flask, render_template, request, jsonify

# Add numba to path for development
sys.path.insert(0, '.')

app = Flask(__name__)


def compile_python_to_wasm(python_code, func_name):
    """Compile Python code to WASM using numba's wasm_jit."""
    from numba.core.wasm import wasm_jit

    # Create a namespace and execute the code
    namespace = {'wasm_jit': wasm_jit}
    exec(python_code, namespace)

    # Get the compiled function
    func = namespace.get(func_name)
    if func is None:
        raise ValueError(f"Function '{func_name}' not found in code")

    # Trigger compilation with sample args to get WASM
    # We need to infer args from the function signature
    import inspect
    sig = inspect.signature(func._py_func)
    n_args = len(sig.parameters)

    # Compile with sample i64 arguments
    sample_args = tuple(range(1, n_args + 1))
    func(*sample_args)

    # Get WASM bytes and function info
    compiled = list(func._overloads.values())[0]
    wasm_bytes = compiled._library.get_wasm_bytes()
    llvm_func_name = compiled._fndesc.llvm_func_name

    return {
        'wasm': wasm_bytes,
        'llvm_func_name': llvm_func_name,
        'n_args': n_args,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/compile', methods=['POST'])
def api_compile():
    data = request.get_json()
    try:
        result = compile_python_to_wasm(data['code'], data['func_name'])
        return jsonify(
            ok=True,
            wasm=base64.b64encode(result['wasm']).decode(),
            size=len(result['wasm']),
            llvm_func_name=result['llvm_func_name'],
            n_args=result['n_args'],
        )
    except Exception as e:
        import traceback
        return jsonify(ok=False, error=str(e), trace=traceback.format_exc()), 400


if __name__ == '__main__':
    print('numba WASM demo')
    print('http://localhost:8080')
    app.run(port=8080, debug=True)
