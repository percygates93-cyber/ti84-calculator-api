# Bzzzt! This is the final, production-grade code for your calculator brain.
# It now uses SymPy for symbolic power and SciPy for numerical precision.

from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- 1. ADD THIS IMPORT
from sympy import sympify, N, SympifyError, diff, integrate, Symbol, Abs, lambdify
from scipy.integrate import quad
import numpy as np

app = Flask(__name__)
CORS(app)  # <--- 2. ADD THIS LINE TO ACTIVATE CORS

# --- Health Check Endpoint for UptimeRobot ---
# (The rest of your code stays exactly the same!)
# ...
@app.route('/', methods=['GET'])
def health_check():
    """A simple endpoint to prove the server is online and accessible."""
    return jsonify({"status": "Calculator brain is online and ready!"})

# --- Endpoint 1: Numerical Calculation ---
# This endpoint remains the same, it works perfectly.
@app.route('/calculate', methods=['POST'])
def calculate():
    """Evaluates a mathematical expression to a numerical value."""
    print("Received calculation request...")
    data = request.get_json()
    if not data or 'expression' not in data:
        return jsonify({"error": "Invalid request. Please provide an 'expression'."}), 400

    expression = data['expression']

    try:
        local_dict = {"Abs": Abs}
        sympy_expr = sympify(expression, locals=local_dict)
        high_precision_result = N(sympy_expr, 50)
        final_result = round(float(high_precision_result), 3)
        return jsonify({"result": str(final_result)})
    except (SympifyError, TypeError, ValueError) as e:
        return jsonify({"error": "Invalid mathematical expression provided."}), 400

# --- Endpoint 2: Symbolic Differentiation ---
# This endpoint remains the same, it works perfectly.
@app.route('/differentiate', methods=['POST'])
def differentiate_expression():
    """Finds the derivative of an expression with respect to a variable."""
    print("Received differentiation request...")
    data = request.get_json()
    if not data or 'expression' not in data or 'variable' not in data:
        return jsonify({"error": "Invalid request. Provide 'expression' and 'variable'."}), 400

    expression = data['expression']
    variable = data['variable']

    try:
        x = Symbol(variable)
        sympy_expr = sympify(expression)
        derivative = diff(sympy_expr, x)
        result_str = str(derivative)
        return jsonify({"result": result_str})
    except (SympifyError, TypeError, ValueError) as e:
        return jsonify({"error": "Could not differentiate the expression."}), 400

# --- Endpoint 3: Upgraded Integration with Dual-Core Brain ---
@app.route('/integrate', methods=['POST'])
def integrate_expression():
    """
    Finds the integral of an expression using the best tool for the job:
    - SciPy's quad for high-precision DEFINITE integrals.
    - SymPy for symbolic INDEFINITE integrals.
    """
    print("Received integration request...")
    data = request.get_json()
    if not data or 'expression' not in data or 'variable' not in data:
        return jsonify({"error": "Invalid request. Provide 'expression' and 'variable'."}), 400

    expression = data['expression']
    variable = data['variable']
    lower_bound = data.get('lower_bound')
    upper_bound = data.get('upper_bound')

    try:
        x = Symbol(variable)
        # We still use sympify to safely parse the math expression string
        local_dict = {"Abs": Abs}
        # Use a mapping for numpy equivalents for lambdify
        module_list = ['numpy']
        sympy_expr = sympify(expression, locals=local_dict)

        # --- THE UPGRADE ---
        # Case 1: Definite Integral (Use SciPy's numerical powerhouse)
        if lower_bound is not None and upper_bound is not None:
            print("Processing definite integral with SciPy...")

            # Convert the symbolic expression into a fast, callable numerical function
            # We explicitly tell it to use numpy functions for speed and compatibility
            numerical_func = lambdify(x, sympy_expr, modules=module_list)

            # Use quad for high-precision numerical integration
            # quad returns (result, estimated_error), we only need the result
            integral_result, _ = quad(numerical_func, float(lower_bound), float(upper_bound))

            final_result = round(integral_result, 3)
            return jsonify({"result": str(final_result), "type": "definite"})

        # Case 2: Indefinite Integral (Use SymPy's symbolic genius)
        else:
            print("Processing indefinite integral with SymPy...")
            indefinite_integral = integrate(sympy_expr, x)
            result_str = str(indefinite_integral)
            return jsonify({"result": result_str, "type": "indefinite"})

    except (SympifyError, TypeError, ValueError, SyntaxError) as e:
        print(f"Error during integration: {e}")
        return jsonify({"error": "Could not process the integration expression."}), 400
    except Exception as e:
        # Catch potential errors from the quad function as well
        print(f"A numerical integration error occurred: {e}")
        return jsonify({"error": "Numerical integration failed. The function may be too complex or discontinuous."}), 400


# This part runs the app in the main Replit workspace.
# Our Gunicorn command bypasses this for production deployment.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
