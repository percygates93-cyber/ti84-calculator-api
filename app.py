# Bzzzt! This is the final, production-grade code for your calculator brain.
# It now uses SymPy for symbolic power and SciPy for numerical precision.

from flask import Flask, request, jsonify
from flask_cors import CORS
from sympy import sympify, N, SympifyError, diff, integrate, Symbol, Abs, lambdify
from scipy.integrate import quad
import numpy as np

app = Flask(__name__)
CORS(app)

# --- Health Check Endpoint for UptimeRobot ---
@app.route('/', methods=['GET'])
def health_check():
    """A simple endpoint to prove the server is online and accessible."""
    return jsonify({"status": "Calculator brain is online and ready!"})

# --- Endpoint 1: Numerical Calculation ---
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
    Finds the INDEFINITE integral of an expression using SymPy.
    """
    print("Received integration request...")
    data = request.get_json()
    if not data or 'expression' not in data or 'variable' not in data:
        return jsonify({"error": "Invalid request. Provide 'expression' and 'variable'."}), 400

    expression = data['expression']
    variable = data['variable']

    try:
        x = Symbol(variable)
        local_dict = {"Abs": Abs}
        sympy_expr = sympify(expression, locals=local_dict)

        print("Processing indefinite integral with SymPy...")
        indefinite_integral = integrate(sympy_expr, x)
        result_str = str(indefinite_integral)
        return jsonify({"result": result_str, "type": "indefinite"})

    except (SympifyError, TypeError, ValueError, SyntaxError) as e:
        print(f"Error during integration: {e}")
        return jsonify({"error": "Could not process the integration expression."}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred during integration."}), 400

# --- Endpoint 4: The TI-84 Powerhouse for Definite Integrals ---
@app.route('/fnInt', methods=['POST'])
def numerical_integrate():
    """
    A dedicated, high-power endpoint for numerically solving definite integrals.
    This is the "fnInt" button. It ONLY uses SciPy's quad.
    """
    print("Received fnInt request...")
    data = request.get_json()
    if not data or 'expression' not in data or 'variable' not in data or 'lower_bound' not in data or 'upper_bound' not in data:
        return jsonify({"error": "Invalid request. Provide 'expression', 'variable', 'lower_bound', and 'upper_bound'."}), 400

    expression = data['expression']
    variable = data['variable']
    lower_bound = data['lower_bound']
    upper_bound = data['upper_bound']
    
    try:
        x = Symbol(variable)
        module_list = ['numpy']
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        
        numerical_func = lambdify(x, sympy_expr, modules=module_list)
        
        integral_result, _ = quad(numerical_func, float(lower_bound), float(upper_bound))
        
        final_result = round(integral_result, 3)
        return jsonify({"result": str(final_result)})

    except Exception as e:
        print(f"A numerical integration error occurred in fnInt: {e}")
        return jsonify({"error": "Numerical integration failed. The function may be too complex or discontinuous."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
