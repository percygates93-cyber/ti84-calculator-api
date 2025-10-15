# Bzzzt! This is the final, production-grade code for your calculator brain.
# It now uses SymPy for symbolic power and SciPy for numerical precision with an upgraded engine.
# VERSION 2.0 - NOW WITH NUMERICAL DERIVATIVE POWER!

from flask import Flask, request, jsonify
from flask_cors import CORS
from sympy import sympify, N, SympifyError, diff, integrate, Symbol, Abs, lambdify
from scipy.integrate import quad
from scipy.misc import derivative as numerical_derivative # We need this for nDeriv!
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
    data = request.get_json()
    expression = data.get('expression')
    if not expression:
        return jsonify({"error": "Invalid request. Please provide an 'expression'."}), 400
    try:
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        final_result = round(float(N(sympy_expr, 50)), 3)
        return jsonify({"result": str(final_result)})
    except (SympifyError, TypeError, ValueError) as e:
        return jsonify({"error": "Invalid mathematical expression provided."}), 400

# --- Endpoint 2: Symbolic Differentiation ---
@app.route('/differentiate', methods=['POST'])
def differentiate_expression():
    """Finds the derivative of an expression with respect to a variable."""
    data = request.get_json()
    expression, variable = data.get('expression'), data.get('variable')
    if not all([expression, variable]):
        return jsonify({"error": "Invalid request. Provide 'expression' and 'variable'."}), 400
    try:
        x = Symbol(variable)
        sympy_expr = sympify(expression)
        derivative = diff(sympy_expr, x)
        return jsonify({"result": str(derivative)})
    except (SympifyError, TypeError, ValueError) as e:
        return jsonify({"error": "Could not differentiate the expression."}), 400

# --- Endpoint 3: Indefinite Integration ---
@app.route('/integrate', methods=['POST'])
def integrate_expression():
    """Finds the INDEFINITE integral of an expression using SymPy."""
    data = request.get_json()
    expression, variable = data.get('expression'), data.get('variable')
    if not all([expression, variable]):
        return jsonify({"error": "Invalid request. Provide 'expression' and 'variable'."}), 400
    try:
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        indefinite_integral = integrate(sympy_expr, x)
        return jsonify({"result": str(indefinite_integral), "type": "indefinite"})
    except Exception as e:
        return jsonify({"error": "Could not process the integration expression."}), 400

# --- Endpoint 4: The TI-84 Powerhouse for Definite Integrals (ENGINE UPGRADED) ---
@app.route('/fnInt', methods=['POST'])
def numerical_integrate():
    """A dedicated, high-power endpoint for numerically solving definite integrals."""
    data = request.get_json()
    expression = data.get('expression')
    variable = data.get('variable')
    lower_bound = data.get('lower_bound')
    upper_bound = data.get('upper_bound')

    if not all([expression, variable, lower_bound, upper_bound]):
        return jsonify({"error": "Invalid request. Provide 'expression', 'variable', 'lower_bound', and 'upper_bound'."}), 400
    
    try:
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        numerical_func = lambdify(x, sympy_expr, modules=['numpy'])
        
        # THE ENGINE UPGRADE: Increased limit gives the engine more power for tough problems.
        integral_result, _ = quad(numerical_func, float(lower_bound), float(upper_bound), limit=200)
        
        final_result = round(integral_result, 3)
        return jsonify({"result": str(final_result)})
    except Exception as e:
        print(f"A numerical integration error occurred in fnInt: {e}")
        return jsonify({"error": f"Numerical integration failed. The math is too complex for the engine: {str(e)}"}), 400

# --- [NEW!] Endpoint 5: The TI-84 Powerhouse for Numerical Derivatives ---
@app.route('/nDeriv', methods=['POST'])
def numerical_differentiate():
    """A dedicated, high-power endpoint for numerically evaluating derivatives at a point."""
    data = request.get_json()
    expression = data.get('expression')
    variable = data.get('variable')
    point = data.get('point')

    if not all([expression, variable, point]):
        return jsonify({"error": "Invalid request. Provide 'expression', 'variable', and 'point'."}), 400

    try:
        x = Symbol(variable)
        # We use the same lambdify trick to convert the expression into a fast numerical function
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        numerical_func = lambdify(x, sympy_expr, modules=['numpy'])

        # This is the SciPy magic for numerical derivatives!
        derivative_result = numerical_derivative(numerical_func, float(point), dx=1e-6)
        
        final_result = round(derivative_result, 3)
        return jsonify({"result": str(final_result)})
    except Exception as e:
        print(f"A numerical differentiation error occurred in nDeriv: {e}")
        return jsonify({"error": f"Numerical differentiation failed. The expression might be too complex or undefined: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
