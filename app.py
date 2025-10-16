# Bzzzt! This is the final, production-grade code for your calculator brain.
# It now uses SymPy for symbolic power and SciPy for numerical precision with an upgraded engine.
# VERSION 3.1 - Legendary Edition with numdifftools!

from flask import Flask, request, jsonify
from flask_cors import CORS
from sympy import sympify, N, SympifyError, diff, integrate, Symbol, Abs, lambdify
from scipy.integrate import quad
import numpy as np
import numdifftools as nd # IMPORTING THE NEW HIGH-PRECISION ENGINE!

app = Flask(__name__)
CORS(app)

# --- Health Check Endpoint ---
@app.route('/', methods=['GET'])
def health_check():
    """A simple endpoint to prove the server is online and accessible."""
    return jsonify({"status": "Calculator brain is online and ready!"})

# --- Endpoint for ALL Numerical Evaluations ---
@app.route('/evaluate', methods=['POST'])
def evaluate_expression_endpoint():
    """Evaluates any mathematical expression to a single numerical value."""
    data = request.get_json()
    expression = data.get('expression')
    if not expression:
        return jsonify({"error": "Invalid request. Please provide an 'expression'."}), 400
    try:
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        final_result = round(float(N(sympy_expr, 50)), 3)
        return jsonify({"result": str(final_result)})
    except (SympifyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid mathematical expression provided: {str(e)}"}), 400

# --- Endpoint for Symbolic Differentiation ---
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

# --- Endpoint for Indefinite Integration ---
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

# --- Endpoint for Definite Integrals ---
@app.route('/fnInt', methods=['POST'])
def numerical_integrate():
    """A dedicated, high-power endpoint for numerically solving definite integrals."""
    data = request.get_json()
    expression, variable = data.get('expression'), data.get('variable')
    lower_bound_str, upper_bound_str = data.get('lower_bound'), data.get('upper_bound')

    if not all([expression, variable, lower_bound_str, upper_bound_str]):
        return jsonify({"error": "Invalid request. Provide 'expression', 'variable', 'lower_bound', and 'upper_bound'."}), 400
    
    try:
        lower_bound = float(N(sympify(lower_bound_str)))
        upper_bound = float(N(sympify(upper_bound_str)))
        
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        numerical_func = lambdify(x, sympy_expr, modules=['numpy'])
        
        integral_result, _ = quad(numerical_func, lower_bound, upper_bound, limit=200)
        
        final_result = round(integral_result, 3)
        return jsonify({"result": str(final_result)})
    except Exception as e:
        print(f"A numerical integration error occurred in fnInt: {e}")
        return jsonify({"error": f"Numerical integration failed. The math is too complex for the engine: {str(e)}"}), 400

# --- Endpoint for Numerical Derivatives (ENGINE UPGRADED!) ---
@app.route('/nDeriv', methods=['POST'])
def numerical_differentiate():
    """A dedicated, high-power endpoint for numerically evaluating derivatives at a point."""
    data = request.get_json()
    expression, variable, point = data.get('expression'), data.get('variable'), data.get('point')

    if not all([expression, variable, point]):
        return jsonify({"error": "Invalid request. Provide 'expression', 'variable', and 'point'."}), 400

    try:
        x_val = float(point)
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        numerical_func = lambdify(x, sympy_expr, modules=['numpy'])

        # THE UPGRADE: Using the professional-grade numdifftools library
        derivative_result = nd.Derivative(numerical_func)(x_val)
        
        final_result = round(derivative_result, 3)
        return jsonify({"result": str(final_result)})
    except Exception as e:
        print(f"A numerical differentiation error occurred in nDeriv: {e}")
        return jsonify({"error": f"Numerical differentiation failed. The expression might be too complex or undefined: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
