# Bzzzt! This is the final, production-grade code for your calculator brain.
# It uses SymPy for symbolic power and SciPy + NumPy + numdifftools for numerical precision.
# VERSION 4.0 – AP Calc AB/BC Complete

from flask import Flask, request, jsonify
from flask_cors import CORS
from sympy import sympify, N, SympifyError, diff, integrate, Symbol, Abs, lambdify
from scipy.integrate import quad
from scipy.optimize import root_scalar
import numpy as np
import numdifftools as nd  # High-precision numerical derivatives

app = Flask(__name__)
CORS(app)

# ---------------------- Health Check ----------------------
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Calculator brain is online and ready!"})

# ---------------------- Core Helpers ----------------------
def _as_func(expr_str, var_name):
    x = Symbol(var_name)
    expr = sympify(expr_str, locals={"Abs": Abs})
    return x, expr, lambdify(x, expr, modules=['numpy'])

def _round3(x):
    return float(f"{float(x):.3f}")

def _unique_sorted(vals):
    # Deduplicate at 3 d.p. and sort
    return sorted({_round3(v) for v in vals})

def _safe_float(x):
    return float(N(sympify(str(x))))

def _scan_brackets(f_vec, a, b, steps):
    """
    Scan [a,b] with 'steps' subintervals to find brackets where f changes sign.
    f_vec: vectorized function that accepts numpy array and returns array.
    """
    xs = np.linspace(a, b, steps + 1)
    vals = f_vec(xs)
    br = []
    for i in range(len(xs) - 1):
        y1, y2 = vals[i], vals[i+1]
        if np.isnan(y1) or np.isnan(y2) or np.isinf(y1) or np.isinf(y2):
            continue
        if y1 == 0.0:
            br.append((xs[i], xs[i]))  # exact grid root
        elif y1 * y2 < 0:
            br.append((xs[i], xs[i+1]))
    return br

def _bisect_root(f_scalar, L, R):
    """Refine a root in [L,R] via Brent; fallback to secant."""
    try:
        if L == R:
            return L
        sol = root_scalar(f_scalar, bracket=[L, R], method='brentq', maxiter=100)
        if sol.converged:
            return sol.root
    except Exception:
        pass
    try:
        sol = root_scalar(f_scalar, x0=L, x1=R, method='secant', maxiter=100)
        if sol.converged:
            return sol.root
    except Exception:
        pass
    return None

# ---------------------- Evaluate (numeric) ----------------------
@app.route('/evaluate', methods=['POST'])
def evaluate_expression_endpoint():
    """Evaluates any mathematical expression to a single numerical value."""
    data = request.get_json()
    expression = data.get('expression')
    if not expression:
        return jsonify({"error": "Invalid request. Please provide an 'expression'."}), 400
    try:
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        final_result = _round3(N(sympy_expr, 50))
        return jsonify({"result": str(final_result)})
    except (SympifyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid mathematical expression provided: {str(e)}"}), 400

# ---------------------- Differentiate (symbolic) ----------------------
@app.route('/differentiate', methods=['POST'])
def differentiate_expression():
    """Finds the derivative of an expression with respect to a variable (symbolic)."""
    data = request.get_json()
    expression, variable = data.get('expression'), data.get('variable')
    if not all([expression, variable]):
        return jsonify({"error": "Invalid request. Provide 'expression' and 'variable'."}), 400
    try:
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        derivative = diff(sympy_expr, x)
        return jsonify({"result": str(derivative)})
    except (SympifyError, TypeError, ValueError):
        return jsonify({"error": "Could not differentiate the expression."}), 400

# ---------------------- Integrate (indefinite) ----------------------
@app.route('/integrate', methods=['POST'])
def integrate_expression():
    """Finds the INDEFINITE integral of an expression using SymPy (symbolic)."""
    data = request.get_json()
    expression, variable = data.get('expression'), data.get('variable')
    if not all([expression, variable]):
        return jsonify({"error": "Invalid request. Provide 'expression' and 'variable'."}), 400
    try:
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        indefinite_integral = integrate(sympy_expr, x)
        return jsonify({"result": str(indefinite_integral), "type": "indefinite"})
    except Exception:
        return jsonify({"error": "Could not process the integration expression."}), 400

# ---------------------- Definite Integral (numeric) ----------------------
@app.route('/fnInt', methods=['POST'])
def numerical_integrate():
    """Numerically solve definite integrals via SciPy quad."""
    data = request.get_json()
    expression, variable = data.get('expression'), data.get('variable')
    lower_bound_str, upper_bound_str = data.get('lower_bound'), data.get('upper_bound')
    if not all([expression, variable, lower_bound_str, upper_bound_str]):
        return jsonify({"error": "Provide 'expression', 'variable', 'lower_bound', 'upper_bound'."}), 400
    try:
        lower_bound = _safe_float(lower_bound_str)
        upper_bound = _safe_float(upper_bound_str)
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        f = lambdify(x, sympy_expr, modules=['numpy'])
        integral_result, _ = quad(f, lower_bound, upper_bound, limit=200)
        return jsonify({"result": str(_round3(integral_result))})
    except Exception as e:
        print(f"A numerical integration error occurred in fnInt: {e}")
        return jsonify({"error": f"Numerical integration failed. {str(e)}"}), 400

# ---------------------- Numeric Derivative at a point ----------------------
@app.route('/nDeriv', methods=['POST'])
def numerical_differentiate():
    """Numerically evaluate the derivative at a point using numdifftools."""
    data = request.get_json()
    expression, variable, point = data.get('expression'), data.get('variable'), data.get('point')
    if not all([expression, variable, point]):
        return jsonify({"error": "Provide 'expression', 'variable', and 'point'."}), 400
    try:
        x_val = _safe_float(point)
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        f = lambdify(x, sympy_expr, modules=['numpy'])
        derivative_result = nd.Derivative(lambda t: f(t))(x_val)
        return jsonify({"result": str(_round3(derivative_result))})
    except Exception as e:
        print(f"A numerical differentiation error occurred in nDeriv: {e}")
        return jsonify({"error": f"Numerical differentiation failed. {str(e)}"}), 400

# ---------------------- Zeros / Critical / Extrema / Inflection ----------------------
@app.route('/zeros', methods=['POST'])
def find_zeros():
    """Roots of f(x) = 0 on [a,b]."""
    d = request.get_json()
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(expr, var)
        F_vec = lambda ts: f(ts)  # vectorized via numpy
        brackets = _scan_brackets(F_vec, a, b, steps)
        roots = []
        for L, R in brackets:
            r = _bisect_root(lambda t: float(f(t)), L, R)
            if r is not None and a - 1e-12 <= r <= b + 1e-12:
                roots.append(r)
        return jsonify({"zeros": _unique_sorted(roots)})
    except Exception as e:
        return jsonify({"error": f"Zero-finding failed: {e}"}), 400

@app.route('/critical', methods=['POST'])
def critical_points():
    """Roots of f'(x) on [a,b] (critical points)."""
    d = request.get_json()
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(expr, var)

        def fp_scalar(t):
            return float(nd.Derivative(lambda s: f(s))(t))
        def fp_vec(ts):
            return np.array([fp_scalar(t) for t in np.atleast_1d(ts)])

        brackets = _scan_brackets(fp_vec, a, b, steps)
        cps = []
        for L, R in brackets:
            r = _bisect_root(fp_scalar, L, R)
            if r is not None and a - 1e-12 <= r <= b + 1e-12:
                cps.append(r)
        return jsonify({"critical_points": _unique_sorted(cps)})
    except Exception as e:
        return jsonify({"error": f"Critical-point search failed: {e}"}), 400

@app.route('/extrema', methods=['POST'])
def extrema():
    """Classify local minima/maxima from critical points and return f(x) values; include endpoints."""
    d = request.get_json()
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(expr, var)

        def fp_scalar(t):
            return float(nd.Derivative(lambda s: f(s))(t))
        def fp_vec(ts):
            return np.array([fp_scalar(t) for t in np.atleast_1d(ts)])

        cps_raw = []
        for L, R in _scan_brackets(fp_vec, a, b, steps):
            r = _bisect_root(fp_scalar, L, R)
            if r is not None: cps_raw.append(r)
        cps = _unique_sorted(cps_raw)

        results = []
        for xc in cps:
            eps = 1e-3
            left = fp_scalar(xc - eps)
            right = fp_scalar(xc + eps)
            if left > 0 and right < 0:
                kind = "local_max"
            elif left < 0 and right > 0:
                kind = "local_min"
            else:
                kind = "neither"
            results.append({"x": _round3(xc), "type": kind, "f": _round3(f(xc))})

        return jsonify({"extrema": results,
                        "endpoints": [{"x": _round3(a), "f": _round3(f(a))},
                                      {"x": _round3(b), "f": _round3(f(b))}]})
    except Exception as e:
        return jsonify({"error": f"Extrema classification failed: {e}"}), 400

@app.route('/inflection', methods=['POST'])
def inflection_points():
    """Inflection points where f'' changes sign on [a,b]."""
    d = request.get_json()
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(expr, var)

        def fpp_scalar(t):
            return float(nd.Derivative(lambda s: f(s), n=2)(t))
        def fpp_vec(ts):
            return np.array([fpp_scalar(t) for t in np.atleast_1d(ts)])

        candidates = []
        for L, R in _scan_brackets(fpp_vec, a, b, steps):
            r = _bisect_root(fpp_scalar, L, R)
            if r is not None:
                candidates.append(r)

        points = []
        for xc in _unique_sorted(candidates):
            eps = 1e-3
            left = fpp_scalar(xc - eps)
            right = fpp_scalar(xc + eps)
            if left * right < 0:
                points.append(_round3(xc))
        return jsonify({"inflection_points": points})
    except Exception as e:
        return jsonify({"error": f"Inflection search failed: {e}"}), 400

# ---------------------- Cartesian Intersections ----------------------
@app.route('/intersections', methods=['POST'])
def intersections():
    """
    Cartesian intersections of f and g on [a,b]: solve f(x)=g(x).
    Returns x-values and corresponding y-values.
    """
    d = request.get_json()
    ef, eg, var, a, b = d.get('expression_f'), d.get('expression_g'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 600))
    if not all([ef, eg, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression_f','expression_g','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(ef, var)
        _, _, g = _as_func(eg, var)

        F_vec = lambda ts: f(ts) - g(ts)
        brackets = _scan_brackets(F_vec, a, b, steps)

        xs = []
        for L, R in brackets:
            r = _bisect_root(lambda t: float(f(t) - g(t)), L, R)
            if r is not None and a - 1e-12 <= r <= b + 1e-12:
                xs.append(r)

        xs = _unique_sorted(xs)
        pts = [{"x": _round3(t), "y": _round3(f(t))} for t in xs]
        return jsonify({"x": xs, "points": pts})
    except Exception as e:
        return jsonify({"error": f"Intersections failed: {e}"}), 400

# ---------------------- Areas: Between Curves ----------------------
@app.route('/areaBetween', methods=['POST'])
def area_between():
    """
    Area between f and g on [a,b]: ∫_a^b |f-g| dx, partitioned at crossings of f-g.
    """
    d = request.get_json()
    ef, eg, var, a, b = d.get('expression_f'), d.get('expression_g'), d.get('variable'), d.get('a'), d.get('b')
    if not all([ef, eg, var]) or a is None or b is None:
        return jsonify({"error":"Provide 'expression_f','expression_g','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(ef, var)
        _, _, g = _as_func(eg, var)
        H = lambda t: f(t) - g(t)
        F_vec = lambda ts: H(ts)
        brackets = _scan_brackets(F_vec, a, b, 600)
        # Build partition points (a, any bracket endpoints, b)
        pts = [a] + [pt for pair in brackets for pt in pair if a < pt < b] + [b]
        pts = sorted(list({round(p, 10) for p in pts}))
        total = 0.0
        for L, R in zip(pts[:-1], pts[1:]):
            integrand = lambda t: np.abs(H(t))
            val, _ = quad(integrand, L, R, limit=200)
            total += val
        return jsonify({"result": str(_round3(total))})
    except Exception as e:
        return jsonify({"error": f"Area between curves failed: {e}"}), 400

# ---------------------- Tabular Calculus ----------------------
@app.route('/tabularIntegral', methods=['POST'])
def tabular_integral():
    """Trapezoidal approximation from (x[], y[])."""
    d = request.get_json()
    xs, ys = d.get('x'), d.get('y')
    if xs is None or ys is None or len(xs) != len(ys) or len(xs) < 2:
        return jsonify({"error":"Provide arrays 'x' and 'y' of equal length ≥ 2."}), 400
    try:
        xs = np.array([_safe_float(v) for v in xs])
        ys = np.array([_safe_float(v) for v in ys])
        area = 0.0
        for i in range(len(xs)-1):
            h = xs[i+1] - xs[i]
            area += 0.5 * h * (ys[i] + ys[i+1])
        return jsonify({"result": str(_round3(area))})
    except Exception as e:
        return jsonify({"error": f"Tabular integral failed: {e}"}), 400

@app.route('/tabularDerivative', methods=['POST'])
def tabular_derivative():
    """Estimate f'(x0) from table via nearest central/one-sided difference."""
    d = request.get_json()
    xs, ys, x0 = d.get('x'), d.get('y'), d.get('x0')
    if xs is None or ys is None or x0 is None:
        return jsonify({"error":"Provide 'x','y','x0'."}), 400
    try:
        xs = np.array([_safe_float(v) for v in xs])
        ys = np.array([_safe_float(v) for v in ys])
        x0 = _safe_float(x0)
        i = int(np.argmin(np.abs(xs - x0)))
        if 0 < i < len(xs)-1:
            der = (ys[i+1] - ys[i-1]) / (xs[i+1] - xs[i-1])
        elif i == 0:
            der = (ys[1] - ys[0]) / (xs[1] - xs[0])
        else:
            der = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return jsonify({"result": str(_round3(der))})
    except Exception as e:
        return jsonify({"error": f"Tabular derivative failed: {e}"}), 400

# ---------------------- Euler’s Method ----------------------
@app.route('/euler', methods=['POST'])
def euler_method_endpoint():
    """
    Euler's method for dy/dx = f(x,y).
    JSON: { "fxy": "x - y", "x0":0, "y0":1, "h":0.2, "n":5 }
    """
    d = request.get_json()
    fxy, x0, y0, h, n = d.get('fxy'), d.get('x0'), d.get('y0'), d.get('h'), d.get('n')
    if not all([fxy is not None, x0 is not None, y0 is not None, h is not None, n is not None]):
        return jsonify({"error":"Provide 'fxy','x0','y0','h','n'."}), 400
    try:
        x_sym, y_sym = Symbol('x'), Symbol('y')
        f_expr = sympify(fxy, locals={"Abs": Abs})
        f = lambdify((x_sym, y_sym), f_expr, modules=['numpy'])
        x = _safe_float(x0); y = _safe_float(y0); h = _safe_float(h); n = int(n)
        pts = [{"x": _round3(x), "y": _round3(y)}]
        for _ in range(n):
            y = y + h * float(f(x, y))
            x = x + h
            pts.append({"x": _round3(x), "y": _round3(y)})
        return jsonify({"points": pts, "y_n": _round3(y)})
    except Exception as e:
        return jsonify({"error": f"Euler failed: {e}"}), 400

# ---------------------- Motion: position & total distance ----------------------
@app.route('/positionFromVelocity', methods=['POST'])
def position_from_velocity():
    """s(t) = s0 + ∫_a^b v(t) dt"""
    d = request.get_json()
    v, var, a, b, s0 = d.get('v'), d.get('variable'), d.get('a'), d.get('b'), d.get('s0', 0)
    if not all([v, var]) or a is None or b is None:
        return jsonify({"error":"Provide 'v','variable','a','b' (and optional 's0')."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b); s0 = _safe_float(s0)
        x, _, vf = _as_func(v, var)
        val, _ = quad(vf, a, b, limit=200)
        return jsonify({"result": str(_round3(s0 + val))})
    except Exception as e:
        return jsonify({"error": f"positionFromVelocity failed: {e}"}), 400

@app.route('/totalDistance', methods=['POST'])
def total_distance():
    """∫_a^b |v(t)| dt"""
    d = request.get_json()
    v, var, a, b = d.get('v'), d.get('variable'), d.get('a'), d.get('b')
    if not all([v, var]) or a is None or b is None:
        return jsonify({"error":"Provide 'v','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, vf = _as_func(v, var)
        abs_v = lambda t: np.abs(vf(t))
        val, _ = quad(abs_v, a, b, limit=200)
        return jsonify({"result": str(_round3(val))})
    except Exception as e:
        return jsonify({"error": f"totalDistance failed: {e}"}), 400

# ---------------------- Parametric & Polar (BC) ----------------------
@app.route('/parametricSlope', methods=['POST'])
def parametric_slope():
    """dy/dx = (dy/dt)/(dx/dt) at a given t."""
    d = request.get_json()
    xt, yt, t0 = d.get('x_t'), d.get('y_t'), d.get('t')
    if not all([xt, yt, t0 is not None]):
        return jsonify({"error":"Provide 'x_t','y_t','t'."}), 400
    try:
        T = Symbol('t')
        t0 = _safe_float(t0)
        xexpr, yexpr = sympify(xt, locals={"Abs": Abs}), sympify(yt, locals={"Abs": Abs})
        dxdt = lambdify(T, diff(xexpr, T), modules=['numpy'])
        dydt = lambdify(T, diff(yexpr, T), modules=['numpy'])
        slope = float(dydt(t0)) / float(dxdt(t0))
        return jsonify({"result": str(_round3(slope))})
    except Exception as e:
        return jsonify({"error": f"parametricSlope failed: {e}"}), 400

@app.route('/parametricArcLength', methods=['POST'])
def parametric_arc_length():
    """∫_a^b sqrt((dx/dt)^2 + (dy/dt)^2) dt"""
    d = request.get_json()
    xt, yt, a, b = d.get('x_t'), d.get('y_t'), d.get('a'), d.get('b')
    if not all([xt, yt]) or a is None or b is None:
        return jsonify({"error":"Provide 'x_t','y_t','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        T = Symbol('t')
        xexpr, yexpr = sympify(xt, locals={"Abs": Abs}), sympify(yt, locals={"Abs": Abs})
        dxt = lambdify(T, diff(xexpr, T), modules=['numpy'])
        dyt = lambdify(T, diff(yexpr, T), modules=['numpy'])
        integrand = lambda t: np.sqrt(dxt(t)**2 + dyt(t)**2)
        val, _ = quad(integrand, a, b, limit=200)
        return jsonify({"result": str(_round3(val))})
    except Exception as e:
        return jsonify({"error": f"parametricArcLength failed: {e}"}), 400

@app.route('/polarArea', methods=['POST'])
def polar_area():
    """(1/2) ∫ r(θ)^2 dθ on [a,b]."""
    d = request.get_json()
    r, a, b = d.get('r'), d.get('a'), d.get('b')
    if r is None or a is None or b is None:
        return jsonify({"error":"Provide 'r','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        th = Symbol('theta')
        rexpr = sympify(r, locals={"Abs": Abs})
        rf = lambdify(th, rexpr, modules=['numpy'])
        integrand = lambda t: 0.5 * (rf(t)**2)
        val, _ = quad(integrand, a, b, limit=200)
        return jsonify({"result": str(_round3(val))})
    except Exception as e:
        return jsonify({"error": f"polarArea failed: {e}"}), 400

@app.route('/polarIntersections', methods=['POST'])
def polar_intersections():
    """Solve r1(θ) = r2(θ) on [a,b]; returns θ values."""
    d = request.get_json()
    r1, r2, a, b = d.get('r1'), d.get('r2'), d.get('a'), d.get('b')
    if not all([r1, r2]) or a is None or b is None:
        return jsonify({"error":"Provide 'r1','r2','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        th = Symbol('theta')
        r1f = lambdify(th, sympify(r1, locals={"Abs": Abs}), modules=['numpy'])
        r2f = lambdify(th, sympify(r2, locals={"Abs": Abs}), modules=['numpy'])
        F = lambda t: r1f(t) - r2f(t)
        F_vec = lambda ts: F(ts)
        brackets = _scan_brackets(F_vec, a, b, 800)
        roots = []
        for L, R in brackets:
            r = _bisect_root(lambda t: float(F(t)), L, R)
            if r is not None:
                roots.append(r)
        return jsonify({"thetas": _unique_sorted(roots)})
    except Exception as e:
        return jsonify({"error": f"polarIntersections failed: {e}"}), 400

# ---------------------- Series & Taylor (BC) ----------------------
@app.route('/partialSum', methods=['POST'])
def partial_sum():
    """Compute S_n = sum_{k=1..n} a(k)."""
    d = request.get_json()
    a_n, n, idx = d.get('a_n'), d.get('n'), d.get('index', 'n')
    if a_n is None or n is None:
        return jsonify({"error":"Provide 'a_n' and 'n'."}), 400
    try:
        n = int(n)
        k = Symbol(idx)
        term = sympify(a_n, locals={"Abs": Abs})
        f = lambdify(k, term, modules=['numpy'])
        s = 0.0
        for i in range(1, n + 1):
            s += float(f(i))
        return jsonify({"result": str(_round3(s))})
    except Exception as e:
        return jsonify({"error": f"partialSum failed: {e}"}), 400

@app.route('/altSeriesError', methods=['POST'])
def alternating_series_error():
    """Alternating Series Error Bound ≈ |a_{n+1}|."""
    d = request.get_json()
    a_n, n, idx = d.get('a_n'), d.get('n'), d.get('index', 'n')
    if a_n is None or n is None:
        return jsonify({"error":"Provide 'a_n' and 'n'."}), 400
    try:
        n = int(n)
        k = Symbol(idx)
        term = sympify(a_n, locals={"Abs": Abs})
        f = lambdify(k, term, modules=['numpy'])
        err = abs(float(f(n + 1)))
        return jsonify({"error_bound": str(_round3(err))})
    except Exception as e:
        return jsonify({"error": f"altSeriesError failed: {e}"}), 400

@app.route('/taylorPoly', methods=['POST'])
def taylor_poly():
    """Return Taylor polynomial of order n around x0 (symbolic)."""
    d = request.get_json()
    fstr, var, x0, n = d.get('f'), d.get('variable'), d.get('x0'), d.get('n')
    if not all([fstr, var, x0 is not None, n is not None]):
        return jsonify({"error":"Provide 'f','variable','x0','n'."}), 400
    try:
        x0 = _safe_float(x0); n = int(n)
        x = Symbol(var)
        f = sympify(fstr, locals={"Abs": Abs})
        series = f.series(x, x0, n + 1).removeO()
        return jsonify({"polynomial": str(series)})
    except Exception as e:
        return jsonify({"error": f"taylorPoly failed: {e}"}), 400

@app.route('/taylorApprox', methods=['POST'])
def taylor_approx():
    """Evaluate Taylor polynomial of order n for f at x_eval."""
    d = request.get_json()
    fstr, var, x0, n, xe = d.get('f'), d.get('variable'), d.get('x0'), d.get('n'), d.get('x_eval')
    if not all([fstr, var, x0 is not None, n is not None, xe is not None]):
        return jsonify({"error":"Provide 'f','variable','x0','n','x_eval'."}), 400
    try:
        x0 = _safe_float(x0); n = int(n); xe = _safe_float(xe)
        x = Symbol(var)
        f = sympify(fstr, locals={"Abs": Abs})
        poly = f.series(x, x0, n + 1).removeO()
        pfunc = lambdify(x, poly, modules=['numpy'])
        return jsonify({"result": str(_round3(pfunc(xe)))})
    except Exception as e:
        return jsonify({"error": f"taylorApprox failed: {e}"}), 400

# ---------------------- Logistic / Exponential + solver ----------------------
@app.route('/logistic', methods=['POST'])
def logistic_value():
    """P(t) for dP/dt = kP(1 - P/L), given P0 at t=0."""
    d = request.get_json()
    P0, k, L, t = d.get('P0'), d.get('k'), d.get('L'), d.get('t')
    if not all([P0, k, L, t]):
        return jsonify({"error":"Provide 'P0','k','L','t'."}), 400
    try:
        P0 = _safe_float(P0); k = _safe_float(k); L = _safe_float(L); t = _safe_float(t)
        val = L / (1 + ((L - P0) / P0) * np.exp(-k * t))
        return jsonify({"result": str(_round3(val))})
    except Exception as e:
        return jsonify({"error": f"logistic failed: {e}"}), 400

@app.route('/expGrowth', methods=['POST'])
def exponential_value():
    """P(t) = P0 * e^{k t}."""
    d = request.get_json()
    P0, k, t = d.get('P0'), d.get('k'), d.get('t')
    if not all([P0, k, t]):
        return jsonify({"error":"Provide 'P0','k','t'."}), 400
    try:
        P0 = _safe_float(P0); k = _safe_float(k); t = _safe_float(t)
        return jsonify({"result": str(_round3(P0 * np.exp(k * t)))})
    except Exception as e:
        return jsonify({"error": f"expGrowth failed: {e}"}), 400

@app.route('/solveForT', methods=['POST'])
def solve_for_t():
    """Solve f(t) = target on [a,b]; return t."""
    d = request.get_json()
    fstr, a, b, target = d.get('f_t'), d.get('a'), d.get('b'), d.get('target')
    if not all([fstr]) or a is None or b is None or target is None:
        return jsonify({"error":"Provide 'f_t','a','b','target'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b); tgt = _safe_float(target)
        t = Symbol('t')
        f = lambdify(t, sympify(fstr, locals={"Abs": Abs}), modules=['numpy'])
        F = lambda z: float(f(z) - tgt)
        F_vec = lambda zs: np.array([F(z) for z in np.atleast_1d(zs)])
        brackets = _scan_brackets(F_vec, a, b, 600)
        for L, R in brackets:
            root = _bisect_root(F, L, R)
            if root is not None:
                return jsonify({"t": str(_round3(root))})
        return jsonify({"error":"No solution in interval."}), 400
    except Exception as e:
        return jsonify({"error": f"solveForT failed: {e}"}), 400

# ---------------------- Run ----------------------
if __name__ == '__main__':
    # In production behind Render/Gunicorn, this block is ignored.
    app.run(host='0.0.0.0', port=5000)
