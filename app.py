# Bzzzt! This is the final, production-grade code for your calculator brain.
# SymPy for symbolic work; SciPy / NumPy / numdifftools for numerical precision.
# VERSION 4.2 – AP Calc AB/BC + Precalc Regressions + round_final toggle

from flask import Flask, request, jsonify
from flask_cors import CORS
from sympy import sympify, N, SympifyError, diff, integrate, Symbol, Abs, lambdify
from scipy.integrate import quad
from scipy.optimize import root_scalar, curve_fit
import numpy as np
import numdifftools as nd  # High-precision numerical derivatives

app = Flask(__name__)
CORS(app)

# ---------------------- Precision helpers ----------------------
def _maybe_round(x, round_final=True):
    """Return 3-decimal value only if round_final=True; else full precision float."""
    try:
        xf = float(x)
    except Exception:
        return x
    return float(f"{xf:.3f}") if round_final else xf

def _bool(data, key, default=True):
    v = data.get(key, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    return bool(v)

def _safe_float(x):
    return float(N(sympify(str(x))))

def _as_func(expr_str, var_name):
    x = Symbol(var_name)
    expr = sympify(expr_str, locals={"Abs": Abs})
    return x, expr, lambdify(x, expr, modules=['numpy'])

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

def _dedupe_sorted(vals, tol=1e-9):
    """Sort and dedupe a list of floats without forcing rounding, using tolerance."""
    vs = sorted(float(v) for v in vals)
    out = []
    for v in vs:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out

# ---------------------- Health Check ----------------------
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Calculator brain is online and ready!"})

# ---------------------- Evaluate (numeric) ----------------------
@app.route('/evaluate', methods=['POST'])
def evaluate_expression_endpoint():
    data = request.get_json()
    expression = data.get('expression')
    if not expression:
        return jsonify({"error": "Invalid request. Please provide an 'expression'."}), 400
    round_final = _bool(data, "round_final", True)
    try:
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        val = float(N(sympy_expr, 50))
        return jsonify({"result": str(_maybe_round(val, round_final))})
    except (SympifyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid mathematical expression provided: {str(e)}"}), 400

# ---------------------- Differentiate (symbolic) ----------------------
@app.route('/differentiate', methods=['POST'])
def differentiate_expression():
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

# ---------------------- Integrate (indefinite, symbolic) ----------------------
@app.route('/integrate', methods=['POST'])
def integrate_expression():
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
    data = request.get_json()
    round_final = _bool(data, "round_final", True)
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
        return jsonify({"result": str(_maybe_round(integral_result, round_final))})
    except Exception as e:
        return jsonify({"error": f"Numerical integration failed. {str(e)}"}), 400

# ---------------------- Numeric Derivative at a point ----------------------
@app.route('/nDeriv', methods=['POST'])
def numerical_differentiate():
    data = request.get_json()
    round_final = _bool(data, "round_final", True)
    expression, variable, point = data.get('expression'), data.get('variable'), data.get('point')
    if not all([expression, variable, point]):
        return jsonify({"error": "Provide 'expression', 'variable', and 'point'."}), 400
    try:
        x_val = _safe_float(point)
        x = Symbol(variable)
        sympy_expr = sympify(expression, locals={"Abs": Abs})
        f = lambdify(x, sympy_expr, modules=['numpy'])
        dval = nd.Derivative(lambda t: f(t))(x_val)
        return jsonify({"result": str(_maybe_round(dval, round_final))})
    except Exception as e:
        return jsonify({"error": f"Numerical differentiation failed. {e}"}), 400

# ---------------------- Zeros / Critical / Extrema / Inflection ----------------------
@app.route('/zeros', methods=['POST'])
def find_zeros():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(expr, var)
        F_vec = lambda ts: f(ts)
        brackets = _scan_brackets(F_vec, a, b, steps)
        roots = []
        for L, R in brackets:
            r = _bisect_root(lambda t: float(f(t)), L, R)
            if r is not None and a - 1e-12 <= r <= b + 1e-12:
                roots.append(r)
        roots = _dedupe_sorted(roots)
        roots = [ _maybe_round(r, round_final) for r in roots ]
        return jsonify({"zeros": roots})
    except Exception as e:
        return jsonify({"error": f"Zero-finding failed: {e}"}), 400

@app.route('/critical', methods=['POST'])
def critical_points():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(expr, var)
        def fp_scalar(t): return float(nd.Derivative(lambda s: f(s))(t))
        def fp_vec(ts): return np.array([fp_scalar(t) for t in np.atleast_1d(ts)])
        brackets = _scan_brackets(fp_vec, a, b, steps)
        cps = []
        for L, R in brackets:
            r = _bisect_root(fp_scalar, L, R)
            if r is not None and a - 1e-12 <= r <= b + 1e-12:
                cps.append(r)
        cps = _dedupe_sorted(cps)
        cps = [ _maybe_round(c, round_final) for c in cps ]
        return jsonify({"critical_points": cps})
    except Exception as e:
        return jsonify({"error": f"Critical-point search failed: {e}"}), 400

@app.route('/extrema', methods=['POST'])
def extrema():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(expr, var)
        def fp_scalar(t): return float(nd.Derivative(lambda s: f(s))(t))
        def fp_vec(ts): return np.array([fp_scalar(t) for t in np.atleast_1d(ts)])
        cps_raw = []
        for L, R in _scan_brackets(fp_vec, a, b, steps):
            r = _bisect_root(fp_scalar, L, R)
            if r is not None: cps_raw.append(r)
        cps = _dedupe_sorted(cps_raw)
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
            results.append({"x": _maybe_round(xc, round_final),
                            "type": kind,
                            "f": _maybe_round(f(xc), round_final)})
        endpoints = [{"x": _maybe_round(a, round_final), "f": _maybe_round(f(a), round_final)},
                     {"x": _maybe_round(b, round_final), "f": _maybe_round(f(b), round_final)}]
        return jsonify({"extrema": results, "endpoints": endpoints})
    except Exception as e:
        return jsonify({"error": f"Extrema classification failed: {e}"}), 400

@app.route('/inflection', methods=['POST'])
def inflection_points():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, f = _as_func(expr, var)
        def fpp_scalar(t): return float(nd.Derivative(lambda s: f(s), n=2)(t))
        def fpp_vec(ts): return np.array([fpp_scalar(t) for t in np.atleast_1d(ts)])
        candidates = []
        for L, R in _scan_brackets(fpp_vec, a, b, steps):
            r = _bisect_root(fpp_scalar, L, R)
            if r is not None:
                candidates.append(r)
        pts = []
        for xc in _dedupe_sorted(candidates):
            eps = 1e-3
            left = fpp_scalar(xc - eps)
            right = fpp_scalar(xc + eps)
            if left * right < 0:
                pts.append(_maybe_round(xc, round_final))
        return jsonify({"inflection_points": pts})
    except Exception as e:
        return jsonify({"error": f"Inflection search failed: {e}"}), 400

# ---------------------- Cartesian Intersections ----------------------
@app.route('/intersections', methods=['POST'])
def intersections():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        xs = _dedupe_sorted(xs)
        pts = [{"x": _maybe_round(t, round_final), "y": _maybe_round(f(t), round_final)} for t in xs]
        xs = [ _maybe_round(t, round_final) for t in xs ]
        return jsonify({"x": xs, "points": pts})
    except Exception as e:
        return jsonify({"error": f"Intersections failed: {e}"}), 400

# ---------------------- Areas: Between Curves ----------------------
@app.route('/areaBetween', methods=['POST'])
def area_between():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        pts = [a] + [pt for pair in brackets for pt in pair if a < pt < b] + [b]
        pts = sorted(list({round(p, 12) for p in pts}))
        total = 0.0
        for L, R in zip(pts[:-1], pts[1:]):
            integrand = lambda t: np.abs(H(t))
            val, _ = quad(integrand, L, R, limit=200)
            total += val
        return jsonify({"result": str(_maybe_round(total, round_final))})
    except Exception as e:
        return jsonify({"error": f"Area between curves failed: {e}"}), 400

# ---------------------- Tabular Calculus ----------------------
@app.route('/tabularIntegral', methods=['POST'])
def tabular_integral():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        return jsonify({"result": str(_maybe_round(area, round_final))})
    except Exception as e:
        return jsonify({"error": f"Tabular integral failed: {e}"}), 400

@app.route('/tabularDerivative', methods=['POST'])
def tabular_derivative():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        return jsonify({"result": str(_maybe_round(der, round_final))})
    except Exception as e:
        return jsonify({"error": f"Tabular derivative failed: {e}"}), 400

# ---------------------- Euler’s Method ----------------------
@app.route('/euler', methods=['POST'])
def euler_method_endpoint():
    """
    Euler's method for dy/dx = f(x,y).
    JSON: { "fxy": "x - y", "x0":0, "y0":1, "h":0.2, "n":5, "round_final":false }
    """
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    fxy, x0, y0, h, n = d.get('fxy'), d.get('x0'), d.get('y0'), d.get('h'), d.get('n')
    if not all([fxy is not None, x0 is not None, y0 is not None, h is not None, n is not None]):
        return jsonify({"error":"Provide 'fxy','x0','y0','h','n'."}), 400
    try:
        x_sym, y_sym = Symbol('x'), Symbol('y')
        f_expr = sympify(fxy, locals={"Abs": Abs})
        f = lambdify((x_sym, y_sym), f_expr, modules=['numpy'])
        x = _safe_float(x0); y = _safe_float(y0); h = _safe_float(h); n = int(n)
        pts = [{"x": _maybe_round(x, round_final), "y": _maybe_round(y, round_final)}]
        for _ in range(n):
            y = y + h * float(f(x, y))
            x = x + h
            pts.append({"x": _maybe_round(x, round_final), "y": _maybe_round(y, round_final)})
        return jsonify({"points": pts, "y_n": _maybe_round(y, round_final)})
    except Exception as e:
        return jsonify({"error": f"Euler failed: {e}"}), 400

# ---------------------- Motion: position & total distance ----------------------
@app.route('/positionFromVelocity', methods=['POST'])
def position_from_velocity():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    v, var, a, b, s0 = d.get('v'), d.get('variable'), d.get('a'), d.get('b'), d.get('s0', 0)
    if not all([v, var]) or a is None or b is None:
        return jsonify({"error":"Provide 'v','variable','a','b' (and optional 's0')."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b); s0 = _safe_float(s0)
        x, _, vf = _as_func(v, var)
        val, _ = quad(vf, a, b, limit=200)
        return jsonify({"result": str(_maybe_round(s0 + val, round_final))})
    except Exception as e:
        return jsonify({"error": f"positionFromVelocity failed: {e}"}), 400

@app.route('/totalDistance', methods=['POST'])
def total_distance():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    v, var, a, b = d.get('v'), d.get('variable'), d.get('a'), d.get('b')
    if not all([v, var]) or a is None or b is None:
        return jsonify({"error":"Provide 'v','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        x, _, vf = _as_func(v, var)
        abs_v = lambda t: np.abs(vf(t))
        val, _ = quad(abs_v, a, b, limit=200)
        return jsonify({"result": str(_maybe_round(val, round_final))})
    except Exception as e:
        return jsonify({"error": f"totalDistance failed: {e}"}), 400

# ---------------------- Parametric & Polar (BC) ----------------------
@app.route('/parametricSlope', methods=['POST'])
def parametric_slope():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        return jsonify({"result": str(_maybe_round(slope, round_final))})
    except Exception as e:
        return jsonify({"error": f"parametricSlope failed: {e}"}), 400

@app.route('/parametricArcLength', methods=['POST'])
def parametric_arc_length():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        return jsonify({"result": str(_maybe_round(val, round_final))})
    except Exception as e:
        return jsonify({"error": f"parametricArcLength failed: {e}"}), 400

@app.route('/polarArea', methods=['POST'])
def polar_area():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        return jsonify({"result": str(_maybe_round(val, round_final))})
    except Exception as e:
        return jsonify({"error": f"polarArea failed: {e}"}), 400

@app.route('/polarIntersections', methods=['POST'])
def polar_intersections():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        roots = _dedupe_sorted(roots)
        roots = [ _maybe_round(r, round_final) for r in roots ]
        return jsonify({"thetas": roots})
    except Exception as e:
        return jsonify({"error": f"polarIntersections failed: {e}"}), 400

# ---------------------- Series & Taylor (BC) ----------------------
@app.route('/partialSum', methods=['POST'])
def partial_sum():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
        return jsonify({"result": str(_maybe_round(s, round_final))})
    except Exception as e:
        return jsonify({"error": f"partialSum failed: {e}"}), 400

@app.route('/altSeriesError', methods=['POST'])
def alternating_series_error():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    a_n, n, idx = d.get('a_n'), d.get('n'), d.get('index', 'n')
    if a_n is None or n is None:
        return jsonify({"error":"Provide 'a_n' and 'n'."}), 400
    try:
        n = int(n)
        k = Symbol(idx)
        term = sympify(a_n, locals={"Abs": Abs})
        f = lambdify(k, term, modules=['numpy'])
        err = abs(float(f(n + 1)))
        return jsonify({"error_bound": str(_maybe_round(err, round_final))})
    except Exception as e:
        return jsonify({"error": f"altSeriesError failed: {e}"}), 400

@app.route('/taylorPoly', methods=['POST'])
def taylor_poly():
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
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    fstr, var, x0, n, xe = d.get('f'), d.get('variable'), d.get('x0'), d.get('n'), d.get('x_eval')
    if not all([fstr, var, x0 is not None, n is not None, xe is not None]):
        return jsonify({"error":"Provide 'f','variable','x0','n','x_eval'."}), 400
    try:
        x0 = _safe_float(x0); n = int(n); xe = _safe_float(xe)
        x = Symbol(var)
        f = sympify(fstr, locals={"Abs": Abs})
        poly = f.series(x, x0, n + 1).removeO()
        pfunc = lambdify(x, poly, modules=['numpy'])
        val = pfunc(xe)
        return jsonify({"result": str(_maybe_round(val, round_final))})
    except Exception as e:
        return jsonify({"error": f"taylorApprox failed: {e}"}), 400

# ---------------------- Logistic / Exponential + solver ----------------------
@app.route('/logistic', methods=['POST'])
def logistic_value():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    P0, k, L, t = d.get('P0'), d.get('k'), d.get('L'), d.get('t')
    if not all([P0, k, L, t]):
        return jsonify({"error":"Provide 'P0','k','L','t'."}), 400
    try:
        P0 = _safe_float(P0); k = _safe_float(k); L = _safe_float(L); t = _safe_float(t)
        val = L / (1 + ((L - P0) / P0) * np.exp(-k * t))
        return jsonify({"result": str(_maybe_round(val, round_final))})
    except Exception as e:
        return jsonify({"error": f"logistic failed: {e}"}), 400

@app.route('/expGrowth', methods=['POST'])
def exponential_value():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    P0, k, t = d.get('P0'), d.get('k'), d.get('t')
    if not all([P0, k, t]):
        return jsonify({"error":"Provide 'P0','k','t'."}), 400
    try:
        P0 = _safe_float(P0); k = _safe_float(k); t = _safe_float(t)
        return jsonify({"result": str(_maybe_round(P0 * np.exp(k * t), round_final))})
    except Exception as e:
        return jsonify({"error": f"expGrowth failed: {e}"}), 400

@app.route('/solveForT', methods=['POST'])
def solve_for_t():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
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
                return jsonify({"t": str(_maybe_round(root, round_final))})
        return jsonify({"error":"No solution in interval."}), 400
    except Exception as e:
        return jsonify({"error": f"solveForT failed: {e}"}), 400

# ---------------------- Precalculus: Regressions ----------------------
@app.route('/regress', methods=['POST'])
def regression_fit():
    """
    Regression fit for model ∈ {linear, quadratic, cubic, quartic, exp, log, power, sinusoidal}.
    JSON: {"model":"quadratic","x":[...],"y":[...],"round_final":true}
    Returns: {"model":..., "params":[...], "R2":...}
    """
    d = request.get_json()
    model = d.get('model')
    x = d.get('x')
    y = d.get('y')
    round_final = _bool(d, "round_final", True)

    if x is None or y is None or model is None:
        return jsonify({"error":"Provide 'model','x','y'."}), 400

    try:
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        # Model definitions
        def linear(x,a,b): return a*x+b
        def quadratic(x,a,b,c): return a*x**2+b*x+c
        def cubic(x,a,b,c,d): return a*x**3+b*x**2+c*x+d
        def quartic(x,a,b,c,d,e): return a*x**4+b*x**3+c*x**2+d*x+e
        def exponential(x,a,b): return a*np.power(b, x)
        def logarithmic(x,a,b): return a*np.log(x)+b
        def power(x,a,b): return a*np.power(x, b)
        def sinusoidal(x,A,B,C,D): return A*np.sin(B*x + C) + D

        models = {
            "linear":      (linear,      [1.0, 0.0]),
            "quadratic":   (quadratic,   [1.0, 0.0, 0.0]),
            "cubic":       (cubic,       [1.0, 0.0, 0.0, 0.0]),
            "quartic":     (quartic,     [1.0, 0.0, 0.0, 0.0, 0.0]),
            "exp":         (exponential, [1.0, 1.1]),
            "log":         (logarithmic, [1.0, 0.0]),
            "power":       (power,       [1.0, 1.0]),
            "sinusoidal":  (sinusoidal,  [1.0, 1.0, 0.0, 0.0])
        }
        if model not in models:
            return jsonify({"error":"Invalid model type. Use linear/quadratic/cubic/quartic/exp/log/power/sinusoidal."}), 400

        func, p0 = models[model]

        # Domain guards for log/power models
        if model in ("log", "power"):
            if np.any(x <= 0):
                return jsonify({"error":"log/ power regression requires x > 0."}), 400

        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)

        params = [ _maybe_round(v, round_final) for v in popt ]
        return jsonify({"model": model, "params": params, "R2": _maybe_round(r2, round_final)})

    except Exception as e:
        return jsonify({"error": f"Regression failed: {e}"}), 400

# ---------------------- Run ----------------------
if __name__ == '__main__':
    # In production behind Render/Gunicorn, this block is ignored.
    app.run(host='0.0.0.0', port=5000)
