# Bzzzt! This is the final, production-grade code for your calculator brain.
# SymPy for symbolic work; SciPy / NumPy / numdifftools for numerical precision.
# VERSION 4.3 – AP Calc AB/BC + Precalc Regressions + round_final toggle
# Changes in 4.3:
#   • New helpers: _adaptive_eps, _as_funcs_for_calculus
#   • Hardened _scan_brackets (catches touching/near-zero roots)
#   • /critical works with f or f'  (expr_is_fprime)
#   • /extrema classifies via sign of f', supports f' + f_anchor
#   • New /inflectionFromFpp for direct f'' inputs

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

# ---------------------- New helpers ----------------------
def _adaptive_eps(a, b):
    """Scale sampling epsilon with interval length; keep a floor for tiny intervals."""
    L = float(b) - float(a)
    return max(1e-9, 1e-4 * L)

def _as_funcs_for_calculus(expr_str, var_name, mode="f"):
    """
    mode ∈ {"f", "fp", "fpp"}
    Returns callables f, fp, fpp (some may be None depending on mode).

    - mode="f":  expr is f; numeric derivatives supply fp and fpp.
    - mode="fp": expr is f'; we DO NOT re-differentiate for extrema classification;
                 we only need fp (and optionally numeric fpp).
    - mode="fpp": expr is f''.
    """
    x = Symbol(var_name)
    expr = sympify(expr_str, locals={"Abs": Abs})
    f = fp = fpp = None

    if mode == "f":
        f = lambdify(x, expr, modules=['numpy'])
        fp = lambda t: float(nd.Derivative(lambda s: f(s))(t))
        fpp = lambda t: float(nd.Derivative(lambda s: f(s), n=2)(t))

    elif mode == "fp":
        fp = lambdify(x, expr, modules=['numpy'])
        fpp = lambda t: float(nd.Derivative(lambda s: fp(s))(t))  # used only if requested

    elif mode == "fpp":
        fpp = lambdify(x, expr, modules=['numpy'])

    return f, fp, fpp

def _scan_brackets(f_vec, a, b, steps, tol_touch=1e-10):
    """
    Scan [a,b] with 'steps' subintervals to find brackets where f changes sign
    or hits (near-)zero at grid points (catches touching roots).
    f_vec: vectorized function that accepts numpy array and returns array.
    """
    xs = np.linspace(a, b, steps + 1)
    vals = f_vec(xs)
    br = []
    for i in range(len(xs) - 1):
        y1, y2 = vals[i], vals[i+1]
        if not np.isfinite(y1) or not np.isfinite(y2):
            continue
        if abs(y1) <= tol_touch:
            br.append((xs[i], xs[i]))          # exact/near grid root
        elif abs(y2) <= tol_touch:
            br.append((xs[i+1], xs[i+1]))      # exact/near grid root
        elif y1 * y2 < 0:
            br.append((xs[i], xs[i+1]))        # sign change
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
    expr_is_fprime = _bool(d, "expr_is_fprime", False)

    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        mode = "fp" if expr_is_fprime else "f"
        _, fp, _ = _as_funcs_for_calculus(expr, var, mode=mode)

        def fp_scalar(t): return float(fp(t))
        def fp_vec(ts): return np.array([fp_scalar(t) for t in np.atleast_1d(ts)])

        # Bracketed roots of f'
        brackets = _scan_brackets(fp_vec, a, b, steps)
        roots = []
        for L, R in brackets:
            r = _bisect_root(fp_scalar, L, R)
            if r is not None and a - 1e-12 <= r <= b + 1e-12:
                roots.append(r)

        # Also catch near-zero grid points (touching zeros)
        xs = np.linspace(a, b, steps + 1)
        vals = fp_vec(xs)
        for xi, vi in zip(xs, vals):
            if np.isfinite(vi) and abs(vi) < 1e-8:
                roots.append(float(xi))

        roots = _dedupe_sorted(roots)
        roots = [ _maybe_round(r, round_final) for r in roots ]
        return jsonify({"critical_points": roots})
    except Exception as e:
        return jsonify({"error": f"Critical-point search failed: {e}"}), 400

@app.route('/extrema', methods=['POST'])
def extrema():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    expr_is_fprime = _bool(d, "expr_is_fprime", False)
    anchor = d.get('f_anchor', None)   # {"t":..., "value":...} optional

    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        mode = "fp" if expr_is_fprime else "f"
        f, fp, _ = _as_funcs_for_calculus(expr, var, mode=mode)

        # If only f' is known but f-values are desired, synthesize f via anchor integration
        F = f
        if F is None and anchor:
            t = Symbol(var)
            fp_expr = sympify(expr, locals={"Abs": Abs})
            fp_num = lambdify(t, fp_expr, modules=['numpy'])
            t0 = _safe_float(anchor.get("t"))
            f0 = _safe_float(anchor.get("value"))
            def F(tt):
                val, _ = quad(fp_num, t0, tt, limit=200)
                return f0 + val

        def fp_scalar(t): return float(fp(t))
        def fp_vec(ts): return np.array([fp_scalar(t) for t in np.atleast_1d(ts)])

        # 1) Critical points = zeros of f'
        cps_raw = []
        for L, R in _scan_brackets(fp_vec, a, b, steps):
            r = _bisect_root(fp_scalar, L, R)
            if r is not None:
                cps_raw.append(r)
        # catch near-zero grid points (touching zeros)
        xs = np.linspace(a, b, steps + 1)
        vals = fp_vec(xs)
        for xi, vi in zip(xs, vals):
            if np.isfinite(vi) and abs(vi) < 1e-8:
                cps_raw.append(float(xi))
        cps = _dedupe_sorted(cps_raw)

        # 2) Classify via sign change of f'
        eps = _adaptive_eps(a, b)
        results = []
        for xc in cps:
            sL = np.sign(fp_scalar(xc - eps))
            sR = np.sign(fp_scalar(xc + eps))
            kind = "neither"
            if sL > 0 and sR < 0: kind = "local_max"
            elif sL < 0 and sR > 0: kind = "local_min"
            f_val = None
            if F is not None:
                f_val = _maybe_round(float(F(xc)), round_final)
            results.append({
                "x": _maybe_round(xc, round_final),
                "type": kind,
                "f": f_val
            })

        # 3) Always include endpoints (with f if available)
        endpoints = [
            {"x": _maybe_round(a, round_final), "f": (_maybe_round(float(F(a)), round_final) if F else None)},
            {"x": _maybe_round(b, round_final), "f": (_maybe_round(float(F(b)), round_final) if F else None)}
        ]

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
        eps = _adaptive_eps(a, b)
        pts = []
        for xc in _dedupe_sorted(candidates):
            left = fpp_scalar(xc - eps)
            right = fpp_scalar(xc + eps)
            if np.isfinite(left) and np.isfinite(right) and np.sign(left) != np.sign(right):
                pts.append(_maybe_round(xc, round_final))
        return jsonify({"inflection_points": pts})
    except Exception as e:
        return jsonify({"error": f"Inflection search failed: {e}"}), 400

@app.route('/inflectionFromFpp', methods=['POST'])
def inflection_from_fpp():
    d = request.get_json()
    round_final = _bool(d, "round_final", True)
    expr, var, a, b = d.get('expression'), d.get('variable'), d.get('a'), d.get('b')
    steps = int(d.get('steps', 400))
    if not all([expr, var]) or a is None or b is None:
        return jsonify({"error": "Provide 'expression','variable','a','b'."}), 400
    try:
        a = _safe_float(a); b = _safe_float(b)
        _, _, fpp = _as_funcs_for_calculus(expr, var, mode="fpp")

        def fpp_scalar(t): return float(fpp(t))
        def fpp_vec(ts): return np.array([fpp_scalar(t) for t in np.atleast_1d(ts)])

        candidates = []
        for L, R in _scan_brackets(fpp_vec, a, b, steps):
            r = _bisect_root(fpp_scalar, L, R)
            if r is not None:
                candidates.append(r)

        eps = _adaptive_eps(a, b)
        pts = []
        for xc in _dedupe_sorted(candidates):
            sL = np.sign(fpp_scalar(xc - eps))
            sR = np.sign(fpp_scalar(xc + eps))
            if np.isfinite(sL) and np.isfinite(sR) and sL != sR:
                pts.append(_maybe_round(xc, round_final))
        return jsonify({"inflection_points": pts})
    except Exception as e:
        return jsonify({"error": f"Inflection-from-fpp failed: {e}"}), 400

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
    from scipy.optimize import curve_fit
    d = request.get_json()
    model_in = (d.get('model') or "").strip().lower()
    x = d.get('x'); y = d.get('y')
    round_final = _bool(d, "round_final", True)

    if x is None or y is None or not len(x) or not len(y):
        return jsonify({"error":"Provide 'model','x','y' arrays."}), 400

    # Accept common synonyms
    alias = {
        "linear":"linear", "lin":"linear",
        "quadratic":"quadratic", "quad":"quadratic",
        "cubic":"cubic",
        "quartic":"quartic", "4th":"quartic", "degree4":"quartic",
        "exp":"exp", "exponential":"exp",
        "log":"log", "logarithmic":"log",
        "power":"power", "powerlaw":"power",
        "sin":"sinusoidal", "sine":"sinusoidal", "sinusoidal":"sinusoidal"
    }
    model = alias.get(model_in)
    if model is None:
        return jsonify({"error":"Invalid model. Use linear/quadratic/cubic/quartic/exp/log/power/sinusoidal."}), 400

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # R² helper
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1.0 - (ss_res/ss_tot if ss_tot != 0 else 0.0)

    try:
        if model in ("linear","quadratic","cubic","quartic"):
            deg = {"linear":1,"quadratic":2,"cubic":3,"quartic":4}[model]
            # stabilize high-degree poly with centered x
            x_mu, x_sigma = np.mean(x), (np.std(x) if np.std(x)>0 else 1.0)
            Xn = (x - x_mu)/x_sigma
            coeffs_n = np.polyfit(Xn, y, deg=deg)
            y_hat = np.polyval(coeffs_n, Xn)
            coeffs = np.polyfit(x, y_hat, deg=deg)
            y_pred = np.polyval(coeffs, x)
            R2 = r2_score(y, y_pred)
            params = [ _maybe_round(c, round_final) for c in coeffs ]

        elif model == "exp":
            if np.any(y <= 0):
                return jsonify({"error":"Exponential regression requires y > 0."}), 400
            Y = np.log(y)
            B, A = np.polyfit(x, Y, 1)   # Y = A + Bx
            a = np.exp(A); b = np.exp(B)
            y_pred = a*np.power(b, x)
            R2 = r2_score(y, y_pred)
            params = [ _maybe_round(a, round_final), _maybe_round(b, round_final) ]

        elif model == "log":
            if np.any(x <= 0):
                return jsonify({"error":"Logarithmic regression requires x > 0."}), 400
            Lx = np.log(x)
            a, b = np.polyfit(Lx, y, 1)  # y = a*ln x + b
            y_pred = a*Lx + b
            R2 = r2_score(y, y_pred)
            params = [ _maybe_round(a, round_final), _maybe_round(b, round_final) ]

        elif model == "power":
            if np.any(x <= 0) or np.any(y <= 0):
                return jsonify({"error":"Power regression requires x > 0 and y > 0."}), 400
            Lx, Ly = np.log(x), np.log(y)
            b, ln_a = np.polyfit(Lx, Ly, 1)  # ln y = ln a + b ln x
            a = np.exp(ln_a)
            y_pred = a*np.power(x, b)
            R2 = r2_score(y, y_pred)
            params = [ _maybe_round(a, round_final), _maybe_round(b, round_final) ]

        elif model == "sinusoidal":
            # Stable harmonic fit: y ≈ A sin(ωx) + B cos(ωx) + D
            x_span = (np.max(x) - np.min(x)) if len(x) > 1 else 1.0

            # Frequency range (rad/unit) based on span
            w_min = float(d.get("w_min", 2*np.pi / (4.0 * x_span + 1e-9)))   # long periods
            w_max = float(d.get("w_max", 2*np.pi / (0.5 * x_span + 1e-9)))   # short periods
            if w_max < w_min:
                w_min, w_max = w_max, w_min

            # Optional period hint
            period_hint = d.get("period_guess", None)
            if period_hint is not None:
                try:
                    w_hint = 2*np.pi / float(period_hint)
                    w_min = min(w_min, w_hint*2.0)
                    w_max = max(w_max, w_hint/2.0)
                except Exception:
                    pass

            def fit_for_omega(omega: float):
                S = np.sin(omega * x)
                Cc = np.cos(omega * x)
                M = np.column_stack([S, Cc, np.ones_like(x)])
                coef, *_ = np.linalg.lstsq(M, y, rcond=None)
                A_hat, B_hat, D_hat = coef
                y_pred = A_hat*S + B_hat*Cc + D_hat
                R2 = r2_score(y, y_pred)
                return R2, A_hat, B_hat, D_hat

            # coarse sweep
            ws = np.linspace(w_min, w_max, 300)
            best = (-np.inf, None, None, None, None)
            for w in ws:
                R2w, A, B, D0 = fit_for_omega(w)
                if R2w > best[0]:
                    best = (R2w, w, A, B, D0)

            # local refine
            R2_best, w_best, A_best, B_best, D_best = best
            w_lo = max(w_min, w_best*0.9)
            w_hi = min(w_max, w_best*1.1)
            ws2 = np.linspace(w_lo, w_hi, 120)
            for w in ws2:
                R2w, A, B, D0 = fit_for_omega(w)
                if R2w > R2_best:
                    R2_best, w_best, A_best, B_best, D_best = R2w, w, A, B, D0

            amp   = float(np.hypot(A_best, B_best))
            phase = float(np.arctan2(B_best, A_best))
            freq  = float(w_best)
            offset= float(D_best)

            params = [
                _maybe_round(amp,   round_final),
                _maybe_round(freq,  round_final),
                _maybe_round(phase, round_final),
                _maybe_round(offset,round_final)
            ]
            R2 = R2_best

        else:
            return jsonify({"error":"Unhandled model."}), 400

        return jsonify({"model": model, "params": params, "R2": _maybe_round(R2, round_final)})

    except Exception as e:
        return jsonify({"error": f"Regression failed: {e}"}), 400



# ---------------------- Run ----------------------
if __name__ == '__main__':
    # In production behind Render/Gunicorn, this block is ignored.
    app.run(host='0.0.0.0', port=5000)

