# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Training-free inference acceleration for SAM 3D flow-matching.

Two methods, ported from the TRELLIS / Hunyuan-2.1 acceleration work and
generalised to SAM3D's **PyTree velocities** (the solver's ``x_t`` and the
backbone velocity are ``torch.utils._pytree`` structures, not single tensors):

  * **HiCache** — the (CFG-combined) velocity at *skipped* solver steps is
    forecast with a dual-scaled physicist's Hermite polynomial instead of calling
    the dynamics function, skipping ``(interval-1)/interval`` of the model
    evaluations. Hooked at the ODE-solver level (it wraps ``dynamics_fn``).

  * **Adaptive-CFG** (Adaptive Guidance, arXiv:2312.12487) — once the conditional
    and unconditional velocities align (cosine >= ``gamma_bar``) the unconditional
    backbone pass is dropped and the guidance term is reconstructed from cached
    anchors. Hooked inside ``ClassifierFreeGuidance.inner_forward``. SAM3D's CFG is
    ``(1+w)*y_cond - w*y_uncond`` (the TRELLIS convention), so the guidance term is
    ``g = w*(y_cond - y_uncond)`` and ``v_cfg = y_cond + g``.

Everything here is model-agnostic except that it operates on PyTrees via
``_pytree.tree_map``. The solver / CFG modules call these helpers directly — there
is NO runtime monkey-patching. The Hermite/finite-difference scalar coefficients
are identical across all leaves, so a forecast is one ``tree_map`` per order.
"""
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils import _pytree


# --------------------------------------------------------------------------- #
# Hermite basis (scalar coefficients, applied per leaf)                        #
# --------------------------------------------------------------------------- #
def physicists_hermite(n: int, x: float) -> float:
    """Physicist's Hermite ``H_n(x)`` via the stable recurrence (scalar)."""
    if n < 0:
        raise ValueError(f"Hermite order must be >= 0, got {n}")
    if n == 0:
        return 1.0
    h_prev, h_curr = 1.0, 2.0 * x
    if n == 1:
        return h_curr
    for k in range(1, n):
        h_prev, h_curr = h_curr, 2.0 * x * h_curr - 2.0 * k * h_prev
    return h_curr


def hermite_coeff(order: int, k: int, sigma: float) -> float:
    """Forecast coefficient ``Htilde_order(k) / order!`` (a scalar, same for all
    leaves). ``Htilde_n(x) = sigma^n H_n(sigma x)``."""
    x = float(k)
    htilde = (sigma ** order) * physicists_hermite(order, sigma * x)
    return htilde / math.factorial(order)


# --------------------------------------------------------------------------- #
# tree helpers                                                                #
# --------------------------------------------------------------------------- #
def tree_axpy(c: float, a: Any, b: Any) -> Any:
    """``a + c * b`` leafwise (c scalar)."""
    return _pytree.tree_map(lambda av, bv: av + c * bv, a, b)


def tree_sub_div(a: Any, b: Any, d: float) -> Any:
    """``(a - b) / d`` leafwise."""
    return _pytree.tree_map(lambda av, bv: (av - bv) / d, a, b)


def tree_detach(a: Any) -> Any:
    return _pytree.tree_map(lambda v: v.detach() if isinstance(v, torch.Tensor) else v, a)


def tree_cosine(a: Any, b: Any, eps: float = 1e-12) -> float:
    """Cosine similarity over the concatenated leaves of two velocity trees."""
    la = [t.reshape(-1).float() for t in _pytree.tree_leaves(a) if isinstance(t, torch.Tensor)]
    lb = [t.reshape(-1).float() for t in _pytree.tree_leaves(b) if isinstance(t, torch.Tensor)]
    fa, fb = torch.cat(la), torch.cat(lb)
    return float((torch.dot(fa, fb) / (fa.norm() * fb.norm() + eps)).item())


# --------------------------------------------------------------------------- #
# HiCache (tree-aware velocity forecast)                                       #
# --------------------------------------------------------------------------- #
def hicache_init(num_steps, interval=4, max_order=1, first_enhance=2,
                 end_enhance=None, sigma=0.5) -> Dict[str, Any]:
    if interval < 1 or max_order < 1:
        raise ValueError("interval and max_order must be >= 1")
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0,1), got {sigma}")
    return {
        "num_steps": int(num_steps), "interval": int(interval), "max_order": int(max_order),
        "first_enhance": int(first_enhance),
        "end_enhance": int(end_enhance if end_enhance is not None else num_steps),
        "sigma": float(sigma),
        "step": 0, "counter": 0,
        "activated_steps": [], "derivatives": {}, "prev_derivatives": {},
        "telemetry": {
            "status": "active",
            "decisions": {"full": 0, "forecast": 0},
            "method_counts": {"hermite": 0},
            "fallbacks": {},
        },
    }


def hicache_decide(state: Dict[str, Any]) -> str:
    step = state["step"]
    if step < state["first_enhance"] or step >= state["end_enhance"] \
            or not state["derivatives"] \
            or state["counter"] >= state["interval"] - 1:
        state["counter"] = 0
        state["activated_steps"].append(step)
        decision = "full"
    else:
        state["counter"] += 1
        decision = "forecast"
    decisions = state["telemetry"]["decisions"]
    decisions[decision] = int(decisions.get(decision, 0)) + 1
    return decision


def hicache_update_tree(state: Dict[str, Any], velocity_tree: Any) -> None:
    prev = state["derivatives"]
    new_deriv = {0: velocity_tree}
    if len(prev) > 0:
        acts = state["activated_steps"]
        dist = max(int(acts[-1] - acts[-2]) if len(acts) >= 2 else state["interval"], 1)
        for order in range(state["max_order"]):
            if order not in prev:
                break
            new_deriv[order + 1] = tree_sub_div(new_deriv[order], prev[order], dist)
    state["prev_derivatives"] = prev
    state["derivatives"] = new_deriv


def hicache_forecast_tree(state: Dict[str, Any]) -> Any:
    deriv = state["derivatives"]
    if 0 not in deriv:
        state["telemetry"]["fallbacks"]["missing_anchor"] = (
            state["telemetry"]["fallbacks"].get("missing_anchor", 0) + 1
        )
        raise RuntimeError("hicache_forecast_tree called before any compute step")
    state["telemetry"]["method_counts"]["hermite"] += 1
    k = state["step"] - state["activated_steps"][-1]
    result = deriv[0]
    order = 1
    while order in deriv:
        result = tree_axpy(hermite_coeff(order, k, state["sigma"]), result, deriv[order])
        order += 1
    return result


def hicache_telemetry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a detached snapshot of decisions, methods, and fallback reasons."""
    return deepcopy(state.get("telemetry", {}))


# --------------------------------------------------------------------------- #
# Adaptive-CFG (tree-aware guidance forecast)                                  #
# --------------------------------------------------------------------------- #
def adaptive_cfg_init(num_steps, gamma_bar=0.94, warmup=2, max_order=1) -> Dict[str, Any]:
    if not (0.0 <= gamma_bar <= 1.0):
        raise ValueError(f"gamma_bar must be in [0,1], got {gamma_bar}")
    return {"num_steps": int(num_steps), "gamma_bar": float(gamma_bar), "warmup": int(warmup),
            "max_order": int(max_order), "step": 0, "anchors": [], "last_gamma": None,
            "n_full": 0, "n_skip": 0}


def adaptive_cfg_decide(state: Dict[str, Any], gamma: Optional[float]) -> bool:
    """True -> run the full (uncond) pass this step."""
    step = state["step"]
    if step < state["warmup"] or step >= state["num_steps"] - 1:
        return True
    if len(state["anchors"]) == 0 or gamma is None:
        return True
    return gamma < state["gamma_bar"]


def guidance_term_tree(y_cond: Any, y_uncond: Any, strength: float) -> Any:
    """SAM3D guidance term ``g = strength * (y_cond - y_uncond)`` (leafwise)."""
    return _pytree.tree_map(lambda c, u: strength * (c - u), y_cond, y_uncond)


def forecast_guidance_tree(anchors: List[Tuple[int, Any]], step: int, max_order: int = 1) -> Any:
    """Newton divided-difference forecast of the guidance-term TREE at ``step``.
    Handles non-uniform anchor spacing (robust to HiCache-skipped steps)."""
    if len(anchors) == 0:
        raise ValueError("forecast_guidance_tree requires at least one anchor")
    if len(anchors) == 1 or max_order < 1:
        return anchors[-1][1]
    used = anchors[-(max_order + 1):]
    xs = [float(s) for s, _ in used]
    col = [g for _, g in used]
    n = len(used)
    coeffs = [col[0]]
    for k in range(1, n):
        col = [tree_sub_div(col[i + 1], col[i], xs[i + k] - xs[i]) for i in range(n - k)]
        coeffs.append(col[0])
    x = float(step)
    result = coeffs[-1]
    for k in range(n - 2, -1, -1):
        xk = xs[k]
        result = _pytree.tree_map(lambda r, c: r * (x - xk) + c, result, coeffs[k])
    return result


def reconstruct_cfg_tree(y_cond: Any, g: Any) -> Any:
    """``v_cfg = y_cond + g`` (leafwise) — the skip-step reconstruction."""
    return _pytree.tree_map(lambda c, gv: c + gv, y_cond, g)


# --------------------------------------------------------------------------- #
