"""CPU regressions for the Fast-SAM3D Hermite baseline adapter."""

import importlib.util
from pathlib import Path

import pytest
import torch

from sam3d_objects.model.backbone.generator.classifier_free_guidance import (
    ClassifierFreeGuidance,
)
from sam3d_objects.model.backbone.generator.flow_matching.accel import (
    adaptive_cfg_init,
    forecast_guidance_tree,
    guidance_term_tree,
    hicache_decide,
    hicache_forecast_tree,
    hicache_init,
    hicache_telemetry,
    hicache_update_tree,
    reconstruct_cfg_tree,
)
from sam3d_objects.model.backbone.generator.flow_matching.solver import (
    Euler,
    RungeKutta4,
)


def _tree(value):
    return {"x": torch.tensor([value, value + 1.0]), "nested": {"y": torch.tensor([[value]])}}


def _dynamics(x, _t):
    return {"x": 0.25 * x["x"] + 1.0, "nested": {"y": x["nested"]["y"] - 0.5}}


def test_hicache_telemetry_records_actual_hermite_forecasts():
    state = hicache_init(num_steps=8, interval=3, first_enhance=0)
    for step in range(6):
        state["step"] = step
        if hicache_decide(state) == "full":
            hicache_update_tree(state, _tree(float(step)))
        else:
            hicache_forecast_tree(state)
        state["step"] += 1

    telemetry = hicache_telemetry(state)
    assert telemetry["status"] == "active"
    assert telemetry["decisions"]["full"] + telemetry["decisions"]["forecast"] == 6
    assert telemetry["method_counts"]["hermite"] > 0


def test_interval_one_preserves_dense_euler_trace():
    times = torch.linspace(0, 1, 7)
    initial = _tree(0.0)
    dense = list(Euler().solve_iter(_dynamics, initial, times))[-1][0]
    cached = list(Euler().enable_hicache(interval=1).solve_iter(_dynamics, initial, times))[-1][0]
    assert torch.equal(dense["x"], cached["x"])
    assert torch.equal(dense["nested"]["y"], cached["nested"]["y"])


def test_non_euler_solver_reports_unsupported_cache_without_skipping():
    calls = 0

    def dynamics(x, t):
        nonlocal calls
        calls += 1
        return _dynamics(x, t)

    solver = RungeKutta4().enable_hicache(interval=3)
    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 4)))
    assert calls == 12
    telemetry = solver.get_hicache_telemetry()
    assert telemetry["fallbacks"] == {"unsupported_solver": 1}
    assert telemetry["solver"] == "RungeKutta4"


def test_cfg_helpers_preserve_sam3d_guidance_coefficients():
    cond, uncond = _tree(2.0), _tree(1.0)
    strength = 3.0
    guidance = guidance_term_tree(cond, uncond, strength)
    reconstructed = reconstruct_cfg_tree(cond, guidance)
    expected = {"x": 4 * cond["x"] - 3 * uncond["x"],
                "nested": {"y": 4 * cond["nested"]["y"] - 3 * uncond["nested"]["y"]}}
    assert torch.equal(reconstructed["x"], expected["x"])
    assert torch.equal(reconstructed["nested"]["y"], expected["nested"]["y"])
    assert torch.equal(forecast_guidance_tree([(0, guidance)], 5)["x"], guidance["x"])
    assert adaptive_cfg_init(6)["gamma_bar"] == 0.94
    cfg = ClassifierFreeGuidance(torch.nn.Identity())
    assert cfg.get_adaptive_guidance_telemetry()["status"] == "disabled"
    cfg.enable_adaptive_guidance().reset_adaptive_guidance(6)
    assert cfg.get_adaptive_guidance_telemetry()["status"] == "active"


def test_benchmark_import_is_portable_and_missing_metrics_is_actionable(monkeypatch):
    path = Path(__file__).parents[1] / "ab_accel_bench.py"
    spec = importlib.util.spec_from_file_location("fastsam3d_ab_bench", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    monkeypatch.delenv("FASTSAM3D_METRICS_PATH", raising=False)
    with pytest.raises(RuntimeError, match="FASTSAM3D_METRICS_PATH"):
        bench._load_metrics()
