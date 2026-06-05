<div align="center">

# Fast-SAM3D **+** &nbsp;·&nbsp; HiCache

**Image-to-3D, faster — Fast-SAM3D with a tree-aware HiCache cache on its slat-stage flow-matching.**

*Skip the dynamics network on most slat-stage solver steps and forecast the cached velocity
with a scaled-Hermite polynomial. Training-free, geometry-preserving, native (no monkey-patching).*

![training&#8209;free](https://img.shields.io/badge/training--free-%E2%9C%93-2e8f5c)
&nbsp;![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
&nbsp;![method HiCache](https://img.shields.io/badge/cache-HiCache%20(Hermite)-2e6db0)
&nbsp;![upstream Fast--SAM3D](https://img.shields.io/badge/fork%20of-Fast--SAM3D-555)

</div>

---

## What it is

A fork of [**Fast-SAM3D**](https://github.com/wlfeng0509/Fast-SAM3D) (single-image → textured 3D
mesh, itself a TaylorSeer-style accelerated [SAM3D](https://github.com/facebookresearch/sam-3d-objects))
that adds **HiCache** — a feature cache — to the **slat-stage flow-matching sampler**.

SAM3D generates in two flow-matching stages; the second (**slat**) stage integrates an ODE
`dx/dt = v_θ(x, t)` with a Euler solver, where each `v_θ` call is an expensive DiT forward.
HiCache runs the network on a sparse schedule and, on every **skipped** step, *forecasts* the
(CFG-combined) velocity from cached anchors instead of calling `v_θ`. SAM3D's velocities are
`torch.utils._pytree` structures (not single tensors), so this fork's HiCache is **tree-aware**:
the Hermite/finite-difference coefficients are scalars shared across all leaves, and a forecast
is one `tree_map` per order. The companion **Adaptive-CFG** drops the unconditional pass once it
aligns with the conditional one. Wiring is **native** — the solver and CFG modules call the
helpers directly; there is no runtime patching.

## Method — scaled-Hermite velocity forecasting

HiCache ([arXiv:2508.16984](https://arxiv.org/abs/2508.16984)) caches the velocity at compute
steps and extrapolates it to skipped steps with a **dual-scaled physicist's Hermite polynomial**
`H̃ₙ(x) = σⁿ Hₙ(σx)`. From the cached anchors it forms finite-difference "derivatives" `Δ⁰…Δᵐ` of
the velocity tree and predicts the velocity `k` steps past the last compute step as a
Hermite-weighted sum of those derivatives. The Hermite basis tempers the high-order extrapolation
that makes a plain Taylor (monomial) cache diverge, so it stays lossless at a larger skip interval
than TaylorSeer — but, being a **polynomial**, it still drifts as the skip grows. The exponential
successor that pushes the lossless skip further lives in [`fastsam3d-plus-plus`](../fastsam3d-plus-plus).

## Enable it (real API)

The cache attaches to the slat-stage `FlowMatching` generator's Euler solver. The model methods are
chainable (they return the model):

```python
# `fm` is the slat-stage FlowMatching generator inside the Fast-SAM3D pipeline.

# HiCache: forecast the velocity on skipped slat steps with the scaled-Hermite polynomial.
fm.enable_hicache(
    interval=3,        # run the DiT 1 step in `interval`; forecast the other (interval-1)
    max_order=1,       # highest finite-difference order used in the Hermite forecast
    first_enhance=2,   # always run full for the first N steps (warm-up)
    end_enhance=None,  # always run full for the final steps (defaults to the last step)
    sigma=0.5,         # Hermite scale σ ∈ (0,1)
)

# ... run the pipeline / sampler as usual ...

fm.disable_hicache()   # back to the dense (uncached) schedule
```

Under the hood `enable_hicache` stores the config on the Euler solver
(`ODESolver.enable_hicache`); on each step the solver calls `hicache_decide` and, when it returns
`"forecast"`, replaces the `dynamics_fn` (DiT) call with `hicache_forecast_tree` — see
[`accel.py`](sam3d_objects/model/backbone/generator/flow_matching/accel.py),
[`solver.py`](sam3d_objects/model/backbone/generator/flow_matching/solver.py), and
[`model.py`](sam3d_objects/model/backbone/generator/flow_matching/model.py).

## Results

On Fast-SAM3D's slat-stage FlowMatching, **HiCache (Hermite) is geometry-lossless (F1 = 1.000) out
to interval-3** (~1.4× over the uncached slat schedule). Past interval-3 the polynomial basis
starts to drift. For the **exponential (DMD) forecaster that holds quality two intervals further**
on the same FlowMatching substrate, see [`fastsam3d-plus-plus`](../fastsam3d-plus-plus) and the
standalone library [`hicache-plus-plus`](../hicache-plus-plus).

> The Hermite ⇄ Taylor swap on the SS stage is a wash — that stage already runs a fixed
> TaylorSeer stride, so the basis change there doesn't move latency. HiCache's gain is on the
> **slat** stage, where it forecasts the skipped velocities.

## Attribution

- **Fast-SAM3D** — © [wlfeng0509](https://github.com/wlfeng0509/Fast-SAM3D)
  ([arXiv:2602.05293](https://arxiv.org/abs/2602.05293)); built on
  [SAM3D](https://github.com/facebookresearch/sam-3d-objects). The upstream README is preserved in
  this fork's git history and its license/attribution is unchanged.
- **HiCache** — scaled-Hermite velocity forecasting, [arXiv:2508.16984](https://arxiv.org/abs/2508.16984).
  This fork is a clean tree-aware reimplementation on Fast-SAM3D's slat flow-matching.
- **Adaptive-CFG** — Adaptive Guidance, [arXiv:2312.12487](https://arxiv.org/abs/2312.12487).
- **HiCache++** (the exponential successor) — [`fastsam3d-plus-plus`](../fastsam3d-plus-plus) ·
  standalone [`hicache-plus-plus`](../hicache-plus-plus).

## Citation

If you use this fork, please cite the base model and the acceleration methods it builds on.

**Fast-SAM3D** (base model):

```bibtex
@misc{feng2026fastsam3d3dfyimagesfaster,
      title={Fast-SAM3D: 3Dfy Anything in Images but Faster}, 
      author={Weilun Feng and Mingqiang Wu and Zhiliang Chen and Chuanguang Yang and Haotong Qin and Yuqi Li and Xiaokun Liu and Guoxin Fan and Zhulin An and Libo Huang and Yulun Zhang and Michele Magno and Yongjun Xu},
      year={2026},
      eprint={2602.05293},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.05293}, 
}
```

**HiCache** (scaled-Hermite velocity forecasting):

```bibtex
@misc{hicache2025,
      title={HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting},
      eprint={2508.16984},
      archivePrefix={arXiv},
      year={2025}
}
```

**TaylorSeer** (the cache the base model accelerates with):

```bibtex
@misc{taylorseer2025,
      title={From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers},
      eprint={2503.06923},
      archivePrefix={arXiv},
      year={2025}
}
```

**Adaptive Guidance**:

```bibtex
@misc{adaptiveguidance2023,
      title={Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models},
      eprint={2312.12487},
      archivePrefix={arXiv},
      year={2023}
}
```

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[wlfeng0509/Fast-SAM3D](https://github.com/wlfeng0509/Fast-SAM3D), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.
