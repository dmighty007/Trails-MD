# Toy committor validation artifacts

The current evidence for `spawn_scheme: committor` is the convergence-gated
closed-loop run, not the older fixed-warmup `_v2` directory.

## Current closed-loop evidence

| directory | grid | iterations | switch mode | switch iteration | post-gate iterations | uniform RMSE | adaptive RMSE | result |
|---|---:|---:|---|---:|---:|---:|---:|---|
| `committor_guided_adaptive_long/` | 18x18 | 100 | convergence | 25 | 75 | 0.1884 | 0.1233 | adaptive wins |
| `committor_guided_adaptive_long_24x24/` | 24x24 | 100 | convergence | 39 | 61 | 0.2701 | 0.2691 | tied within noise |
| `committor_guided_adaptive/` | 18x18 | 30 | convergence | 25 | 5 | 0.4549 | 0.3949 | short smoke test only |

The 18x18 long run gives a real post-gate runway and supports the acquisition
logic at that resolution. The 24x24 run does not show a meaningful RMSE win at
the same 9600-shot budget, though it leaves fewer isolated states. Treat the
acquisition weights as plausible but not universally tuned.

## Superseded artifact

`committor_guided_adaptive_v2/` is historical. It used an older fixed-warmup
handoff and should not be compared directly with the current spawner, which
switches only after the exploration support gate opens.

## Resolution check

`validation_report.json` contains the static local-stitch resolution sweep
against the PDE reference. It shows that the local-stitch estimator itself is
accurate for this toy problem when the grid is adequately sampled:

| grid | shots/cell | max steps | RMSE vs PDE | isolated states |
|---:|---:|---:|---:|---:|
| 18x18 | 40 | 4000 | 0.0406 | 0 |
| 30x30 | 40 | 8000 | 0.0492 | 0 |
| 40x40 | 40 | 8000 | 0.0304 | 0 |

The closed-loop experiments are a harder allocation problem because they use a
fixed total budget rather than sampling every cell uniformly.
