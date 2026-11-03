# Burgers setup bakeoff (token + convergence)

| Setup | Best MSE | Trials | Commits | Total tokens | Avg tok/trial | Prompt toks | Completion toks | <0.026 @ trial | <0.02 @ trial |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `burger_base` | 0.068616 | 80 | 15 | 166109 | 2076.4 | 124087 | 42022 | never | never |
| `burger_breakwall` | 0.068123 | 100 | 10 | 184383 | 1843.8 | 144708 | 39675 | never | never |
| `burger_compact` | 0.023429 | 100 | 35 | 190133 | 1901.3 | 148547 | 41586 | 47 | never |

## Setup differences

- `burger_base`: 3 candidates, history=3, fixed gamma bands.
- `burger_breakwall`: 2 candidates, history=1, phase-adaptive gamma, hotspot + PDE/extrap telemetry, min_commit_drop=1e-5.
- `burger_compact`: same polish policy, shorter prompts (no PDE/extrap blocks), 2 candidates.

## Finding

- **Compact broke the prior ~0.026 wall** (best **0.0234** in 100 trials).
- Breakwall stayed near the old mid-loss plateau (~0.068) in this bakeoff.
- None reached 0.02 / 0.01 yet; compact is closest with similar total tokens to breakwall.
