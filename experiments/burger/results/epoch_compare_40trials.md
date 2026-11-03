# Burgers Base (~10K) Epoch-Level Convergence Comparison

Metric: **epochs / trials** and **loss drop per epoch-level step**, not wall-clock time.
One Adam epoch = one full pass over the train set. One online trial = one tournament slice update.

## Adam baseline (SiLU MLP)

- Parameters: `10065`
- Train MSE: `0.853918` -> best `0.009565` (final `0.009565`)
- Extrapolation MSE final: `108.448807`
- PDE residual: `7.446118`
- Epochs to best / stop: `205` / `205`
- Converged: `True` (`reached_train_mse_0.01`)
- Mean train-MSE drop per epoch: `0.003548`
- New-best advancements: `82` over `205` epochs (0.400 advances/epoch)

### Epochs to train-MSE convergence levels

| Level (train MSE) | First epoch hit | Epochs from start |
| ---: | ---: | ---: |
| <= 1.00 | 1 | 1 |
| <= 0.50 | 29 | 29 |
| <= 0.25 | 42 | 42 |
| <= 0.10 | 60 | 60 |
| <= 0.05 | 74 | 74 |
| <= 0.02 | 151 | 151 |
| <= 0.01 | 205 | 205 |

### Convergence per epoch band (25-epoch levels)

| Epochs | Start MSE | End MSE | Band drop | Drop / epoch |
| :--- | ---: | ---: | ---: | ---: |
| 1-25 | 0.733355 | 0.530487 | +0.202869 | +0.008115 |
| 26-50 | 0.506921 | 0.180051 | +0.326870 | +0.013075 |
| 51-75 | 0.158345 | 0.062059 | +0.096286 | +0.003851 |
| 76-100 | 0.051602 | 0.036676 | +0.014927 | +0.000597 |
| 101-125 | 0.038365 | 0.033719 | +0.004645 | +0.000186 |
| 126-150 | 0.032955 | 0.038350 | -0.005396 | -0.000216 |
| 151-175 | 0.019901 | 0.059407 | -0.039506 | -0.001580 |
| 176-200 | 0.025539 | 0.037215 | -0.011676 | -0.000467 |
| 201-205 | 0.011957 | 0.009565 | +0.002392 | +0.000478 |

### Condensed new-best trail (epoch-level)

| Epoch | Best train MSE | Delta |
| ---: | ---: | ---: |
| 1 | 0.733355 | +0.000000 |
| 2 | 0.727437 | -0.005918 |
| 3 | 0.724019 | -0.003418 |
| 4 | 0.720539 | -0.003481 |
| 5 | 0.718059 | -0.002480 |
| 6 | 0.711881 | -0.006178 |
| 7 | 0.708437 | -0.003444 |
| 8 | 0.702341 | -0.006096 |
| 9 | 0.698222 | -0.004119 |
| 18 | 0.651043 | -0.003601 |
| 26 | 0.506921 | -0.023566 |
| 35 | 0.332395 | -0.032218 |
| 46 | 0.189352 | -0.017314 |
| 59 | 0.103634 | -0.003077 |
| 81 | 0.045390 | -0.001606 |
| 103 | 0.031783 | -0.003772 |
| 169 | 0.016416 | -0.000745 |
| 170 | 0.015409 | -0.001007 |
| 172 | 0.015158 | -0.000251 |
| 180 | 0.013274 | -0.001884 |
| 181 | 0.012339 | -0.000935 |
| 201 | 0.011957 | -0.000382 |
| 204 | 0.010142 | -0.001815 |
| 205 | 0.009565 | -0.000577 |

## Online tournament (1 trial = 1 epoch-level update)

- Train MSE: `0.319128` -> best `0.064972` (+79.64%)
- Trials: `40` | commits `20` | rejected `20`
- Mean best-MSE drop per trial: `0.006354`
- New-best advancements: `20` over `40` trials (0.500 advances/trial)

### Trials to train-MSE convergence levels

| Level (train MSE) | First trial with best<=level |
| ---: | ---: |
| <= 1.00 | 0 |
| <= 0.50 | 0 |
| <= 0.25 | 1 |
| <= 0.10 | 8 |
| <= 0.05 | never |
| <= 0.02 | never |
| <= 0.01 | never |

### Convergence per trial band (5-trial levels)

| Trials | Start best MSE | End best MSE | Band drop | Drop / trial |
| :--- | ---: | ---: | ---: | ---: |
| 1-5 | 0.155006 | 0.141918 | +0.013087 | +0.002617 |
| 6-10 | 0.138949 | 0.087381 | +0.051567 | +0.010313 |
| 11-15 | 0.084845 | 0.070217 | +0.014628 | +0.002926 |
| 16-20 | 0.070217 | 0.069878 | +0.000339 | +0.000068 |
| 21-25 | 0.069878 | 0.069744 | +0.000134 | +0.000027 |
| 26-30 | 0.069744 | 0.069744 | +0.000000 | +0.000000 |
| 31-35 | 0.069744 | 0.067480 | +0.002264 | +0.000453 |
| 36-40 | 0.065393 | 0.064972 | +0.000421 | +0.000084 |

### Condensed trial log (commits + checkpoints)

| Trial | Slice | Status | Loss | Best | Delta vs prior best |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 1 | `l1` | `committed_natively` | 0.155006 | 0.155006 | -0.164094 |
| 2 | `l2@r0c0` | `committed_natively` | 0.151235 | 0.151235 | -0.003765 |
| 3 | `l2@r0c1` | `committed_natively` | 0.147646 | 0.147646 | -0.003554 |
| 4 | `l2@r0c2` | `committed_natively` | 0.147329 | 0.147329 | -0.000271 |
| 5 | `l2@r0c3` | `committed_natively` | 0.141918 | 0.141918 | -0.005382 |
| 6 | `l2@r0c4` | `committed_natively` | 0.138949 | 0.138949 | -0.002951 |
| 7 | `l2@r1c0` | `committed_natively` | 0.133787 | 0.133787 | -0.005113 |
| 8 | `l2@r1c1` | `committed_natively` | 0.087523 | 0.087523 | -0.046277 |
| 9 | `l2@r1c2` | `committed_natively` | 0.087381 | 0.087381 | -0.000119 |
| 10 | `l2@r1c3` | `rejected` | 0.093421 | 0.087381 | +0.006021 |
| 11 | `l2@r1c4` | `committed_natively` | 0.084845 | 0.084845 | -0.002555 |
| 12 | `l2@r2c0` | `committed_natively` | 0.078618 | 0.078618 | -0.006182 |
| 13 | `l2@r2c1` | `committed_natively` | 0.075286 | 0.075286 | -0.003314 |
| 14 | `l2@r2c2` | `committed_natively` | 0.070526 | 0.070526 | -0.004774 |
| 15 | `l2@r2c3` | `committed_natively` | 0.070217 | 0.070217 | -0.000283 |
| 18 | `l2@r3c1` | `committed_natively` | 0.069985 | 0.069985 | -0.000215 |
| 19 | `l2@r3c2` | `committed_natively` | 0.069878 | 0.069878 | -0.000122 |
| 20 | `l2@r3c3` | `rejected` | 0.069855 | 0.069878 | -0.000045 |
| 25 | `l2@r4c3` | `committed_natively` | 0.069744 | 0.069744 | -0.000156 |
| 30 | `l3@r0c3` | `rejected` | 0.069604 | 0.069744 | -0.000096 |
| 34 | `l3@r1c2` | `committed_natively` | 0.067480 | 0.067480 | -0.002220 |
| 35 | `l3@r1c3` | `rejected` | 0.072608 | 0.067480 | +0.005108 |
| 36 | `l3@r1c4` | `committed_natively` | 0.065393 | 0.065393 | -0.002107 |
| 40 | `l3@r2c3` | `committed_natively` | 0.064972 | 0.064972 | -0.000428 |

## Head-to-head (epoch-level effort only)

| Convergence level | Adam epochs | Online trials | Winner (fewer steps) |
| ---: | ---: | ---: | :--- |
| <= 1.00 | 1 | 0 | online |
| <= 0.50 | 29 | 0 | online |
| <= 0.25 | 42 | 1 | online |
| <= 0.10 | 60 | 8 | online |
| <= 0.05 | 74 | never | adam |
| <= 0.02 | 151 | never | adam |
| <= 0.01 | 205 | never | adam |

## Convergence rate (per epoch-level step)

- Adam mean drop / epoch: `0.003548` over `205` epochs
- Online mean drop / trial: `0.006354` over `40` trials
- Advances / epoch (Adam): `0.400`
- Advances / trial (Online): `0.500`

## Verdict (epoch/trial units)

- Adam reached best train MSE `0.009565` in `205` epochs.
- Online reached best train MSE `0.064972` in `40` trials.
- Adam still converges further (to ~0.01); online plateaued near ~0.065 in this run.

