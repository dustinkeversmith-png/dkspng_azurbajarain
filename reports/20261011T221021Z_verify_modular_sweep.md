# verify modular sweep

Written 20261011T221021Z UTC

One slice each for MlpRegressor and BottleneckAutoencoder.



## mlp_regressor

- Layer: layer1@r0c0
- Seed loss: 1.6996605939970129
- Seed checksum: 19bd79c1eb824adb
- Active w: `r:1|g:0.002|A:-99 -33 33 99|B:74 74 74 74`
- Active b: `0.1 0.1 0.1 0.1`
- Gradient: A:[+ + + -] B:[+ + + +]
- Attempts: 1
- Billed tokens: 776
- Status: committed_natively
- Winner: c1
- Loss: 1.6950122199053277
- Delta: -0.0046877800946723
- Checksum: c283336dd346dea9

| Candidate | Distance | Status | Loss | w | b |
| --- | --- | --- | --- | --- | --- |
| c1 | 7 | scored | 1.6950122199053277 | `r:1|g:0.025|A:-33 -33 -33 39|B:68 68 68 68` | `0.12 0.12 0.12 0.12` |
| c2 | 8 | scored | 1.6950122199053277 | `r:1|g:0.045|A:-40 -40 -40 46|B:64 64 64 64` | `0.11 0.11 0.11 0.11` |
| c3 | 8 | scored | 1.6950122199053277 | `r:1|g:0.065|A:-50 -50 -50 56|B:60 60 60 60` | `0.13 0.13 0.13 0.13` |

### Raw replies

#### Attempt 1 accepted=True

```json
{
  "reasoning": "Positive probes on A[0..2] and B all increased loss, so we should decrease those coordinates; A[3] decreased loss when perturbed positively so increase it. Gammas chosen per role: small for conservative follower, moderate for momentum, larger for explorer to probe basins.",
  "confidence": 0.78,
  "c1": {
    "w": "r:1|g:0.025|A:-33 -33 -33 39|B:68 68 68 68",
    "b": "0.12 0.12 0.12 0.12"
  },
  "c2": {
    "w": "r:1|g:0.045|A:-40 -40 -40 46|B:64 64 64 64",
    "b": "0.11 0.11 0.11 0.11"
  },
  "c3": {
    "w": "r:1|g:0.065|A:-50 -50 -50 56|B:60 60 60 60",
    "b": "0.13 0.13 0.13 0.13"
  }
}
```


## bottleneck_ae

- Layer: encode@r0c0
- Seed loss: 3.281823669603035
- Seed checksum: b64256ac5f1244a4
- Active w: `r:1|g:0.0025|A:-14 23 60 96|B:99 61`
- Active b: `0.1 0.1`
- Gradient: A:[+ + + +] B:[- +]
- Attempts: 1
- Billed tokens: 723
- Status: rejected_divergent
- Winner: none
- Loss: 101.0
- Delta: 0.0
- Checksum: b64256ac5f1244a4

| Candidate | Distance | Status | Loss | w | b |
| --- | --- | --- | --- | --- | --- |
| c1 | 6 | rejected_divergent | 101.0 | `r:1|g:0.023|A:10 21 54 90|B:101 63` | `0.12 0.12` |
| c2 | 6 | rejected_divergent | 101.0 | `r:1|g:0.042|A:8 19 52 88|B:103 65` | `0.10 0.10` |
| c3 | 6 | rejected_divergent | 101.0 | `r:1|g:0.067|A:6 17 50 86|B:105 67` | `0.14 0.14` |

### Raw replies

#### Attempt 1 accepted=True

```json
{
  "reasoning": "Probe shows A all '+' so increasing A raised loss -> we should decrease those A ints; B shows [- +] so decrease first B and increase second B. Gammas set per role: small for conservative c1, moderate for momentum c2, larger for exploratory c3.",
  "confidence": 0.78,
  "c1": {
    "w": "r:1|g:0.023|A:10 21 54 90|B:101 63",
    "b": "0.12 0.12"
  },
  "c2": {
    "w": "r:1|g:0.042|A:8 19 52 88|B:103 65",
    "b": "0.10 0.10"
  },
  "c3": {
    "w": "r:1|g:0.067|A:6 17 50 86|B:105 67",
    "b": "0.14 0.14"
  }
}
```
