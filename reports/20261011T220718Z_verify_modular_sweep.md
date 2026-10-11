# verify modular sweep

Written 20261011T220718Z UTC

One slice each for MlpRegressor and BottleneckAutoencoder.



## mlp_regressor

- Layer: layer1@r0c0
- Seed loss: 1.6996605939970129
- Seed checksum: 19bd79c1eb824adb
- Active w: `r:1|g:0.002|A:-99 -33 33 99|B:74 74 74 74`
- Active b: `0.1 0.1 0.1 0.1`
- Gradient: A:[+ + + -] B:[+ + + +]
- Attempts: 1
- Billed tokens: 760
- Status: committed_natively
- Winner: c3
- Loss: 1.6950122199053277
- Delta: -0.0046877800946723
- Checksum: 22c33dd35059e42d

| Candidate | Distance | Status | Loss | w | b |
| --- | --- | --- | --- | --- | --- |
| c1 | 8 | scored | 1.697644384685282 | `r:1|g:0.025|A: -3 -3 -3 4|B: -4 -4 -4 -4` | `0.12 0.12 0.12 0.12` |
| c2 | 8 | scored | 1.6993147334339533 | `r:1|g:0.045|A: -4 -4 -4 5|B: -5 -5 -5 -5` | `0.11 0.11 0.11 0.11` |
| c3 | 8 | scored | 1.6950122199053277 | `r:1|g:0.065|A: 5 5 5 -4|B: -6 -6 -6 -6` | `0.13 0.13 0.13 0.13` |

### Raw replies

#### Attempt 1 accepted=True

```json
{
  "reasoning": "Positive perturbations increased loss on most A entries and all B entries, so we should mostly decrease those coordinates; step sizes reflect conservative, momentum-aligned, and exploratory strategies per role.",
  "confidence": 0.72,
  "c1": {
    "w": "r:1|g:0.025|A: -3 -3 -3 4|B: -4 -4 -4 -4",
    "b": "0.12 0.12 0.12 0.12"
  },
  "c2": {
    "w": "r:1|g:0.045|A: -4 -4 -4 5|B: -5 -5 -5 -5",
    "b": "0.11 0.11 0.11 0.11"
  },
  "c3": {
    "w": "r:1|g:0.065|A: 5 5 5 -4|B: -6 -6 -6 -6",
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
- Billed tokens: 722
- Status: rejected_divergent
- Winner: none
- Loss: 101.0
- Delta: 0.0
- Checksum: b64256ac5f1244a4

| Candidate | Distance | Status | Loss | w | b |
| --- | --- | --- | --- | --- | --- |
| c1 | 6 | rejected_divergent | 101.0 | `r:1|g:0.025|A:10 21 54 90|B:105 59` | `0.12 0.10` |
| c2 | 6 | rejected_divergent | 101.0 | `r:1|g:0.045|A:8 19 50 92|B:102 57` | `0.10 0.08` |
| c3 | 6 | rejected_divergent | 101.0 | `r:1|g:0.072|A:6 17 56 86|B:104 63` | `0.14 0.12` |

### Raw replies

#### Attempt 1 accepted=True

```json
{
  "reasoning": "Probe shows A all '+' (positive perturbation increased loss) so A coords should be decreased; B shows [- +] so first B coord should be increased, second decreased. Gammas chosen per role: small for conservative, medium for momentum-aligned, larger for exploratory.",
  "confidence": 0.82,
  "c1": {
    "w": "r:1|g:0.025|A:10 21 54 90|B:105 59",
    "b": "0.12 0.10"
  },
  "c2": {
    "w": "r:1|g:0.045|A:8 19 50 92|B:102 57",
    "b": "0.10 0.08"
  },
  "c3": {
    "w": "r:1|g:0.072|A:6 17 56 86|B:104 63",
    "b": "0.14 0.12"
  }
}
```
