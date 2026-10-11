# verify modular sweep

Written 20261011T215208Z UTC

One slice each for MlpRegressor and BottleneckAutoencoder.



## mlp_regressor

- Layer: layer1@r0c0
- Seed loss: 1.6996605939970129
- Seed checksum: 19bd79c1eb824adb
- Active w: `r:1|g:0.002|A:-99 -33 33 99|B:74 74 74 74`
- Active b: `0.1 0.1 0.1 0.1`
- Gradient: A:[+ + + -] B:[+ + + +]
- Attempts: 2
- Billed tokens: 1034
- Status: committed_natively
- Winner: c1
- Loss: 1.6961644784150676
- Delta: -0.0035355215849324217
- Checksum: a0ce31999247ee21

| Candidate | Distance | Status | Loss | w | b |
| --- | --- | --- | --- | --- | --- |
| c1 | 8 | scored | 1.6961644784150676 | `r:1|g:0.020|A:-12 -20 15 7|B:16 9 -4 22` | `0.1 0.1 0.1 0.1` |
| c2 | 8 | scored | 1.696943586078521 | `r:1|g:0.030|A:-11 -19 16 8|B:17 10 -3 23` | `0.1 0.1 0.1 0.1` |
| c3 | 8 | scored | 1.6993570707562033 | `r:1|g:0.050|A:5 -7 12 -3|B:-8 14 -20 6` | `0.1 0.1 0.1 0.1` |

### Raw replies

#### Attempt 1 accepted=False
Error: c1 changes 0 ints; need at least 2 and must not copy w

```json
{"c1":{"w":"r:1|g:0.020|A:-99 -33 33 99|B:74 74 74 74","b":"0.1 0.1 0.1 0.1"},"c2":{"w":"r:1|g:0.022|A:-24 -12 12 24|B:76 76 76 76","b":"0.1 0.1 0.1 0.1"},"c3":{"w":"r:1|g:0.055|A:7 -5 15 -11|B:70 78 66 82","b":"0.1 0.1 0.1 0.1"}}
```

#### Attempt 2 accepted=True

```json
{"c1":{"w":"r:1|g:0.020|A:-12 -20 15 7|B:16 9 -4 22","b":"0.1 0.1 0.1 0.1"},"c2":{"w":"r:1|g:0.030|A:-11 -19 16 8|B:17 10 -3 23","b":"0.1 0.1 0.1 0.1"},"c3":{"w":"r:1|g:0.050|A:5 -7 12 -3|B:-8 14 -20 6","b":"0.1 0.1 0.1 0.1"}}
```


## bottleneck_ae

- Layer: encode@r0c0
- Seed loss: 3.281823669603035
- Seed checksum: b64256ac5f1244a4
- Active w: `r:1|g:0.0025|A:-14 23 60 96|B:99 61`
- Active b: `0.1 0.1`
- Gradient: A:[+ + + +] B:[- +]
- Attempts: 2
- Billed tokens: 954
- Status: committed_natively
- Winner: c1
- Loss: 3.2783658708205867
- Delta: -0.0034341291794133255
- Checksum: 6edc21335275c3f5

| Candidate | Distance | Status | Loss | w | b |
| --- | --- | --- | --- | --- | --- |
| c1 | 6 | scored | 3.2783658708205867 | `r:1|g:0.0025|A:-12 21 58 94|B:97 59` | `0.1 0.1` |
| c2 | 4 | scored | 3.4415516339605716 | `r:1|g:0.0050|A:-10 23 60 92|B:95 57` | `0.1 0.1` |
| c3 | 6 | scored | 3.660025368407268 | `r:1|g:0.0070|A:-8 25 62 90|B:93 55` | `0.1 0.1` |

### Raw replies

#### Attempt 1 accepted=False
Error: c1 changes 0 ints; need at least 2 and must not copy w

```json
{"c1":{"w":"r:1|g:0.0025|A:-14 23 60 96|B:99 61","b":"0.1 0.1"},"c2":{"w":"r:1|g:0.0028|A:-13 24 58 94|B:97 63","b":"0.1 0.1"},"c3":{"w":"r:1|g:0.0031|A:-11 22 56 92|B:95 65","b":"0.1 0.1"}}
```

#### Attempt 2 accepted=True

```json
{"c1":{"w":"r:1|g:0.0025|A:-12 21 58 94|B:97 59","b":"0.1 0.1"},"c2":{"w":"r:1|g:0.0050|A:-10 23 60 92|B:95 57","b":"0.1 0.1"},"c3":{"w":"r:1|g:0.0070|A:-8 25 62 90|B:93 55","b":"0.1 0.1"}}
```
