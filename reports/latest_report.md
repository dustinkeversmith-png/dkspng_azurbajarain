# tournament shootout

Written 20261011T225205Z UTC

========================================================================================
TOURNAMENT BENCHMARK SHOOTOUT RESULTS (10 TRIALS PER ARM)
========================================================================================
Arm Name             L_init  L_final      Delta L (%)   Commits   Diverged   Win (c1/c2/c3)   Avg Tokens
----------------------------------------------------------------------------------------
MLP-Standard         1.6997   0.4065 +1.2932 (+76.1%)   4/10       1/10  (1 / 2 / 1)       874.5
MLP-Aggressive       1.6997   1.6592 +0.0405 ( +2.4%)   2/10       0/10  (0 / 0 / 2)       983.7
AE-Standard          3.2818   0.8248 +2.4570 (+74.9%)   6/10       0/10  (5 / 0 / 1)      1022.7
AE-TightBounds       3.2818   2.9467 +0.3351 (+10.2%)   1/10       0/10  (1 / 0 / 0)       928.5
========================================================================================

PER-TRIAL TOKENS (uncapped completion)
Arm                Trial  Prompt  Completion   Total  Status               Winner  Loss
MLP-Standard           1     581         276     857  committed_natively   c1      1.6981
MLP-Standard           2     599         263     862  rejected             c1      1.6981
MLP-Standard           3     628         279     907  committed_natively   c2      1.6741
MLP-Standard           4     655         268     923  rejected             c1      1.6741
MLP-Standard           5     606         204     810  committed_natively   c3      0.6872
MLP-Standard           6     630         252     882  rejected_divergent   none    101.0000
MLP-Standard           7     629         250     879  rejected             c1      0.6872
MLP-Standard           8     631         264     895  committed_natively   c2      0.4065
MLP-Standard           9     620         256     876  rejected             c1      0.4065
MLP-Standard          10     625         229     854  rejected             c1      0.7774
MLP-Standard       TOTAL    8745  avg 874.5
MLP-Aggressive         1     581         279     860  rejected             c3      1.7068
MLP-Aggressive         2     581         255     836  rejected             c1      1.6997
MLP-Aggressive         3     629         271     900  committed_natively   c3      1.6665
MLP-Aggressive         4     656         255     911  rejected             c1      1.6665
MLP-Aggressive         5     624         209     833  rejected             c1      1.8745
MLP-Aggressive         6     638         283     921  committed_natively   c3      1.6592
MLP-Aggressive         7     656         257     913  rejected             c1      1.6592
MLP-Aggressive         8    1358         557    1915  rejected             c1      1.6670
MLP-Aggressive         9     637         262     899  rejected             c1      1.6592
MLP-Aggressive        10     625         224     849  rejected             c1      1.7552
MLP-Aggressive     TOTAL    9837  avg 983.7
AE-Standard            1     578         273     851  committed_natively   c1      2.9171
AE-Standard            2     591         314     905  committed_natively   c1      2.3782
AE-Standard            3     595         247     842  rejected             c1      2.3782
AE-Standard            4    1390         648    2038  committed_natively   c3      1.6976
AE-Standard            5     623         243     866  rejected             c1      1.6976
AE-Standard            6     675         316     991  committed_natively   c1      1.2267
AE-Standard            7     623         268     891  rejected             c1      1.2267
AE-Standard            8     654         333     987  committed_natively   c1      0.9359
AE-Standard            9     624         248     872  rejected             c1      0.9359
AE-Standard           10     657         327     984  committed_natively   c1      0.8248
AE-Standard        TOTAL   10227  avg 1022.7
AE-TightBounds         1     578         261     839  committed_natively   c1      2.9467
AE-TightBounds         2     593         319     912  rejected             c2      3.0897
AE-TightBounds         3     607         261     868  rejected             c1      3.2788
AE-TightBounds         4     660         337     997  rejected             c1      3.0640
AE-TightBounds         5     637         258     895  rejected             c1      3.3253
AE-TightBounds         6     662         340    1002  rejected             c1      2.9887
AE-TightBounds         7     639         260     899  rejected             c1      3.2816
AE-TightBounds         8     663         324     987  rejected             c1      2.9773
AE-TightBounds         9     638         255     893  rejected             c1      3.2286
AE-TightBounds        10     662         331     993  rejected             c1      3.0106
AE-TightBounds     TOTAL    9285  avg 928.5
