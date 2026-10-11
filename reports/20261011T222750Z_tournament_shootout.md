# tournament shootout

Written 20261011T222750Z UTC

========================================================================================
TOURNAMENT BENCHMARK SHOOTOUT RESULTS (10 TRIALS PER ARM)
========================================================================================
Arm Name             L_init  L_final      Delta L (%)   Commits   Diverged   Win (c1/c2/c3)   Avg Tokens
----------------------------------------------------------------------------------------
MLP-Standard         1.6997   0.8002 +0.8995 (+52.9%)   3/10       1/10  (1 / 0 / 2)       793.2
MLP-Aggressive       1.6997   0.6180 +1.0817 (+63.6%)   6/10       1/10  (6 / 0 / 0)       804.9
AE-Standard          3.2818   3.2818 +0.0000 ( +0.0%)   0/10       5/10  (0 / 0 / 0)       996.3
AE-TightBounds       3.2818   3.2818 +0.0000 ( +0.0%)   0/10       5/10  (0 / 0 / 0)       992.8
========================================================================================

PER-TRIAL TOKENS (uncapped completion)
Arm                Trial  Prompt  Completion   Total  Status               Winner  Loss
MLP-Standard           1     495         284     779  committed_natively   c3      1.6843
MLP-Standard           2     521         260     781  rejected             c1      1.6843
MLP-Standard           3     527         280     807  rejected_divergent   none    101.0000
MLP-Standard           4     554         258     812  rejected             c1      1.6843
MLP-Standard           5     523         231     754  committed_natively   c1      1.1991
MLP-Standard           6     546         267     813  rejected             c1      1.1991
MLP-Standard           7     564         253     817  rejected             c1      1.1991
MLP-Standard           8     562         249     811  rejected             c1      1.1991
MLP-Standard           9     564         258     822  rejected             c1      1.1991
MLP-Standard          10     531         205     736  committed_natively   c3      0.8002
MLP-Standard       TOTAL    7932  avg 793.2
MLP-Aggressive         1     495         268     763  rejected_divergent   none    101.0000
MLP-Aggressive         2     497         264     761  committed_natively   c1      1.6997
MLP-Aggressive         3     534         309     843  rejected             c1      1.7230
MLP-Aggressive         4     554         262     816  committed_natively   c1      1.6997
MLP-Aggressive         5     546         216     762  committed_natively   c1      0.7346
MLP-Aggressive         6     547         279     826  rejected             c3      0.8066
MLP-Aggressive         7     562         259     821  committed_natively   c1      0.7346
MLP-Aggressive         8     571         296     867  rejected             c1      2.7033
MLP-Aggressive         9     563         261     824  committed_natively   c1      0.7346
MLP-Aggressive        10     545         221     766  committed_natively   c1      0.6180
MLP-Aggressive     TOTAL    8049  avg 804.9
AE-Standard            1    1018         524    1542  rejected_divergent   none    101.0000
AE-Standard            2     507         320     827  rejected             c1      4.7471
AE-Standard            3     512         262     774  rejected             c1      5.1509
AE-Standard            4     565         321     886  rejected             c1      3.3918
AE-Standard            5     550         257     807  rejected_divergent   none    101.0000
AE-Standard            6     565         313     878  rejected_divergent   none    101.0000
AE-Standard            7     532         272     804  rejected_divergent   none    101.0000
AE-Standard            8    1136         620    1756  rejected             c2      5.2229
AE-Standard            9     532         268     800  rejected_divergent   none    101.0000
AE-Standard           10     556         333     889  rejected             c1      3.6294
AE-Standard        TOTAL    9963  avg 996.3
AE-TightBounds         1    1018         494    1512  rejected_divergent   none    101.0000
AE-TightBounds         2    1056         655    1711  rejected             c1      4.6813
AE-TightBounds         3     512         257     769  rejected_divergent   none    101.0000
AE-TightBounds         4     556         335     891  rejected             c1      4.2705
AE-TightBounds         5     541         255     796  rejected_divergent   none    101.0000
AE-TightBounds         6     556         331     887  rejected             c1      3.9610
AE-TightBounds         7     541         261     802  rejected_divergent   none    101.0000
AE-TightBounds         8     556         324     880  rejected             c1      3.8735
AE-TightBounds         9     541         256     797  rejected_divergent   none    101.0000
AE-TightBounds        10     556         327     883  rejected             c2      4.6397
AE-TightBounds     TOTAL    9928  avg 992.8
