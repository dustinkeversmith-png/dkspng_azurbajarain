# tournament shootout

Written 20261011T221713Z UTC

========================================================================================
TOURNAMENT BENCHMARK SHOOTOUT RESULTS (10 TRIALS PER ARM)
========================================================================================
Arm Name             L_init  L_final      Delta L (%)   Commits   Diverged   Win (c1/c2/c3)   Avg Tokens
----------------------------------------------------------------------------------------
MLP-Standard         1.6997   0.5158 +1.1839 (+69.7%)   4/10       4/10  (0 / 2 / 2)       768.5
MLP-Aggressive       1.6997   0.7865 +0.9132 (+53.7%)   5/10       2/10  (2 / 1 / 2)       840.7
AE-Standard          3.2818   3.2818 +0.0000 ( +0.0%)   0/10      10/10  (0 / 0 / 0)       725.1
AE-TightBounds       3.2818   2.7341 +0.5478 (+16.7%)   4/10       5/10  (1 / 0 / 3)       738.4
========================================================================================

PER-TRIAL TOKENS (uncapped completion)
Arm                Trial  Prompt  Completion   Total  Status               Winner  Loss
MLP-Standard           1     450         274     724  committed_natively   c2      1.6948
MLP-Standard           2     476         273     749  committed_natively   c3      1.6875
MLP-Standard           3     497         286     783  committed_natively   c2      1.5001
MLP-Standard           4     526         315     841  rejected_divergent   none    101.0000
MLP-Standard           5     491         227     718  rejected             c1      1.7463
MLP-Standard           6     519         263     782  committed_natively   c3      0.5158
MLP-Standard           7     519         281     800  rejected_divergent   none    101.0000
MLP-Standard           8     519         273     792  rejected_divergent   none    101.0000
MLP-Standard           9     510         283     793  rejected             c1      1.0807
MLP-Standard          10     484         219     703  rejected_divergent   none    101.0000
MLP-Standard       TOTAL    7685  avg 768.5
MLP-Aggressive         1     450         265     715  rejected_divergent   none    101.0000
MLP-Aggressive         2     468         266     734  committed_natively   c3      1.6997
MLP-Aggressive         3     489         282     771  committed_natively   c3      1.5122
MLP-Aggressive         4     524         281     805  committed_natively   c1      1.5122
MLP-Aggressive         5     499         218     717  rejected             c1      1.7316
MLP-Aggressive         6     526         266     792  rejected             c1      11.9339
MLP-Aggressive         7     534         262     796  rejected_divergent   none    101.0000
MLP-Aggressive         8     520         279     799  rejected             c3      1.6457
MLP-Aggressive         9     526         276     802  committed_natively   c2      1.5122
MLP-Aggressive        10    1026         450    1476  committed_natively   c1      0.7865
MLP-Aggressive     TOTAL    8407  avg 840.7
AE-Standard            1     436         247     683  rejected_divergent   none    101.0000
AE-Standard            2     447         258     705  rejected_divergent   none    101.0000
AE-Standard            3     475         246     721  rejected_divergent   none    101.0000
AE-Standard            4     495         255     750  rejected_divergent   none    101.0000
AE-Standard            5     487         238     725  rejected_divergent   none    101.0000
AE-Standard            6     487         227     714  rejected_divergent   none    101.0000
AE-Standard            7     495         244     739  rejected_divergent   none    101.0000
AE-Standard            8     495         262     757  rejected_divergent   none    101.0000
AE-Standard            9     487         240     727  rejected_divergent   none    101.0000
AE-Standard           10     487         243     730  rejected_divergent   none    101.0000
AE-Standard        TOTAL    7251  avg 725.1
AE-TightBounds         1     436         240     676  rejected_divergent   none    101.0000
AE-TightBounds         2     447         238     685  rejected             c1      15.8002
AE-TightBounds         3     484         261     745  rejected_divergent   none    101.0000
AE-TightBounds         4     504         258     762  rejected_divergent   none    101.0000
AE-TightBounds         5     496         242     738  committed_natively   c3      2.9171
AE-TightBounds         6     499         229     728  committed_natively   c1      2.9171
AE-TightBounds         7     515         263     778  committed_natively   c3      2.7765
AE-TightBounds         8     523         245     768  committed_natively   c3      2.7341
AE-TightBounds         9     515         244     759  rejected_divergent   none    101.0000
AE-TightBounds        10     507         238     745  rejected_divergent   none    101.0000
AE-TightBounds     TOTAL    7384  avg 738.4
