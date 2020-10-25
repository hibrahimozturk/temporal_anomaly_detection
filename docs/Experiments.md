### Experiments

* F1@25-50 scores measured stage by stage in I3D - MS-TCN experiments on test set
    * 10. experiment has 64 window size 5 stage and 6 layers
    * 24. experiment has different loss which aims to increase distance between closest normal and abnormal clips
    * 36. experiment has 14 layers in first stage and 6 layers in consecutive stages with mse loss 
    * 4. experiment has 128 window size 5 stage and 6 layers trained with TSM features instead of I3D
    * 5. experiment has 128 window size 5 stage and 6 layers trained with TSM features and repeat at first stage

| Stages | 36    | 24    | 10    | 4     | 5     |
|--------|-------|-------|-------|-------|-------|
| 1      | 41.23 | 23.43 | 24.98 | 11.67 | 28.44 |
| 2      | 45.52 | 37.46 | 44.86 | 35.98 | 37.34 |
| 3      | 47.26 | 42.89 | 47.35 | 38.78 | 40.67 |
| 4      | 48.16 | 45.91 | 48.66 | 41.85 | 42.84 | 
| 5      |       |       |       | 42.83 | 43.07 |


* Doubled success at first stage does not affect final success a lot. There is an up limit at F1@25-50 score. 


* F1@25-50 scores on test set

| Experiments                    | F1@10-50 | F1@25-50  | F1@50-50 | AUC       |
|--------------------------------|----------|-----------|----------|-----------|
| MSTCN/10-w64-s5-l6             | 58.89    | **50.75** | 34.69    | **67.63** |
| TSM-MSTCN/4-w64-channelsize128 | 53.76    | 42.83     | 26.34    |           |
| TSM-MSTCN/5-w64-ssrepeat       | 53.62    | 43.07     | 26.81    | 66.85     |


* Doubling channel size, 64 to 128, does not affect results on validation set.
64 channel size is enough, number of parameters are saturated.
