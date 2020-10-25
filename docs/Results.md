### Experimental Results

* Weakly Supervised github model: AUC Score 0.5646
* 11_06_2019_complete_train: AUC Score 1.epoch sonrası 0.6328

F1@IOU-Threshold
#### Abnormal Videos Abnormal Segmentation Results on Validation Set

IOU of abnormal segments are evaluated on abnormal videos

| Experiments       | AUC       | F1@0.1-0.5  | F1@0.1-0.75 | F1@0.1-0.9 |
|-------------------|-----------|-------------|-------------|------------|
| Weakly Supervised*|   0.57    |  5.10       |   2.89      |   1.86     |
|      MLP          |   0.55    |  14.77      |   10.79     |   9.54     |
| TCN W20-K11-S2    |   0.61    |  41.86      |   32.34     |   21.15    |
| TCN W52-K21-S2    | **0.64**  |  41.38      |   32.77     |   27.27    |
| TCN W52-K21-S2    |   0.63    |  47.85      |   31.00     |   20.10    |
| TCN W52-K21-S2    |   0.57    |  50.21      |   35.60     |   30.85    |
| MSTCN W52-L3-S3(1)|    -      |  46.79      |     -       |     -      |
|MSTCN W52-S10-L5(5)|   0.51    |  51.30      |   41.26     |   28.09    |
|MSTCN W52-S10-L5(6)|   0.61    |  58.10      |   51.52     | **41.71**  |
|MSTCN W96-S10-L5(7)|   0.62    |**75.90**    | **62.30**   |   26.63    |


* TCN   W: Window, S: # of sub windows, K: Kernel size 
* MSTCN W: Window, S: # of stages, L: # of layers 


#### Abnormal Videos Abnormal Segmentation Results on Validation Set

IOU of abnormal and normal segments are evaluated on abnormal videos


| Experiments       | AUC       | F1@0.1-0.5  | F1@0.25-0.5 | F1@0.5-0.5 |
|-------------------|-----------|-------------|-------------|------------|
| Weakly Supervised*|   0.57    |   5.10      |   2.04      |   0.00     |
|MSTCN W96-S10-L5(8)|   0.58    |  38.05      |  33.17      |  18.53     |


#### Abnormal Videos Normal and Abnormal Segmentation Results on Validation Set

IOU of abnormal and normal segments are evaluated on abnormal videos


| Experiments       | AUC       | F1@0.1-0.5  | F1@0.25-0.5 | F1@0.5-0.5 |
|-------------------|-----------|-------------|-------------|------------|
| Weakly Supervised*|   0.57    |  36.66      |  28.52      |  18.15     |
|MSTCN W96-S10-L5(8)|   0.58    |  49.82      |  40.00      |  23.15     |



#### Abnormal and Normal Videos Normal and Abnormal Segmentation Results on Validation Set

* IOU of abnormal and normal segments are evaluated on abnormal and normal videos
* First 43 normal videos have been included
* 399 abnormal windows, 445 normal windows when window size equals to 96

| Experiments       | AUC       | F1@0.1-0.5  | F1@0.25-0.5 | F1@0.5-0.5 |
|-------------------|-----------|-------------|-------------|------------|
| Weakly Supervised*|   0.79    |  39.55      |  33.42      |   25.34    |
|MSTCN W96-S10-L5(8)|   0.69    |  54.01      |  47.98      |   33.84    |


#### Test Set Results
| Experiments   | AUC           | F1@0.1-0.5  | F1@0.1-0.75 | F1@0.1-0.9 |
| ------------- | ------------- |    -------- |       ----  |       ---- |
| Weakly Supervised | 0.518     | 9.0452      | 8.5937      | 3.5503     |