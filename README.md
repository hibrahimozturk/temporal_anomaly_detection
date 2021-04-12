## ADNet: Temporal Anomaly Detection in Surveillance Videos

![alt text](docs/figures/explosion8-2.jpg "Explosion 8 video groundtruth and prediction timelines")

![alt text](docs/figures/robbery102.jpg "Robbery 102 video groundtruth and prediction timelines")


### Dataset

UCF Crime data set consists of 13 anomaly classes.  We have added two different anomaly classes to the data
set, which are ”molotov bomb” and ”protest” classes. We also have added 33 videos to fighting class. In total, we have added 216 videos to the training set, 17
videos to the test set. Training videos of UCF Crime data set are classified in video-level and temporal annotations are not provided for the training set.
 To train models with temporal information, we annotated anomalies of training videos in temporal domain.

Annotations and videos are publicly available at [the link](https://drive.google.com/file/d/1TnzMzk3TiHJHVsJmqQhzJXvNqml4MijB/view?usp=sharing)

![alt text](docs/figures/banner03.gif "Protest class example")

Protest class example


![alt text](docs/figures/molotof004.gif "Molotov bomb class example")

Molotov bomb class example
