gg### Method

#### Feature Extraction

* I3D[5] accepts consecutive 79 frames without temporal slide in frames. 
After last layer before final layer features are shrinked with global average pooling.
In our implementation 10x7x7 pooling is applied, this means that a huge information loss
is happened at pooling operation.

* TSM[10] accepts 8 frames in a clip, but distance between consecutive frames is (64//8) 8.
 Features are extracted for each frame with resnet50 and passed to fully connected layer
 in TSM. We use features passed to fc layer. There is not intersection between clip
 windows in TSM. In kinetics tests 3 crops are applied to input frames and results
 are averaged to get better results, but it increases computation. We extracted
 features with 1 crop to run model real time. 10 crops are extracted from test videos
 during Kinetics[5] dataset test. We use TSM which consists Non-Local block [9]. 