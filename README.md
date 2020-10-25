## Violance Detection

* [Related Words](docs/RelatedWorks.md)
* [Method](docs/Method.md)
* [Logs](#logs)
* [Dataset](docs/Dataset.md)
* [ToDo](docs/ToDo.md)
* [Experimental ToDo](docs/ToDo.md#experimental-todo)
* [Experiment Results](docs/Results.md)
* [Paper List](docs/RelatedWorks.md#paper-list)
* [References](docs/RelatedWorks.md#references)


### Logs

* (03/02/2020) feature extraction is more efficient and stable.
* Molotov, posters and fighting videos are gathered in the Workspace / Dataset / UCF-Crime folder.
* Name of banner category should be protestation.
* Videos from molotov and banner categories have been added to Anomaly_Test.txt .
* There is not pytorch implementation of new action recognition architectures such as csn[3], r2plus1d[6].
    * Therefore we use i3d-rgb[5] for feature extraction.
* Selected videos from molotov bomb and banner categories have been added to Anomaly_Detection_splits/Anomaly_Test.txt file.
* (14/12/2019) There are 2141 temporal annotations, but total number of videos are 2139.
    * Number of videos have been checked, there are some multiple annotations and missing annotations.
* Training, test and validation set has been created.
    * There are only abnormal videos in created sets. 
* (22/12/2019) Features can be extracted from I3D network
* Global average pooling layer has been appended to I3D network to reduce feature size to 1024.
* Temporal stride is 16.