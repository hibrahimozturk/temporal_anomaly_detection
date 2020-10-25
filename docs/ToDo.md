### TODO
* [ ] First stages of I3D MS-TCN will be classification stages, consecutive stages will be regression  
* [ ] MS-TCN with TSM features will be trained with doubled stage as first stage
* [x] In order to fix some false annotations and add missing videos,
 annotation files will be edited 
    * [x] Missing fighting videos will be added
    * [x] Wrong stealing video names will be fixed  
* [x] MSTCN will be trained with TSM features
* [x] TSM feature extractor will be implemented
    * [ ] Features will be extracted with different number of crops, 1 and 3 
* [x] Models will be trained with normal and abnormal videos
    * [ ] During validation empty clips will not be computed, directly classified as normal
* [ ] There is not background category in categories during feature extraction, it will be added
