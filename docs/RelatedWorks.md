### Related Works

* I3D [5]
    * SGD optimizer
* **Enhancing Temporal Action Localization with Transfer Learning from ActionRecognition** [7]
    * Temporal smoothing is applied to consecutive frames.
    * Output layer of I3D is trained with smoothed features which extracted from small dataset.
    * A visual bag of words is created. It comprises 4000 components.
        * A feature map kernel is applied to provide non-linearity.
    *  Temporal convolution network (TCN[8]) is applied at the end.
        * 20 temporal 1D convolution layers with kernel size 3
    * Without BOW results (output retraining + TCN) are hopeful. 
    * Source code does not exists.
* **Temporal convolutional networks for action segmentation and detection** [8]
   * Input to Temporal Convolutional Networks(TCN) is T spatiotemporal or temporal features from consecutive T frames.
   * Where F is number of filters, F 1D convolution is applied to T Spatiotemporal Features which has F components.
      * Input features has 128 dimensions.
   * Dimension of an output and input is same. 
   * Maxpool is applied to decrease T features to T/2 features.
   * There are two architectures, Encoder Decoder TCN (ED-TCN) and Dilated TCN.
      * ED-TCN is more successful than Dilated TCN.
         * F1 mAP-10 scores are 76.5, 68.0 of ED-TCN and Dilated TCN respetively.
   * In decoder upsampling is applied by simply repeating each entry twice.
   * Output of decoder is passed to classifier.
      * Categorical cross entropy loss with SGD and Adam is applied.
   * There are 3 layers in decoder and encoder parts.

### Paper List
* [ ] [Farha, Yazan Abu, and Jurgen Gall. "Ms-tcn: Multi-stage temporal convolutional network for action segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.](https://arxiv.org/pdf/1903.01945.pdf)

### References

1. [Sultani, Waqas, Chen Chen, and Mubarak Shah. "Real-world anomaly detection in surveillance videos." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf)
    1. * [AnomalyDetectionCVPR2018](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)
2. [Ghadiyaram, Deepti, Du Tran, and Dhruv Mahajan. "Large-scale weakly-supervised pre-training for video action recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghadiyaram_Large-Scale_Weakly-Supervised_Pre-Training_for_Video_Action_Recognition_CVPR_2019_paper.pdf)
3. [Tran, Du, et al. "Video Classification with Channel-Separated Convolutional Networks." arXiv preprint arXiv:1904.02811 (2019).](https://arxiv.org/pdf/1904.02811.pdf)
4. [Girdhar, Rohit, et al. "Video action transformer network." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.](http://openaccess.thecvf.com/content_CVPR_2019/papers/Girdhar_Video_Action_Transformer_Network_CVPR_2019_paper.pdf)
5. [Carreira, Joao, and Andrew Zisserman. "Quo vadis, action recognition? a new model and the kinetics dataset." proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf)
6. [Tran, Du, et al. "A closer look at spatiotemporal convolutions for action recognition." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018.](https://arxiv.org/pdf/1711.11248.pdf)
7. [Iqbal, Ahsan, Alexander Richard, and Juergen Gall. "Enhancing Temporal Action Localization with Transfer Learning from Action Recognition." Proceedings of the IEEE International Conference on Computer Vision Workshops. 2019.](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CoView/Iqbal_Enhancing_Temporal_Action_Localization_with_Transfer_Learning_from_Action_Recognition_ICCVW_2019_paper.pdf)
8. [Lea, Colin, et al. "Temporal convolutional networks for action segmentation and detection." proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf)
9. [Wang, Xiaolong, et al. "Non-local neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.](https://arxiv.org/pdf/1711.07971.pdf)
10. [Lin, Ji, Chuang Gan, and Song Han. "Tsm: Temporal shift module for efficient video understanding." Proceedings of the IEEE International Conference on Computer Vision. 2019.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf)