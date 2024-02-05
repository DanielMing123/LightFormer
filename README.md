# LightFormer
LightFormer: An End-to-End Model for Intersection Right-of-Way Recognition Using Traffic Light Signals and an Attention Mechanism [[Paper](https://arxiv.org/pdf/2307.07196.pdf)]
![image](https://github.com/DanielMing123/LightFormer/blob/main/imgs/LightFormer.png)
# Abstract
For smart vehicles driving through signalised intersections, it is crucial to determine whether the vehicle has right-of-way given the state of the traffic lights. To address this issue, camera-based sensors can be used to determine whether the vehicle has permission to proceed straight, turn left or turn right. To the best of our knowledge, the current research in this domain primarily focuses on traffic light detection and recognition based on object detection algorithms, and there is no end-to-end approach to solve this problem. Thus, in this paper, we propose a novel end-to-end intersection right-of-way recognition model called LightFormer to generate right-of-way status for available driving directions in complex urban intersections. The model includes a spatial-temporal inner structure with an attention mechanism, which incorporates features from past image to contribute to the classification of the current frame's right-of-way status. In addition, a modified, multi-weight arcface loss is introduced to enhance the model's classification performance. Finally, the proposed LightFormer is trained and tested on two public traffic light datasets with manually augmented labels to demonstrate its effectiveness.
# Overall Architecture
![image](https://github.com/DanielMing123/LightFormer/blob/main/imgs/crop_LightFormer_page-0001.jpg)
# Encoder Layer Inner Structure
![image](https://github.com/DanielMing123/LightFormer/blob/main/imgs/crop_Encoder_Layer_page-0001.jpg)
# Installation
* CUDA>=11.7
* python>=3.8
* pytorch>=1.13.0
* pytorch-lightning>=1.6.1
*  torchvision>=0.14.0
*  mmcv-full>=1.7.0
# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{ming2023lightformer,
  title={LightFormer: An End-to-End Model for Intersection Right-of-Way Recognition Using Traffic Light Signals and an Attention Mechanism},
  author={Ming, Zhenxing and Berrio, Julie Stephany and Shan, Mao and Nebot, Eduardo and Worrall, Stewart},
  journal={arXiv preprint arXiv:2307.07196},
  year={2023}
}
```
