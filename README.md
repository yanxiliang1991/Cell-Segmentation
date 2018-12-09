
Deep Learning Based Cell Segmentation
===

This is an example cell segmentation procedure in histopathological images based on convolutional neural networks.

Unlike traditional deep learning based approaches which use 2-classes ("background" and "cell"), we use 4-classes ("background", "cell center", "cell innerboundary" and "cell outerboundary") as inputs to train the model. In this way, we can delineate nucleus boundaries more accurately. Following figure demonstrates a training patch extraction procedure for 4-class model.

<p align="center">
  <img src="./patchExtraction.png" width="50%" height="50%"/>
</p>

[Caffe](http://caffe.berkeleyvision.org/) framework and Matlab is used for the implementation of the project

***

In the following figure, sample test image is shown on the left. Estimated class labels at the output of the model is shown in the middle. In this class labels, white, blue, red and green regions represent background, cell center, cell inner boundary and cell outer boundary classes, respectively. Figure at the right shows final segmented cells after applying region growing.
<p align="center">
  <img src="./sampleSegmentation.png" width="80%" height="80%"/>
</p>

Prerequisites
-------------
Matlab, Caffe, Matcaffe

Project is implemented on Ubuntu 16.04 machine with Matlab R2014a installed

Contact
-------
E-mail: deniz.mail@gmail.com

References
------------
- http://caffe.berkeleyvision.org/
