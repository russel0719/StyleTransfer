# Season Style Transfer
 Creating season changing video using Style Transfer

## Abstract
&nbsp;The recent Adobe Photoshop 2022 version of the landscape photo mixer filter in the new feature, neural filter provides the ability to transform images to suit predefined themes. Similarly, this project aims to change the image to a specific season style. To this end, the tree that best reflects the characteristics of the season was selected as Style Data, and the VGG pretrained network and Transformer network were used to learn to convert the season of the image into a style corresponding to style data. In addition, data processing and learning were also conducted to implement image transformation in the middle of the two seasons.

&nbsp;As a result of the learning, it was found that the uniformity of the wooden images selected by style data was a condition of stable performance, and the method of mixing the intermediate style conversion feature of the two seasons obtained natural results.

## Usage
1. DownLoad COCO Dataset

  download coco 2017 val dataset in http://images.cocodataset.org/zips/val2017.zip
  move val2017 folder to /data/COCO/

2. Add style images

  add Season Style Images to /data/ in format 'spring, summer, fall, winter'

3. Add test image

  add test image to /data/test folder

## Reference
  [1]	Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2414–2423, 2016. 5
  
  [2]	Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super resolution. In European conference on computer vision, pages 694–711. Springer, 2016. 5
  
  https://github.com/hoya012/fast-style-transfer-tutorial-pytorch/blob/master/Fast-Style-Transfer-PyTorch.ipynb
