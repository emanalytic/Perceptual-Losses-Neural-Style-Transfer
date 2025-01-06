# Neural Style Transfer with Perceptual Losses

This project is an implementation of the paper ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/pdf/1603.08155) by Johnson et al. The main idea is to train a feed forward neural network for style transfer using perceptual losses, enabling real-time style transfer. 

Neural style transfer was initially introduced in the paper by Gatys et al., which used an optimization based approach to generate stylized images. While this method produces high-quality results, it is computationally expensive and slow.
Pytorch has a tutorial for implementing the Gatys' Optimzation Based method, check this out [Pytorch Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). 

## Implementation 




For a detailed understanding of the underlying concepts, check out my [notebook](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/notebooks/experimentation_v02.ipynb) included in this repository.

---

## How to Use

### 1. Train you own Model
- Download the COCO dataset from [this link](https://cocodataset.org/#download) and place it in the `data/coco` directory.

- Use the `train.py` script to train the style transfer model.
- Run the following command:
  ```bash
  python train.py 
  ```
(!): If you don't have gpus to train this you can utilize kaggle gpus. ;)

### 4. Use the Pretrained Model
- If you don't want to train the model yourself, you can use the pretrained model available in this repository. Download it from [this link](#). 
---

## Results

Below are some example results of the style transfer model:

| Content Image | Style Image | Stylized Image |
|---------------|-------------|----------------|
| ![Content](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/content_dir/test_image_01.png) | ![style](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/Vincent_van_Gogh.png) | ![stylized](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/styled_image.png) |

---

## Files in the Repository

- `train.py`: Script to train the style transfer model.
- `vgg.py`: Script for hook manager to get features
- `model.py`: Model Implementation from pytorch example repo
- `loss.py` : Perceptual Loss Implementation
- `experimentation_v02.ipynb`: Jupyter notebook explaining the concepts and code in detail.
- `data/`: Directory for storing the content images and style image.

---

## References

- ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/pdf/1603.08155)
- [Gatys et al.'s Neural Style Transfer](https://arxiv.org/pdf/1508.06576)
- [PyTorch Neural Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [Pytorch Example](https://github.com/pytorch/examples)
- [Reference Repository](https://github.com/francislata/Perceptual-Losses/tree/master)
- [Helpful Article](https://sh-tsang.medium.com/brief-review-perceptual-losses-for-real-time-style-transfer-and-super-resolution-ac4fd2658b8)
---
:)
