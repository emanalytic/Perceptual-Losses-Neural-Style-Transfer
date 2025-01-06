# Neural Style Transfer with Perceptual Losses

This project is an implementation of the paper ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/pdf/1603.08155) by Johnson et al. The main idea is to train a feed forward neural network for style transfer using perceptual losses, enabling real-time style transfer. 

Neural style transfer was initially introduced in the paper by Gatys et al., which used an optimization based approach to generate stylized images. While this method produces high-quality results, it is computationally expensive and slow.
Pytorch has a tutorial for implementing the Gatys' Optimzation Based method, check this out [Pytorch Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). 

## Overview 

This implementation achieves real-time neural style transfer by:

* Training a feed-forward transformation network to directly generate stylized images
* Using perceptual losses computed from a pre-trained VGG-16 network
* Optimizing both content and style representations simultaneously
* Enabling fast inference with a single forward pass

The model architecture follows Johnson et al.'s design, featuring residual blocks and upsampling layers. The perceptual loss combines:

* Content loss: MSE between feature representations of content and stylized images
* Style loss: MSE between Gram matrices of feature maps

For mathematical details and implementation insights, check out my [notebook](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/notebooks/neural_style_transfer.ipynb) included in this repository.

---
## Installation 

```bash
git clone https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer.git
cd Perceptual-Losses-Neural-Style-Transfer
pip install -r requirements.txt
```

## Hardware and GPU Utilization
For this project, I used an NVIDIA GeForce GTX 1650 GPU with 4GB VRAM to train on a smaller patch of about 40K images. However, the full training was done on 82K images from the COCO dataset.
If you don’t have a GPU, you can use Kaggle’s Tesla GPUs, which come with 16GB of VRAM. They’re pretty fast, and the best part is that you don’t need to download the datasets locally it’s all handled on the platform!
## Note(!)

Ensure you install the version of PyTorch compatible with your GPU. 
You can find the correct version for your setup by visiting the [PyTorch installation page](https://pytorch.org/get-started/locally/).
I recommend using CUDA 12.1 or higher for better performance.

## How to Use
### 1. Train you own Model
- Download the COCO dataset from [this link](https://cocodataset.org/#download) and place it in the `data/content_dir` directory.(Also change the paths in `config.ini`)

- Use the `train.py` script to train the style transfer model.
- Run the following command:
  ```bash
  python train.py 
  ```

### 4. Use the Pretrained Model
- If you don't want to train the model yourself, you can use the pretrained model available in this repository.
  You can download the checkpoints [Pretrained Model](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/models/checkpoints/).
  And after loading them you can just simply test the different version of model. 
---

## Results
Below are some example results of the style transfer model:

| Content Image | Style Image | Stylized Image |
|---------------|-------------|----------------|
| ![Content](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/content_dir/test_image_01.png) | ![style](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/style_images/Vincent_van_Gogh.png) | ![stylized](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/styled_image.png) |
| ![new1](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/content_dir/family-gc23518eae_640.jpg) | ![new2](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/style_images/mosaic.jpg) | ![new3](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/styled_image_01.png) |
| ![new4](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/content_dir/image_04.jpg) | ![new5](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/style_images/image_01.jpg) | ![new6](https://github.com/emanalytic/Perceptual-Losses-Neural-Style-Transfer/blob/main/data/styled_image_02.png) |


---

## Files in the Repository

- `train.py`: Script to train the style transfer model.
- `vgg.py`: Script for hook manager to get features
- `model.py`: Model Implementation from pytorch example repo
- `loss.py` : Perceptual Loss Implementation
- `neural_style_transfer.ipynb`: Jupyter notebook explaining the concepts and code in detail.
- `data/`: Directory for storing the content images and style image.
- `utils/`: For configuration and utility functions 
---

## References

- ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/pdf/1603.08155)
- [Gatys et al.'s Neural Style Transfer](https://arxiv.org/pdf/1508.06576)
- [PyTorch Neural Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [Pytorch Example](https://github.com/pytorch/examples)
- [Reference Repository](https://github.com/francislata/Perceptual-Losses/tree/master)
- [Helpful Article](https://sh-tsang.medium.com/brief-review-perceptual-losses-for-real-time-style-transfer-and-super-resolution-ac4fd2658b8)
---

