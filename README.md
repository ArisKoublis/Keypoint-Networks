# Keypoint-Networks

## Introduction
Neural networks are commonly used in Formula Student to detect cone keypoints for improved distance estimation (see: [FSOCO paper](https://arxiv.org/abs/2007.13971)).

This repository explores lightweight neural network architectures for keypoint heatmap prediction. I systematically evaluate the impact of design choices—including network architecture, optimizers, learning rates, batch size, normalization strategies, and output activations—while keeping models small enough for real-time embedded systems (~80 KB memory footprint).

---

## Training Data
The models are trained on a subset of the **FSOCO dataset**, containing ~4k images of Formula Student cones. Images were resized to 64×48 pixels and annotated with 7 keypoints per cone using **Supervisely**.

Rather than predicting keypoint coordinates directly, the network predicts **heatmaps** for each keypoint. Direct coordinate regression is unstable because small errors can lead to large jumps in output and weak gradient signals. It can also fail in out-of-distribution cases: when multiple locations are plausible, regression often predicts their average, producing meaningless points.

To address this, **Gaussian heatmaps** are used as targets:
- They provide smoother, more informative supervision than single-pixel labels.
- They make training more tolerant to small localization errors and labeling noise, which is common in low-resolution or blurry images.

---

## Architectures

### ResNet
**ResNet (Residual Network)** is a convolutional architecture with skip (residual) connections. Each block adds its input to the output, allowing layers to learn residual functions. This improves gradient flow, reduces training instability, and enables deeper networks to train effectively. Even in small, lightweight models, ResNet helps the network learn meaningful features efficiently.

### U-Net
**U-Net** is an encoder–decoder architecture for dense, pixel-wise predictions. The encoder captures semantic context through downsampling, while the decoder restores spatial resolution. Skip connections pass fine-grained details from the encoder to the decoder, allowing the network to localize features accurately while maintaining global context. This makes U-Net especially effective for segmentation and keypoint heatmap prediction.

---

## Other Design Choices

### Heatmaps vs. Coordinate Regression
Predicting coordinates directly turns localization into a single regression problem. Small spatial errors can cause large coordinate deviations, and the model may average multiple plausible points in ambiguous cases, resulting in meaningless predictions. Heatmap prediction preserves spatial structure, provides dense supervision, and allows multiple high-probability regions, making training more stable and robust. Gaussian heatmaps further improve stability by rewarding predictions that are close to the true keypoint, instead of penalizing small deviations heavily.

### Batch Normalization
Batch normalization stabilizes training by normalizing layer activations within each mini-batch, addressing **internal covariate shift**—the changing distribution of layer inputs during training. It improves gradient flow, allows higher learning rates, and acts as a regularizer. Its effectiveness depends on batch size: very small batches can make statistics noisy.

### Final Layer Activation Functions
- **Sigmoid** maps each output independently to [0,1], suitable for multi-label or independent pixel predictions.  
- **Softmax** converts a vector of values into a probability distribution that sums to 1, emphasizing a single high-probability location per keypoint channel.  

In keypoint heatmaps, sigmoid allows multiple high-probability regions, while softmax emphasizes a single peak.

### Optimizers
Various optimizers (e.g., Adam, RMSProp) are tested to study their effect on convergence speed and final accuracy.

### Batch Size
**Batch size** is the number of examples processed together per forward/backward pass. It affects training stability, convergence, and generalization:

- **Small batch sizes (4–16):**  
  - Provide noisy gradient estimates, which can improve generalization.  
  - Can make BatchNorm unstable due to limited statistics.  
  - Require less memory, useful for lightweight models.  

- **Large batch sizes (32–128):**  
  - Give more stable and accurate gradient estimates.  
  - Require more memory and may sometimes reduce generalization if too large.
