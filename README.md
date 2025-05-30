# BayesWarp

**BayesWarp** is a principled framework for white-box testing of deep neural networks, leveraging Bayesian optimization and interpretability techniques to systematically uncover diverse model failures while keeping generated samples close to the original data distribution.

---

## 🚩 Key Features

* **Principled Failure Discovery:**
  Region localization via GradCAM, Integrated Gradients, and SmoothGrad, focusing search on critical input areas.
* **Bayesian Optimization:**
  Adaptive, SVGP-accelerated perturbation search to expose diverse and realistic model failures.
* **Comprehensive Metrics:**
  Built-in evaluation metrics (NoF, DoF, Seed Coverage, FID) for rigorous comparison.
* **Flexible Model and Dataset Support:**
  Pre-defined pipelines for LeNet, ResNet, VGG; compatible with MNIST, CIFAR-10, and ImageNet-50.

---

## 💻 Installation

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

### MNIST / CIFAR-10

```bash
cd data
run mnist/cifar10.py
```

### ImageNet

Please organize your subset into the following directory structure (only include the selected 50 classes):

```
data/imagenet/train
data/imagenet/val
```

---

## 🚀 Quick Start

### Train a Model

**LeNet4/LeNet5 on MNIST:**

```bash
python bayeswarp/lenet4/train/train_lenet4.py
python bayeswarp/lenet5/train/train_lenet5.py
```

**ResNet / VGG on CIFAR-10 or ImageNet:**

```bash
python bayeswarp/resnet18/train/train_resnet18.py
python bayeswarp/vgg16/train/train_vgg16.py
python bayeswarp/resnet50/train/train_resnet50.py
python bayeswarp/vgg19/train/train_vgg19.py
```

### Run BayesWarp Testing

Use the `core.py` in each model file.
For example:

```bash
python bayeswarp/lenet4/core.py
```

### Evaluate and Analyze Results

Evaluation scripts and metrics are provided for NoF, DoF, Seed Coverage, and FID.

---

## 📊 Reproducing Experiments

All scripts for training, testing, and evaluation are provided to facilitate reproducibility.


