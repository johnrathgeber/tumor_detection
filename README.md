# Tumor Detection CNN

This repository contains an image classification pipeline for tumor detection using convolutional neural networks (CNNs). It includes two models: a custom CNN trained from scratch and a VGG16-based model using transfer learning. The project also supports evaluation using saliency maps and LIME visualizations. Read the full report here: https://www.johnrathgeber.com/Final_Report.pdf

## Features

- Train from scratch or fine-tune a VGG16 model
- Visual explanations using LIME and saliency maps
- TensorBoard logging (with optional confusion matrix)
- Checkpoint saving and reloading
- Configurable training and evaluation via CLI

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-image
- LIME

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Training From Scratch (Custom CNN)

```bash
python main.py --task 1 --data path/to/data
```

### Fine-Tuning VGG16

```bash
python main.py --task 3 --data path/to/data --load-vgg path/to/vgg16_imagenet.h5
```

### Evaluating a Model

```bash
python main.py --task 1 --evaluate --load-checkpoint path/to/checkpoint.h5 --lime-image path/to/image.jpg
```

### Optional Arguments

- `--load-checkpoint`: Resume training or evaluate a specific checkpoint
- `--lime-image`: Path to a sample image for LIME and saliency map visualization
- `--confusion`: Log confusion matrices to TensorBoard
- `--evaluate`: Skip training and run evaluation on the test set

## Visualization Output

LIME and saliency visualizations are saved automatically during evaluation:

- `lime_explainer_images/<timestamp>/`: LIME output images
- `saliency_visualization_<timestamp>.png`: Saliency overlay

## Example Hyperparameters File

```python
# hyperparameters.py
img_size = 224
learning_rate = 0.0001
num_epochs = 20
max_num_weights = 5
```

## Notes

- The custom model is defined as a stack of Conv2D, MaxPooling, Dropout, and Dense layers.
- The VGG16-based model freezes convolutional layers and trains a new classifier head.
- Dataset must be formatted with train/val/test directories as expected by `ImageDataGenerator`.
