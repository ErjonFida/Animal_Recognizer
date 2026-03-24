# Animal Recognition CNN

This project trains a custom Convolutional Neural Network (CNN) from scratch using PyTorch to recognize images of 15 categories of animals.

## Dataset
The model is trained on the publicly available Kaggle Animal data dataset. (https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset)

## Model Architecture
The primary model (`SimpleCnn`) uses a custom deep CNN architecture:
- 5 Convolutional Blocks (each with two `Conv2d` layers, Batch Normalization, ReLU activation function, and Max Pooling)
- Global Average Pooling (`AdaptiveAvgPool2d`)
- A fully connected classifier with Dropout for regularization

## Training Details
- **Data Augmentation**: Augmentation is applied to the training set (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomAffine`, `RandomErasing`) to prevent overfitting.
- **Optimizer**: Adam (`lr=0.0001`, `weight_decay=1e-4`)
- **Learning Rate Scheduler**: `CosineAnnealingLR` (T_max=100)
- **Gradient Clipping**: Applied to stabilize training.
- **Hardware Acceleration**: Supports Intel GPUs via `torch_directml` (if installed) or standard CUDA/CPU via PyTorch.

## Usage

1. **Install Dependencies**: Ensure you have `torch`, `torchvision`, `Pillow`, `matplotlib`, `numpy`, and `scipy` installed.
    ```bash
    pip install torch torchvision Pillow matplotlib numpy scipy
    ```
    *Optional: Install `torch-directml` for Intel GPU support on Windows.*

2. **Prepare Data**: Place the training data in `Kaggle_Animal_data/Training Data/Training Data`.

3. **Train Model**: Run the training script.
    ```bash
    python Insect_Recognize.py
    ```
    The script will automatically detect the best available device, train the model for 100 epochs, and save checkpoints (`Animal_model_epoch_*.pth`) after each epoch. It will also seamlessly resume training from the latest checkpoint if interrupted.

## Evaluation
Validation accuracy is reported after every epoch, and a final test set evaluation is performed at the end of training. After 5 training epochs the testing accuraccy reaches 62.36%.
