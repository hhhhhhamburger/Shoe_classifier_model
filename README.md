# Shoe Brand Classification with Transfer Learning

## Overview

This Jupyter notebook implements a deep learning model for classifying shoe brands using transfer learning with ResNet50. The model distinguishes between three major shoe brands: Adidas, Converse, and Nike.

## File Description

- **File**: `ML_project.ipynb`
- **Type**: Jupyter Notebook
- **Language**: Python
- **Framework**: TensorFlow/Keras

## Project Features

- **Transfer Learning**: Uses pre-trained ResNet50 model
- **Data Augmentation**: Comprehensive image augmentation techniques
- **Model Evaluation**: Validation and testing procedures
- **Visualization**: Training history plots

## Dataset Information

- **Training Set**: 603 images (201 per class)
- **Validation Set**: 90 images (30 per class)
- **Classes**: Adidas, Converse, Nike
- **Image Size**: 224x224 pixels

## Model Architecture

### Base Model
- **Pre-trained Model**: ResNet50 with ImageNet weights
- **Transfer Learning**: Last 15 layers are fine-tuned
- **Input Shape**: (224, 224, 3)

### Custom Layers
```
GlobalAveragePooling2D()
BatchNormalization()
Dense(512, activation='relu')
Dropout(0.5)
Dense(256, activation='relu')
Dropout(0.3)
Dense(3, activation='softmax')
```

## Data Augmentation

- Rotation: ±15 degrees
- Width/Height Shift: ±10%
- Zoom: ±10%
- Horizontal/Vertical Flip
- Brightness Adjustment: ±10%

## Training Configuration

- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 25 (with early stopping)
- **Callbacks**: ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

## Performance

- **Validation Accuracy**: 67.78%
- **Model Format**: HDF5 (.h5)

## Requirements

```
tensorflow>=2.0.0
keras
numpy
matplotlib
pathlib
```

## Usage

1. **Install Dependencies**:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

2. **Prepare Dataset Structure**:
   ```
   ML_dataset/
   ├── train/
   │   ├── adidas/
   │   ├── converse/
   │   └── nike/
   └── validate/
       ├── adidas/
       ├── converse/
       └── nike/
   ```

3. **Run the Notebook**:
   - Open `ML_project.ipynb` in Jupyter Notebook
   - Execute cells sequentially
   - Monitor training progress

## Key Implementation Details

- **Transfer Learning**: Leverages ResNet50 pre-trained on ImageNet
- **Data Preprocessing**: Image resizing and normalization
- **Model Optimization**: Learning rate scheduling and early stopping
- **Regularization**: Dropout layers and batch normalization

## Results

The model achieves 67.78% validation accuracy on the three-class shoe brand classification task. The notebook includes:

- Training history visualization
- Loss and accuracy plots
- Model evaluation metrics
