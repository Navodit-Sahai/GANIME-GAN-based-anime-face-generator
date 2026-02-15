# GANIME - GAN-based Anime Face Generator

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating realistic anime faces using PyTorch.

## Overview

GANIME uses a DCGAN architecture to learn and generate anime-style faces from the Anime Face Dataset. The model consists of a generator network that creates images from random noise and a discriminator network that distinguishes between real and generated images.

## Features

- **DCGAN Architecture**: Implements the proven Deep Convolutional GAN design
- **64x64 RGB Output**: Generates high-quality 64x64 pixel anime faces
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs
- **Training Visualization**: Saves generated samples during training to track progress
- **Model Persistence**: Save and load trained models

## Requirements

```bash
pip install torch torchvision
pip install opendatasets
pip install matplotlib tqdm
```

## Dataset

The project uses the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) from Kaggle, which contains thousands of anime character faces.

## Architecture

### Generator
- **Input**: 128-dimensional latent vector
- **Architecture**: 5 transposed convolutional layers with batch normalization
- **Output**: 64×64×3 RGB image with Tanh activation
- **Feature maps**: 512 → 256 → 128 → 64 → 3

### Discriminator
- **Input**: 64×64×3 RGB image
- **Architecture**: 4 convolutional layers with batch normalization and LeakyReLU
- **Output**: Single value (real/fake classification)
- **Feature maps**: 64 → 128 → 256 → 512 → 1

## Training Details

- **Optimizer**: Adam with β₁=0.5, β₂=0.999
- **Learning Rate**: 0.0002
- **Batch Size**: 128
- **Loss Function**: Binary Cross-Entropy with Logits
- **Label Smoothing**: Real labels smoothed to 0.9 (reduces discriminator overconfidence)
- **Normalization**: Images normalized to [-1, 1] range

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Navodit-Sahai/GANIME-GAN-based-anime-face-generator.git
cd GANIME-GAN-based-anime-face-generator
```

### 2. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Navodit-Sahai/GANIME-GAN-based-anime-face-generator/blob/main/GANIME.ipynb)

### 3. Run Training

```python
# Set hyperparameters
lr = 0.0002
epochs = 15

# Train the model
history = fit(epochs, lr)
```

### 4. Generate Images

```python
# Load trained generator
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Generate new faces
latent = torch.randn(64, 128, 1, 1).to(device)
with torch.no_grad():
    fake_images = generator(latent)
    
show_images(fake_images)
```

## Project Structure

```
GANIME/
├── GANIME.ipynb           # Main Jupyter notebook
├── README.md              # Project documentation
├── generator.pth          # Trained generator weights (after training)
├── discriminator.pth      # Trained discriminator weights (after training)
└── generated/             # Folder containing generated samples during training
    ├── generated-images-0001.png
    ├── generated-images-0002.png
    └── ...
```

## Training Output

The model saves generated samples at the end of each epoch to the `generated/` directory. You can monitor training progress by viewing these images.

Example output metrics:
```
Epoch [1/15] | Loss_G: 1.2345 | Loss_D: 0.8765 | Real Score: 0.7543 | Fake Score: 0.2891
```

- **Loss_G**: Generator loss (lower = generator is fooling discriminator better)
- **Loss_D**: Discriminator loss (balanced around 0.7-1.4 is ideal)
- **Real Score**: Discriminator's confidence on real images (should be ~0.8-0.9)
- **Fake Score**: Discriminator's confidence on fake images (should be ~0.1-0.3)

## Tips for Best Results

1. **Training Duration**: Train for at least 15-25 epochs for good results
2. **GPU Recommended**: Training on CPU is very slow; use Google Colab's free GPU
3. **Monitor Scores**: 
   - If discriminator dominates (real_score ≈ 1.0, fake_score ≈ 0.0), decrease discriminator training frequency
   - If generator dominates (fake_score ≈ 1.0), increase discriminator training frequency
4. **Avoid Mode Collapse**: If all generated faces look identical, reduce learning rate or add more training data variety

## Common Issues & Solutions

### Issue: Loss values are NaN or extremely high
**Solution**: Ensure discriminator outputs raw logits (no sigmoid) when using `binary_cross_entropy_with_logits`

### Issue: Generated images are just noise
**Solution**: Train for more epochs; early epochs typically produce noise before learning features

### Issue: Mode collapse (all faces look the same)
**Solution**: 
- Add label smoothing (already implemented with 0.9)
- Reduce learning rate
- Train discriminator multiple times per generator update

## Model Performance

The quality of generated anime faces improves significantly over epochs:
- **Epochs 1-5**: Blurry shapes and colors emerge
- **Epochs 5-10**: Basic facial features become recognizable
- **Epochs 10-15**: Sharp, detailed anime faces with distinct features
- **Epochs 15+**: High-quality, diverse anime character faces

## Future Improvements

- [ ] Implement Progressive GAN for higher resolution output
- [ ] Add conditional generation (control hair color, eye color, etc.)
- [ ] Implement Wasserstein GAN with gradient penalty for more stable training
- [ ] Add FID (Fréchet Inception Distance) metric for quantitative evaluation
- [ ] Web interface for interactive generation

## References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434) - Radford et al., 2015
- [GAN Paper](https://arxiv.org/abs/1406.2661) - Goodfellow et al., 2014
- [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)

## License

This project is open source and available under the MIT License.

## Author

**Navodit Sahai**

- GitHub: [@Navodit-Sahai](https://github.com/Navodit-Sahai)

## Acknowledgments

- Thanks to the Kaggle community for the Anime Face Dataset
- Inspired by the original DCGAN paper and implementation
- Built with PyTorch

---

⭐ If you found this project helpful, please consider giving it a star!
