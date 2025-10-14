# Periodic Pattern Extraction and Uniform Lighting Correction

This is the third project in ECE 4367 - Image Processing. There are 2 goals/task in this project: \
Given an image:
1) Extract the vague periodic pattern in the image
2) Apply Uniform Lighting throughout the original image

## Overview

Many images contain subtle periodic patterns—textures, grids, or repeating structures—that can be detected and manipulated in the frequency domain. This tool leverages the power of Fourier analysis to:

- **Extract** repeating patterns and textures
- **Suppress** unwanted noise or illumination gradients
- **Correct** uneven lighting for uniform brightness
- **Analyze** the frequency composition of images

### How It Works

1. **Fourier Transform (FFT)**: Converts images from the spatial domain (pixel intensities) to the frequency domain, where repeating patterns appear as distinct peaks in the magnitude spectrum.

2. **Magnitude and Phase Analysis**: 
   - The magnitude spectrum reveals the strength of frequency components
   - The phase spectrum encodes spatial arrangement

3. **Band-pass Filtering**: Isolates specific frequency bands to extract patterns while suppressing low-frequency illumination changes and high-frequency noise.

4. **Inverse FFT**: Reconstructs the filtered image back into the spatial domain.

## Features

✨ **FFT Computation** - Uses OpenCV's `cv2.dft` for efficient 2D Fourier Transform  
📊 **Magnitude/Phase Analysis** - Visualize pattern strength and orientation  
🎛️ **Frequency-Domain Filtering** - Apply circular or band-pass masks  
🔍 **Pattern Extraction** - Isolate repeating textures and structures  
💡 **Lighting Correction** - Remove illumination gradients for uniform brightness  
📈 **Visualization** - Examine original, FFT, filter masks, and results

## Mathematical Foundation

### Discrete Fourier Transform (DFT)

$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-2\pi i(\frac{ux}{M} + \frac{vy}{N})}$$

Where $f(x,y)$ is the grayscale image and $F(u,v)$ is the complex frequency spectrum. Peaks in $|F(u,v)|$ indicate strong periodic components.

### Magnitude Spectrum

$$|F(u,v)| = \sqrt{\text{Re}(F(u,v))^2 + \text{Im}(F(u,v))^2}$$

Highlights dominant frequencies corresponding to repeating patterns.

### Phase Spectrum

$$\phi(u,v) = \arctan\left(\frac{\text{Im}(F(u,v))}{\text{Re}(F(u,v))}\right)$$

Retains the spatial alignment of patterns.

### Band-pass Filtering

$$H(u,v) = \begin{cases} 
1, & r_{\text{in}} \leq \sqrt{(u-u_0)^2 + (v-v_0)^2} \leq r_{\text{out}} \\
0, & \text{otherwise}
\end{cases}$$

Suppresses unwanted frequencies while retaining the desired pattern.

### Inverse FFT

$$f_{\text{filtered}}(x,y) = \text{IDFT}[H(u,v) \cdot F(u,v)]$$

Reconstructs the spatial image with only selected frequencies.

## Installation

### Requirements

- Python 3.9+
- OpenCV
- NumPy
- Matplotlib

### Install Dependencies

```bash
    pip install opencv-python numpy matplotlib
```

## Usage

```bash
    python main.py <path/to/image_file>
```

### Processing Pipeline(as of now)

1. Converts input image to grayscale
2. Computes FFT, magnitude, and phase spectra
3. Applies band-pass filter to extract periodic patterns
4. Reconstructs filtered pattern in spatial domain
5. Displays visualizations for analysis


## Project Structure

```
.
├── main.py       # Main script
├── README.md           # This file
└── example_image.jpg   # Sample input image
```

---

**Note**: This is still under dev as I am trying to extract the pattern