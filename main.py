import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import List


# -------------------- FFT Utils -------------------- #
def get_fft(gray: np.ndarray):
    """Compute the centered FFT, magnitude, and phase of a grayscale image."""
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)

    # Magnitude and phase
    magnitude = np.abs(f_shift)
    magnitude_log = np.log1p(magnitude)  # for visualization
    phase = np.angle(f_shift)

    return f_shift, magnitude, phase, magnitude_log



def get_peaking_frequencies(mag: np.ndarray, num_peaks: int = 12, exclude_center: int = 8) -> list[tuple[int, int]]:
    """Detect top N peaking frequencies, excluding DC center."""
    rows, cols = mag.shape
    crow, ccol = rows // 2, cols // 2
    
    mag_copy = mag.copy()
    Y, X = np.ogrid[:rows, :cols]
    center_mask = np.sqrt((X - ccol)**2 + (Y - crow)**2) < exclude_center
    mag_copy[center_mask] = 0

    flat_indices = np.argsort(mag_copy.ravel())[-num_peaks:]
    peak_coords = np.unravel_index(flat_indices, mag_copy.shape)

    relative_coords = []
    for y, x in zip(peak_coords[0], peak_coords[1]):
        u = x - ccol
        v = y - crow
        relative_coords.append((u, v))
        print(f"u: {u}, v: {v}")

    return relative_coords


def generate_filter(gray: np.ndarray, sigma: float, peaking_freqs: List[tuple[int, int]]) -> np.ndarray:
    """
    Generates a Gaussian Bandpass Filter or a High-Pass Filter (HPF).
    """

    
    h, w = gray.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    cy, cx = h // 2, w // 2
    sigma_sq = sigma ** 2

    mask = np.zeros((h, w), dtype=np.float32)

        # Include symmetric pairs and the DC component
    freqs = list(set(peaking_freqs + [(-u, -v) for (u, v) in peaking_freqs]))
    freqs.append((0, 0))

    for (u, v) in freqs:
        x0, y0 = cx + u, cy + v
    
        # 2D Gaussian centered at the peak (x0, y0)
        G = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma_sq))
        mask += G

    # Normalize the mask to ensure max value is 1
    max_val = np.max(mask)
    if max_val > 1e-6:
        mask /= max_val
        
    return mask

def plot_peaks(mag: np.ndarray, peaking_freqs: list[tuple[int, int]], radius: int = 5) -> np.ndarray:
    """Overlay red circles on log-magnitude spectrum to visualize detected peaks."""
    mag_disp = np.log1p(mag)
    mag_disp = cv.normalize(mag_disp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    mag_disp = cv.merge([mag_disp]*3)

    h, w = mag.shape
    crow, ccol = h // 2, w // 2
    
    # Include symmetric pairs and DC component for plotting circles
    coords = set(peaking_freqs + [(-u, -v) for (u, v) in peaking_freqs] + [(0, 0)])

    for (u, v) in coords:
        x, y = int(ccol + u), int(crow + v)
        cv.circle(mag_disp, (x, y), radius, (0, 0, 255), 1)

    return mag_disp


def reconstruct_pattern(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstruct directly using Inverse FFT with the mask applied."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Apply the mask: keep only the selected frequencies
    f_filtered = fshift * (mask / 255)

    ishift = np.fft.ifftshift(f_filtered)
    img_back = np.fft.ifft2(ishift)
    return np.abs(img_back)

def apply_uniform_lighting(gray: np.ndarray, blur_kernel: int = 101) -> np.ndarray:
    """Divides the image by its heavily blurred version to remove illumination."""
    # Large kernel for strong blur, adjust for your image scale
    blurred = cv.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    normalized = gray.astype(np.float32) / (blurred.astype(np.float32) + 1e-8)  # Avoid division by zero
    normalized = cv.normalize(normalized, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return normalized


# -------------------- Visualization -------------------- #
def show_image(working_img: np.ndarray, mag: np.ndarray, mag_peaks: np.ndarray) -> None:
    """Display 3 panels for analysis."""
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    axs[0].imshow(working_img, cmap='gray')
    axs[0].set_title(f"Working")
    axs[0].axis("off")

    axs[1].imshow(mag, cmap='magma')
    axs[1].set_title(f"(Magnitude Spectrum (Log))")
    axs[1].axis("off")

    axs[2].imshow(mag_peaks, cmap="magma")
    axs[2].set_title(f"(Detected Frequency Peaks)")
    axs[2].axis("off")


    plt.tight_layout()
    plt.show()
# -------------------- Main -------------------- #

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_file>")
        sys.exit(1)

    img = cv.imread(sys.argv[1])
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # --- 1. Original Image Analysis ---
    fft, mag, _, mag_log = get_fft(gray_img)
    peak_freqs = get_peaking_frequencies(mag)
    mag_peaks_vis = plot_peaks(mag, peak_freqs)
    show_image(gray_img, mag_log, mag_peaks_vis, title_prefix="Original")

    # --- 2. Pattern Extraction ---
    mask = generate_filter(gray=gray_img, peaking_freqs=peak_freqs, sigma=1.0)
    reconstructed = reconstruct_pattern(fft, mask)
    _, p_mag, _, p_mag_log = get_fft(reconstructed)
    p_peaks = get_peaking_frequencies(p_mag)
    p_vis = plot_peaks(p_mag, p_peaks)
    show_image(reconstructed, p_mag_log, p_vis, title_prefix="Extracted Pattern")

    
    
    


if __name__ == "__main__":
    main()
