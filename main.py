import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# -------------------- FFT Utils -------------------- #
def get_fft(gray: np.ndarray):
    """Compute the centered FFT, magnitude, and phase of a grayscale image."""
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)

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

def plot_peaks(mag: np.ndarray, peaking_freqs: List[Tuple[int,int]], radius: int = 5) -> np.ndarray:
    """Overlay red circles on magnitude spectrum to visualize detected peaks."""
    mag_disp = cv.normalize(np.log1p(mag), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    mag_disp = cv.merge([mag_disp]*3)

    h, w = mag.shape
    crow, ccol = h // 2, w // 2
    
    coords = set(peaking_freqs + [(-u, -v) for (u, v) in peaking_freqs])
    for (u, v) in coords:
        x, y = int(ccol + u), int(crow + v)
        cv.circle(mag_disp, (x, y), radius, (0, 0, 255), 1)

    return mag_disp


def reconstruct_pattern(fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstruct image using inverse FFT with mask applied."""
    f_filtered = fft * mask  # mask already 0–1
    ishift = np.fft.ifftshift(f_filtered)
    img_back = np.fft.ifft2(ishift)
    return np.abs(img_back)


def apply_uniform_lighting(original: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Remove periodic pattern to produce uniform lighting."""
    uniform = original / (pattern + 1e-5)  # avoid divide by zero
    uniform = cv.normalize(uniform, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return uniform


# -------------------- Visualization -------------------- #
def show_full_analysis(gray, gray_uniform, pattern, mag_log, mag_peaks, filter_mask, title_prefix=""):
    """Display 2x3 grid for image + frequency analysis."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Top row
    axs[0, 0].imshow(gray, cmap='gray')
    axs[0, 0].set_title(f"{title_prefix}Gray Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gray_uniform, cmap='gray')
    axs[0, 1].set_title(f"{title_prefix}Uniform Lighting")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(pattern, cmap='gray')
    axs[0, 2].set_title("Extracted Pattern")
    axs[0, 2].axis("off")

    # Bottom row
    axs[1, 0].imshow(mag_log, cmap='magma')
    axs[1, 0].set_title("Magnitude Spectrum (Log)")
    axs[1, 0].axis("off")

    # Only use cmap if image is single-channel
    if mag_peaks.ndim == 2 or (mag_peaks.ndim == 3 and mag_peaks.shape[2] == 1):
        axs[1, 1].imshow(mag_peaks, cmap='magma')
    else:
        axs[1, 1].imshow(mag_peaks)
    axs[1, 1].set_title("Magnitude + Peaking Freqs")
    axs[1, 1].axis("off")

    # For filter_mask, handle channel case
    if filter_mask.ndim == 2 or (filter_mask.ndim == 3 and filter_mask.shape[2] == 1):
        axs[1, 2].imshow(filter_mask, cmap='jet')
    else:
        axs[1, 2].imshow(filter_mask[:, :, 0], cmap='jet')
    axs[1, 2].set_title("Pattern Extraction Filter")
    axs[1, 2].axis("off")

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

    # --- 2. Pattern Extraction ---
    mask = generate_filter(gray=gray_img, peaking_freqs=peak_freqs, sigma=1.0)
    reconstructed = reconstruct_pattern(fft, mask)
    _, p_mag, _, p_mag_log = get_fft(reconstructed)
    p_peaks = get_peaking_frequencies(p_mag)
    p_vis = plot_peaks(p_mag, p_peaks)


    # --- 3. Uniform Lighting ---
    uniform_img = apply_uniform_lighting(gray_img, reconstructed)
    
    show_full_analysis(gray=gray_img, gray_uniform=uniform_img, pattern=reconstructed, mag_log=mag_log, mag_peaks=mag_peaks_vis, filter_mask=mask, title_prefix="Project3 Image")

    


if __name__ == "__main__":
    main()
