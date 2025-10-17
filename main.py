import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def get_fft(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the centered Fast Fourier Transform (FFT) of a grayscale image.

    Args:
        gray (np.ndarray): Input grayscale image.

    Returns:
        tuple: 
            - f_shift (np.ndarray): Shifted FFT of the image.
            - magnitude (np.ndarray): Magnitude spectrum of the FFT.
            - phase (np.ndarray): Phase spectrum of the FFT.
            - magnitude_log (np.ndarray): Logarithm of the magnitude (for visualization).
    """
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)
    magnitude_log = np.log1p(magnitude)
    phase = np.angle(f_shift)
    return f_shift, magnitude, phase, magnitude_log


def get_peaking_frequencies(
        mag: np.ndarray,
        num_peaks: int = 8,
        exclude_center: int = 8
) -> list[tuple[int, int]]:
    """
    Locate the top N peaking frequency coordinates in the magnitude spectrum,
    excluding the DC component at the center.

    Args:
        mag (np.ndarray): Magnitude spectrum of the FFT.
        num_peaks (int, optional): Number of peaks to detect. Defaults to 12.
        exclude_center (int, optional): Radius around the center to exclude. Defaults to 8.

    Returns:
        list[tuple[int, int]]: List of relative (u, v) frequency coordinates for the detected peaks.
    """
    rows, cols = mag.shape
    crow, ccol = rows // 2, cols // 2

    mag_copy = mag.copy()
    Y, X = np.ogrid[:rows, :cols]
    center_mask = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2) < exclude_center
    mag_copy[center_mask] = 0

    flat_indices = np.argsort(mag_copy.ravel())[-num_peaks:]
    peak_coords = np.unravel_index(flat_indices, mag_copy.shape)

    relative_coords = []
    for y, x in zip(peak_coords[0], peak_coords[1]):
        u = x - ccol
        v = y - crow
        relative_coords.append((u, v))

    return relative_coords


def generate_filter(
        gray: np.ndarray,
        sigma: float,
        peaking_freqs: List[tuple[int, int]]
) -> np.ndarray:
    """
    Generate a Gaussian bandpass filter mask centered at the detected peaking frequencies.

    Args:
        gray (np.ndarray): Input grayscale image (for shape).
        sigma (float): Standard deviation for the Gaussian.
        peaking_freqs (List[tuple[int, int]]): List of frequency coordinates to center Gaussians.

    Returns:
        np.ndarray: Frequency domain filter mask.
    """
    h, w = gray.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    cy, cx = h // 2, w // 2
    sigma_sq = sigma ** 2

    mask = np.zeros((h, w), dtype=np.float32)
    freqs = {(u, v) for (u, v) in peaking_freqs}
    freqs |= {(-u, -v) for (u, v) in peaking_freqs}
    freqs.add((0, 0))  

    for (u, v) in freqs:
        x0, y0 = cx + u, cy + v
        gauss = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma_sq))
        mask += gauss

    # Normalize the mask
    max_val = np.max(mask)
    if max_val > 1e-6:
        mask /= max_val

    return mask


def plot_peaks(
        mag_log: np.ndarray,
        peaking_freqs: List[Tuple[int, int]],
        radius: int = 5
) -> np.ndarray:
    """
    Overlay red circles on the log-magnitude spectrum to visualize detected peaking frequencies.

    Args:
        mag (np.ndarray): Magnitude spectrum of the FFT.
        peaking_freqs (List[Tuple[int, int]]): List of (u, v) frequency coordinates.
        radius (int, optional): Radius of the circle to draw. Defaults to 5.

    Returns:
        np.ndarray: RGB visualization of magnitude spectrum with overlaid peaks.
    """
    mag_disp = cv.normalize(mag_log, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    mag_disp = cv.merge([mag_disp] * 3)

    h, w = mag_log.shape
    crow, ccol = h // 2, w // 2

    coords = {(u, v) for (u, v) in peaking_freqs}
    coords |= {(-u, -v) for (u, v) in peaking_freqs}

    for (u, v) in coords:
        x, y = int(ccol + u), int(crow + v)
        cv.circle(mag_disp, (x, y), radius, (0, 0, 255), 1)

    return mag_disp


def reconstruct_pattern(
        fft: np.ndarray,
        mask: np.ndarray
) -> np.ndarray:
    """
    Reconstructs an image pattern by applying a filter mask in the frequency domain,
    then inverting the FFT.

    Args:
        fft (np.ndarray): Shifted FFT of the image.
        mask (np.ndarray): Frequency domain filter mask.

    Returns:
        np.ndarray: Reconstructed image with filtered pattern.
    """
    f_filtered = fft * mask
    ishift = np.fft.ifftshift(f_filtered)
    img_back = np.fft.ifft2(ishift)
    return np.abs(img_back)


def apply_uniform_lighting(
        original: np.ndarray,
        pattern: np.ndarray
) -> np.ndarray:
    """
    Remove a periodic pattern from an image to produce a uniform lighting effect.

    Args:
        original (np.ndarray): Original grayscale image.
        pattern (np.ndarray): Extracted periodic pattern to divide out.

    Returns:
        np.ndarray: Image with uniform lighting correction applied.
    """
    uniform = original / (pattern + 1e-5)  # Avoid divide by zero.
    uniform = cv.normalize(uniform, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return uniform


def show_full_analysis(
        gray: np.ndarray,
        gray_uniform: np.ndarray,
        pattern: np.ndarray,
        mag_log: np.ndarray,
        mag_peaks: np.ndarray,
        filter_mask: np.ndarray,
        title_prefix: str = ""
):
    """
    Display a 2x3 grid panel visualizing all stages of frequency analysis.

    Top row:
        [Gray Image, Uniform Lighting, Extracted Pattern]
    Bottom row:
        [Magnitude Spectrum, Magnitude with Peaks, Filter Mask]

    Args:
        gray (np.ndarray): Original grayscale image.
        gray_uniform (np.ndarray): Image after uniform lighting correction.
        pattern (np.ndarray): Extracted periodic pattern.
        mag_log (np.ndarray): Log-magnitude spectrum for visualization.
        mag_peaks (np.ndarray): Magnitude spectrum with peaks highlighted.
        filter_mask (np.ndarray): Frequency filter visualization.
        title_prefix (str, optional): Prefix for subplot titles. Defaults to "".
    """
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
    axs[1, 0].set_title("Magnitude Spectrum")
    axs[1, 0].axis("off")

    if mag_peaks.ndim == 2 or (mag_peaks.ndim == 3 and mag_peaks.shape[2] == 1):
        axs[1, 1].imshow(mag_peaks, cmap='magma')
    else:
        axs[1, 1].imshow(mag_peaks)
    axs[1, 1].set_title("Magnitude + Peaking Freqs")
    axs[1, 1].axis("off")

    if filter_mask.ndim == 2 or (filter_mask.ndim == 3 and filter_mask.shape[2] == 1):
        axs[1, 2].imshow(filter_mask, cmap='gray')
    else:
        axs[1, 2].imshow(filter_mask[:, :, 0], cmap='gray')
    axs[1, 2].set_title("Pattern Extraction Filter")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main():

    if len(sys.argv) < 2:
        print("Usage: python script.py <image_file>")
        sys.exit(1)

    img = cv.imread(sys.argv[1])
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # --- 1. Original Image Analysis ---
    fft, mag, _, mag_log = get_fft(gray_img)
    peak_freqs = get_peaking_frequencies(mag)
    mag_peaks_vis = plot_peaks(mag_log, peak_freqs)

    # --- 2. Pattern Extraction ---
    mask = generate_filter(gray=gray_img, sigma=1.0, peaking_freqs=peak_freqs)
    reconstructed = reconstruct_pattern(fft, mask)

    # --- 3. Uniform Lighting ---
    uniform_img = apply_uniform_lighting(gray_img, reconstructed)

    show_full_analysis(
        gray=gray_img,
        gray_uniform=uniform_img,
        pattern=reconstructed,
        mag_log=mag_log,
        mag_peaks=mag_peaks_vis,
        filter_mask=mask,
        title_prefix="Project3 Image "
    )


if __name__ == "__main__":
    main()
