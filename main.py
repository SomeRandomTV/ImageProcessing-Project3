import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np

# -------------------- FFT and Filtering -------------------- #
def get_fft(gray: np.ndarray):
    """Compute the centered FFT, magnitude, and phase of a grayscale image."""
    f = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f)

    # Magnitude and phase
    magnitude = cv.magnitude(f_shift[:, :, 0], f_shift[:, :, 1])
    magnitude_log = np.log1p(magnitude)  # for visualization
    phase = cv.phase(f_shift[:, :, 0], f_shift[:, :, 1])

    return f_shift, magnitude_log, phase


def generate_filter(gray: np.ndarray, r_in: int = 10, r_out: int = 30) -> np.ndarray:
    """Create a circular band-pass mask."""
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    mask = np.logical_and(distance >= r_in, distance <= r_out).astype(np.uint8)
    return mask


def apply_filter(fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to both channels of FFT."""
    return fft * mask[:, :, np.newaxis]


def reconstruct_image(filtered_fft: np.ndarray) -> np.ndarray:
    """Reconstruct spatial image from filtered FFT."""
    complex_fft = filtered_fft[:, :, 0] + 1j * filtered_fft[:, :, 1]
    f_ishift = np.fft.ifftshift(complex_fft)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return img_back


# -------------------- Visualization -------------------- #
def show_image(working_img: np.ndarray, fft: np.ndarray, mag: np.ndarray, phase: np.ndarray, filter: np.ndarray = None):
    """
    Display the original image, FFT, magnitude, phase, and optional filter.
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    # Working image
    axs[0, 0].imshow(working_img, cmap="gray")
    axs[0, 0].set_title("Working Image")
    axs[0, 0].axis("off")

    # FFT visualization (log magnitude of FFT)
    fft_vis = np.log1p(cv.magnitude(fft[:, :, 0], fft[:, :, 1]))
    axs[0, 1].imshow(fft_vis, cmap="viridis")
    axs[0, 1].set_title("Fourier Transform (Shifted + Log Magnitude)")
    axs[0, 1].axis("off")

    # Magnitude
    axs[0, 2].imshow(mag, cmap="viridis")
    axs[0, 2].set_title("Magnitude Spectrum")
    axs[0, 2].axis("off")

    # Phase
    axs[1, 0].imshow(phase, cmap="viridis")
    axs[1, 0].set_title("Phase / Angle")
    axs[1, 0].axis("off")

    # Optional filter
    if filter is not None:
        axs[1, 1].imshow(filter, cmap="gray")
        axs[1, 1].set_title("Filter / Mask")
        axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


# -------------------- Main -------------------- #
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_file>")
        sys.exit(1)

    img = cv.imread(sys.argv[1])
    if img is None:
        sys.exit("Could not read the image.")

    # initial checks
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fft, mag, phase = get_fft(gray_img)
    show_image(gray_img, fft, mag, phase)
    
    # after filtering 
    # -------------------- Filtering -------------------- #
    mask = generate_filter(gray_img, r_in=10, r_out=25)  # example radius
    filtered_fft = apply_filter(fft, mask)

    # Compute filtered magnitude and phase
    filtered_mag = cv.magnitude(filtered_fft[:, :, 0], filtered_fft[:, :, 1])
    filtered_mag = np.log1p(filtered_mag)
    filtered_phase = cv.phase(filtered_fft[:, :, 0], filtered_fft[:, :, 1])

    # Reconstruct spatial image
    reconst_img = reconstruct_image(filtered_fft)

    # Show filtered results
    show_image(working_img=reconst_img, fft=filtered_fft, mag=filtered_mag, phase=filtered_phase, filter=mask)



if __name__ == "__main__":
    main()
