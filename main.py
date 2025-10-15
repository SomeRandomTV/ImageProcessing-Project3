import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np

def get_fft(gray: np.ndarray):
    """Compute the centered FFT, magnitude, and phase of a grayscale image."""
    f = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f)

    # Magnitude and phase
    magnitude = cv.magnitude(f_shift[:, :, 0], f_shift[:, :, 1])
    magnitude_log = np.log1p(magnitude)  # for visualization
    phase = cv.phase(f_shift[:, :, 0], f_shift[:, :, 1])

    return f_shift, magnitude_log, phase


def get_peaking_frequencies(mag: np.ndarray, num_peaks: int = 45,
                                   exclude_center: int = 20) -> list:
    """
    Detect top N peaking frequencies (simpler alternative).
    
    Args:
        mag (np.ndarray): Magnitude spectrum (should be fft-shifted, centered).
        num_peaks (int): Number of top peaks to return (default: 10).
        exclude_center (int): Radius around DC component to exclude (default: 20).
    
    Returns:
        list: List of (u, v) tuples for top N peaks relative to center.
    """
    rows, cols = mag.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a copy and exclude center
    mag_copy = mag.copy()
    Y, X = np.ogrid[:rows, :cols]
    center_mask = np.sqrt((X - ccol)**2 + (Y - crow)**2) < exclude_center
    mag_copy[center_mask] = 0
    
    # Flatten and get top N indices
    flat_indices = np.argsort(mag_copy.ravel())[-num_peaks:]
    
    # Convert back to 2D coordinates
    peak_coords = np.unravel_index(flat_indices, mag_copy.shape)
    
    # Convert to relative coordinates
    relative_coords = []
    for y, x in zip(peak_coords[0], peak_coords[1]):
        u = x - ccol
        v = y - crow
        relative_coords.append((u, v))
    
    return relative_coords
    



def generate_filter(gray: np.ndarray, r_in: float = 15, r_out: float = 30, 
                    filter_type: str = 'L', notch_coords: list = None, 
                    notch_radius: float = 10) -> np.ndarray:
    """
    Create a frequency-domain filter (Gaussian, low-pass, high-pass, band-pass, or notch).

    Args:
        gray (np.ndarray): Grayscale image (used for dimensions).
        r_in (float): Inner radius (low-frequency cutoff).
        r_out (float): Outer radius (high-frequency cutoff).
        filter_type (str): 
            'L' = Low-pass (keeps low frequencies)
            'H' = High-pass (keeps high frequencies)
            'B' = Band-pass (keeps mid frequencies)
            'G' = Gaussian band-pass (default)
            'N' = Notch filter (removes specific frequencies)
            'NR' = Notch-reject (keeps ONLY notch frequencies, removes all others)
        notch_coords (list): List of (u, v) tuples for notch filter centers.
                            Coordinates relative to spectrum center.
                            E.g., [(50, 0), (-50, 0)] for symmetric horizontal notches
        notch_radius (float): Radius of each notch (default: 10).

    Returns:
        np.ndarray: 2D filter mask (float32).
    """
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Create coordinate grid centered at the spectrum origin
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)

    # Default Gaussian mask
    mask = np.ones_like(distance, dtype=np.float32)

    if filter_type == 'L':
        # Gaussian low-pass
        mask = np.exp(-(distance**2) / (2 * (r_out**2)))

    elif filter_type == 'H':
        # Gaussian high-pass
        mask = 1 - np.exp(-(distance**2) / (2 * (r_in**2)))

    elif filter_type == 'B' or filter_type == 'G':
        # Band-pass (Gaussian)
        low_pass = np.exp(-(distance**2) / (2 * (r_out**2)))
        high_pass = 1 - np.exp(-(distance**2) / (2 * (r_in**2)))
        mask = high_pass * low_pass

    elif filter_type == 'N':
        # Notch filter - removes specific frequency peaks
        if notch_coords is None:
            raise ValueError("notch_coords must be provided for notch filter")
        
        # Start with all frequencies passed (mask = 1)
        mask = np.ones_like(distance, dtype=np.float32)
        
        # Apply notches at specified coordinates
        for (u, v) in notch_coords:
            # Calculate distance from this notch center
            notch_dist = np.sqrt((X - (ccol + u))**2 + (Y - (crow + v))**2)
            # Create Gaussian notch (0 at center, 1 far away)
            notch = 1 - np.exp(-(notch_dist**2) / (2 * (notch_radius**2)))
            # Multiply with existing mask
            mask *= notch

    elif filter_type == 'NR':
        # Notch-reject filter - keeps ONLY specified frequencies, removes all others
        if notch_coords is None:
            raise ValueError("notch_coords must be provided for notch-reject filter")
        
        # Start with all frequencies blocked (mask = 0)
        mask = np.zeros_like(distance, dtype=np.float32)
        
        # Add passes at specified coordinates
        for (u, v) in notch_coords:
            # Calculate distance from this notch center
            notch_dist = np.sqrt((X - (ccol + u))**2 + (Y - (crow + v))**2)
            # Create Gaussian pass (1 at center, 0 far away)
            notch_pass = np.exp(-(notch_dist**2) / (2 * (notch_radius**2)))
            # Add to existing mask (logical OR)
            mask = np.maximum(mask, notch_pass)

    else:
        raise ValueError(f"Unknown filter type '{filter_type}'. Use 'L', 'H', 'B', 'G', 'N', or 'NR'.")

    return mask.astype(np.float32)


def extract_pattern(gray: np.ndarray, fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract periodic pattern using a frequency mask.
    """
    # Ensure mask is float type
    mask = mask.astype(np.float32)

    # Apply mask to complex FFT (convert 2-channel OpenCV FFT to complex)
    fft_complex = fft[:, :, 0] + 1j * fft[:, :, 1]
    fft_masked = fft_complex * mask

    # Inverse shift and inverse FFT to reconstruct periodic component
    fft_ishift = np.fft.ifftshift(fft_masked)
    pattern_complex = np.fft.ifft2(fft_ishift)

    # Take real part
    pattern = np.real(pattern_complex)

    # Normalize for visualization
    pattern_norm = cv.normalize(pattern, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    return pattern_norm


    


def reconstruct_image(filtered_fft: np.ndarray) -> np.ndarray:
    """Reconstruct spatial image from filtered FFT."""
    complex_fft = filtered_fft[:, :, 0] + 1j * filtered_fft[:, :, 1]
    f_ishift = np.fft.ifftshift(complex_fft)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return img_back


def show_image(working_img: np.ndarray, fft: np.ndarray, mag: np.ndarray, phase: np.ndarray, filter: np.ndarray = None):
    """
    Display the original image, FFT magnitude, phase, and optional filter.
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    # Working image
    axs[0, 0].imshow(working_img, cmap="gray")
    axs[0, 0].set_title("Working Image")
    axs[0, 0].axis("off")

    # FFT magnitude visualization
    fft_vis = np.log1p(cv.magnitude(fft[:, :, 0], fft[:, :, 1]))
    axs[0, 1].imshow(fft_vis, cmap="magma")
    axs[0, 1].set_title("Fourier Transform (Shifted Log Magnitude)")
    axs[0, 1].axis("off")

    # Magnitude
    axs[0, 2].imshow(mag, cmap="magma")
    axs[0, 2].set_title("Magnitude Spectrum")
    axs[0, 2].axis("off")

    # Phase
    axs[1, 0].imshow(phase, cmap="twilight")
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
    
    peaking_freqs = get_peaking_frequencies(mag=mag)
    
    for peak in peaking_freqs:
        print(f"Found peaking freq at: {peak}")
   
    
    # after filtering 
    # -------------------- Filtering -------------------- #
    mask = generate_filter(gray_img, filter_type='NR' ,notch_coords=peaking_freqs)  # example radius
    extracted_pattern = extract_pattern(gray=gray_img, fft=fft, mask=mask)
    # after you extract the pattern
    print("Error here V")
    pattern_fft, pattern_mag, pattern_phase = get_fft(extracted_pattern)
    show_image(extracted_pattern, pattern_fft, pattern_mag, phase=pattern_phase, filter=mask)

    


if __name__ == "__main__":
    main()
