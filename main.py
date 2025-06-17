import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def compress_fft_channel(channel, keep_fraction=0.1, save_spectrum=False, channel_name="", output_dir="spectra_output"):
    fft = np.fft.fft2(channel)
    
    if save_spectrum:
        magnitude_before = np.log(np.abs(fft) + 1)
        os.makedirs(output_dir, exist_ok=True)
        plt.imsave(f"{output_dir}/{channel_name}_before_shift.png", magnitude_before, cmap='gray')
    
    fft_shifted = np.fft.fftshift(fft)

    if save_spectrum:
        magnitude_before = np.log(np.abs(fft_shifted) + 1)
        os.makedirs(output_dir, exist_ok=True)
        plt.imsave(f"{output_dir}/{channel_name}_after_shift.png", magnitude_before, cmap='gray')

    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    r_keep = int(rows * keep_fraction / 2)
    c_keep = int(cols * keep_fraction / 2)

    mask = np.zeros_like(channel, dtype=bool)
    mask[crow - r_keep : crow + r_keep, ccol - c_keep : ccol + c_keep] = True

    fft_shifted_filtered = fft_shifted.copy()
    fft_shifted_filtered[~mask] = 0

    if save_spectrum:
        magnitude_after = np.log(np.abs(fft_shifted_filtered) + 1)
        plt.imsave(f"{output_dir}/{channel_name}_after_filter.png", magnitude_after, cmap='gray')

    fft_inverse = np.fft.ifft2(np.fft.ifftshift(fft_shifted_filtered))
    return np.abs(fft_inverse)

def get_file_size(path):
    size_bytes = os.path.getsize(path)
    size_kb = size_bytes / 1024
    return size_bytes, size_kb

def compress_color_image_fft(image_path, keep_fraction=0.1, show=True, save_path="compressed_image.jpg", save_spectra=True):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    compressed_channels = []
    channel_names = ['Rosso', 'Verde', 'Blu']

    for i in range(3):
        channel = img_np[:, :, i]
        compressed = compress_fft_channel(
            channel,
            keep_fraction,
            save_spectrum=save_spectra,
            channel_name=channel_names[i]
        )
        compressed = np.clip(compressed, 0, 255)
        compressed_channels.append(compressed.astype(np.uint8))

    compressed_img = np.stack(compressed_channels, axis=2)
    compressed_pil = Image.fromarray(compressed_img)

    if show:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img)
        axs[0].set_title("Originale")
        axs[0].axis("off")
        axs[1].imshow(compressed_pil)
        axs[1].set_title(f"Compressa (keep {int(keep_fraction*100)}%)")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()

    # Salva immagini su disco
    original_temp_path = "original_image_temp.jpg"
    img.save(original_temp_path)
    compressed_pil.save(save_path)

    original_size, original_kb = get_file_size(original_temp_path)
    compressed_size, compressed_kb = get_file_size(save_path)

    print(f"\n📊 Dimensione immagini su disco:")
    print(f" - Originale: {original_size} bytes ({original_kb:.2f} KB)")
    print(f" - Compressa: {compressed_size} bytes ({compressed_kb:.2f} KB)")

    os.remove(original_temp_path)

    return compressed_pil

# === ESEMPIO DI USO ===
keep_fraction = 0.05  # Tieni solo il 5% delle frequenze centrali
print(f"Comprimo l'immagine originale con keep_fraction={keep_fraction}")
compress_color_image_fft(
    "original.jpg",
    keep_fraction=keep_fraction,
    save_path="compressed.jpg",
    save_spectra=True  # Salva gli spettri
)
