import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def compress_fft_channel(channel, keep_fraction=0.1):
    fft = np.fft.fft2(channel)
    fft_shifted = np.fft.fftshift(fft)

    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    r_keep = int(rows * keep_fraction / 2)
    c_keep = int(cols * keep_fraction / 2)

    mask = np.zeros_like(channel, dtype=bool)
    mask[crow - r_keep : crow + r_keep, ccol - c_keep : ccol + c_keep] = True

    fft_shifted[~mask] = 0
    fft_inverse = np.fft.ifft2(np.fft.ifftshift(fft_shifted))
    return np.abs(fft_inverse)

def get_file_size(path):
    size_bytes = os.path.getsize(path)
    size_kb = size_bytes / 1024
    return size_bytes, size_kb

def compress_color_image_fft(image_path, keep_fraction=0.1, show=True, save_path="compressed_image.jpg"):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    compressed_channels = []
    for i in range(3):
        channel = img_np[:, :, i]
        compressed = compress_fft_channel(channel, keep_fraction)
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
        plt.show()

    # Salva immagini
    original_save_path = "original_image_temp.jpg"
    img.save(original_save_path)
    compressed_pil.save(save_path)

    # Calcola dimensioni
    original_size, original_kb = get_file_size(original_save_path)
    compressed_size, compressed_kb = get_file_size(save_path)

    print(f"\n📊 Dimensione immagini su disco:")
    print(f" - Originale: {original_size} bytes ({original_kb:.2f} KB)")
    print(f" - Compressa: {compressed_size} bytes ({compressed_kb:.2f} KB)")

    # Facoltativo: elimina file temporaneo
    os.remove(original_save_path)

    return compressed_pil

# ESEMPIO DI USO
print("Comprimo l'immagine originale con keep_fraction=0.1")
compress_color_image_fft("original.jpg", keep_fraction=0.05, save_path="compressed.jpg")
compress_color_image_fft("compressed.jpg", keep_fraction=0.05, save_path="compressed_2.jpg")
