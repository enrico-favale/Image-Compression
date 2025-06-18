import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def get_file_size(path):
    size_bytes = os.path.getsize(path)
    size_kb = size_bytes / 1024
    return size_bytes, size_kb

class ImageCompressor:
    def __init__(self, input_dir, image_name):
        self.input_dir = input_dir
        self.image_name = image_name
        self.image_path = f"{input_dir}/{image_name}"
    def __compress_fft_channel(self, channel, keep_fraction=0.1, filter="low-pass", save_spectrum=False, channel_name="", output_dir="spectra_output"):
        """
        Effettua compressione FFT su un singolo canale di una immagine. Ad ogni passaggio salva lo spettro delle frequenze.
        
        Parametri
        ---------
        channel : numpy.ndarray => Canale dell'immagine su cui applicare la compressione FFT.
        keep_fraction : float => Frazione di frequenze centrali da mantenere.
        save_spectrum : bool => Se True, salva lo spettro delle frequenze prima e dopo lo shift.
        channel_name : str => Nome del canale per identificare le immagini di output.
        output_dir : str => Percorso in cui salvare gli output spettrali.
        
        Funzionamento
        -------------
        - Applica fft2 al canale
        - Applica shift per centrare le frequenze.
        - Trova il centro dell'immagine e calcola il raggio per mantenere solo una frazione delle frequenze centrali.
        - Mantiene solo le frequenze centrali.
        - Applica ifftshift e ifft2 per ottenere il canale compresso.
        
        Returns
        -------
        numpy.ndarray => Canale compresso.
        
        """
        fft = np.fft.fft2(channel)
        
        if save_spectrum:
            magnitude_before = np.log(np.abs(fft) + 1)
            os.makedirs(output_dir, exist_ok=True)
            plt.imsave(f"{output_dir}/{channel_name}_before_shift.png", magnitude_before, cmap='gray')
        
        fft_shifted = np.fft.fftshift(fft)

        if save_spectrum:
            magnitude_before = np.log(np.abs(fft_shifted) + 1)
            os.makedirs(output_dir, exist_ok=True)
            plt.imsave(f"{output_dir}/{channel_name}_shifted.png", magnitude_before, cmap='gray')

        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        r_keep = int(rows * keep_fraction / 2)
        c_keep = int(cols * keep_fraction / 2)

        mask = np.zeros_like(channel, dtype=bool)
        mask[crow - r_keep : crow + r_keep, ccol - c_keep : ccol + c_keep] = True

        fft_shifted_filtered = fft_shifted.copy()
        if filter == "low-pass": 
            fft_shifted_filtered[~mask] = 0
        else:
            fft_shifted_filtered[mask] = 0

        if save_spectrum:
            magnitude_after = np.log(np.abs(fft_shifted_filtered) + 1)
            plt.imsave(f"{output_dir}/{channel_name}_filtered.png", magnitude_after, cmap='gray')

        fft_inverse = np.fft.ifft2(np.fft.ifftshift(fft_shifted_filtered))
        return np.abs(fft_inverse)

    def compress_color_image_fft(self, keep_fraction=0.1, filter="low-pass", show=True, save_name="compressed.jpg", save_spectra=True):
        """
        Comprime una immagine utilizzando la compressione FFT su ciascun canale mantendendo una frazione delle frequenze.
        
        Parametri
        ---------
        keep_fraction : float => Frazione di frequenze centrali da mantenere.
        filter : str => Tipo di filtro da applicare. "low-pass" o "high-pass".
        show : bool => Se True, mostra l'immagine originale e la versione compressa.
        save_name : str => Nome con cui salvare l'immagine compressa.
        save_spectra : bool => Se True, salva gli spettri delle frequenze prima e dopo la compressione.
        
        Funzionamento
        -------------
        - Carica l'immagine e la porta in array.
        - Comprime ciascun canale (Rosso, Verde, Blu) utilizzando la funzione compress_fft_channel.
        - Combina i canali compressi in un'immagine.
        - Salva l'immagine compressa.
        - Compara le dimensioni dell'immagine originale e compressa.
        
        Return
        ------
        compressed_pil : PIL.Image.Image => Immagine compressa.
        
        """
        
        print(f"Comprimo l'immagine originale con keep_fraction={keep_fraction}")
        
        img = Image.open(self.image_path).convert("RGB")
        img_np = np.array(img)

        compressed_channels = []
        channel_names = ['Rosso', 'Verde', 'Blu']

        for i in range(3):
            channel = img_np[:, :, i]
            compressed = self.__compress_fft_channel(
                channel=channel,
                keep_fraction=keep_fraction,
                filter=filter,
                save_spectrum=save_spectra,
                channel_name=channel_names[i],
                output_dir=f"{self.input_dir}/spectra_output_{filter}"
            )
            compressed = np.clip(compressed, 0, 255)
            compressed_channels.append(compressed.astype(np.uint8))

        compressed_img = np.stack(compressed_channels, axis=2)
        compressed_pil = Image.fromarray(compressed_img)

        if show:
            fig, axs = plt.subplots(1, 2, figsize=(12, 10))
            axs[0].imshow(img)
            axs[0].set_title("Originale")
            axs[0].axis("off")
            axs[1].imshow(compressed_pil)
            axs[1].set_title(f"Compressa (keep {int(keep_fraction*100)}%)")
            axs[1].axis("off")
            plt.tight_layout()
            plt.show()

        original_temp_path = "original_image_temp.jpg"
        img.save(original_temp_path)
        compressed_pil.save(f"{self.input_dir}/{save_name}")

        original_size, original_kb = get_file_size(original_temp_path)
        compressed_size, compressed_kb = get_file_size(f"{self.input_dir}/{save_name}")

        print(f"\n📊 Dimensione immagini su disco:")
        print(f" - Originale: {original_size} bytes ({original_kb:.2f} KB)")
        print(f" - Compressa: {compressed_size} bytes ({compressed_kb:.2f} KB)")

        os.remove(original_temp_path)

        return compressed_pil

# === ESEMPIO DI USO ===
if __name__ == "__main__":
    image_compressor = ImageCompressor(input_dir="Rubick", image_name="original.jpg")
    image_compressor.compress_color_image_fft(
        keep_fraction=0.1,
        filter="low-pass",
        show=True,
        save_name="compressed.jpg",
        save_spectra=True
    )
    
    image_compressor = ImageCompressor(input_dir="Bridge", image_name="original.jpg")
    image_compressor.compress_color_image_fft(
        keep_fraction=0.1,
        filter="low-pass",
        show=True,
        save_name="compressed.jpg",
        save_spectra=True
    )
    
    image_compressor = ImageCompressor(input_dir="Wolf", image_name="original.jpg")
    image_compressor.compress_color_image_fft(
        keep_fraction=0.05,
        filter="low-pass",
        show=True,
        save_name="compressed.jpg",
        save_spectra=True
    )