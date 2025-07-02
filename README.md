# 🖼️ Image Compression via FFT and Wavelet Transforms

This project provides a simple Python-based interface for compressing color images using two frequency-domain techniques:
- **FFT (Fast Fourier Transform)**
- **DWT (Discrete Wavelet Transform)**

It is designed for educational and experimental use, with support for visualization and side-by-side comparisons. Example usages are provided in the `notebook.ipynb` notebook.

---

## 📁 Project Structure

- `main.py` — Contains the `ImageCompressor` class with all compression and visualization methods.
- `requirements.txt` — List of required Python packages.
- `notebook.ipynb` — Example notebook showing how to use the tool.
- Output directories are created automatically:
  - `spectra_output_*` — FFT spectra (optional)
  - `wavelet_coeffs_*` — Wavelet coefficient visualizations (optional)

---

## ✒️​ Thesis and presentation

This project also provides a thesis and a presentation in order to better understand the mathematical and theoretical operation behind FFT and Wavelet compression. The files are in Italian.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-compression-fft-dwt.git
cd image-compression-fft-dwt
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Example Usage

### Import and Initialize

```python
from main import ImageCompressor

compressor = ImageCompressor(input_dir="Rubik", image_name="orignal.jpg")
```

### ⚡ FFT Compression

```python
compressor.compress_color_image_fft(
    keep_fraction=0.1,
    filter="low-pass",     # or "high-pass"
    show=True,
    save_spectra=True
)
```

### 🌊 Wavelet Compression

```python
compressor.compress_color_image_dwt(
    wavelet="haar",        # or "db1", "bior1.3", etc.
    level=1,
    threshold=20,
    keep_ll_only=True,
    show=True,
    save_coeffs=True
)
```

> See `notebook.ipynb` for other examples.

---

## 📉 Output and Visualization

During compression:
- Compressed images are saved in the specified `input_dir`.
- FFT spectra or wavelet coefficient images are saved if enabled.
- File sizes before and after compression are printed to console.

---

## 🔧 Requirements

Main libraries used:
- `numpy`
- `pywt` (PyWavelets)
- `Pillow`
- `matplotlib`

Everything is listed in the `requirements.txt`.

---

## 🧠 Key Concepts

- **FFT**: Compresses an image by retaining only a fraction of its central frequencies.
- **DWT (Wavelet Transform)**: Provides a compact, localized representation. Optionally keeps only the approximation coefficients (LL).
- Both methods act as **dimensionality reduction** techniques, retaining the most important features with fewer data.

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

---

## 🙋 Contact

For questions, feedback, or collaborations:
- [Enrico Favale] – [enrico.favale@outlook.it]
- GitHub: [@enrico-favale](https://github.com/enrico-favale)
