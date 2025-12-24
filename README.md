# 🍚 Rice Image Classification (PyTorch & TensorFlow)

<p align="center">
  <img src="https://raw.githubusercontent.com/Mirtaheri-ai/Image-Classification/main/Rice%20Image%20Classification/Data/Rice_Image_Dataset/.gitkeep" alt="" width="0" height="0"/>
</p>

<p align="center">
  <a href="https://github.com/Mirtaheri-ai/Image-Classification"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repository-black?logo=github"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python">
  <img alt="Notebook" src="https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter">
  <img alt="DL" src="https://img.shields.io/badge/Deep%20Learning-CNNs-success">
</p>

This repository contains an end-to-end **rice image classification** project implemented in **two tracks**:

- **PyTorch notebook** (training + evaluation)
- **TensorFlow/Keras notebook** (training + evaluation)

The goal is a clean, reproducible pipeline: **data → preprocessing → training → evaluation → inference**.

---

## 📁 Repository layout

```text
.
├── Rice Image Classification/
│   ├── Data/
│   │   └── Rice_Image_Dataset/          # dataset folder (place/download here)
│   ├── rice-image-classification-by-pytorch.ipynb
│   ├── rice-image-classification-by-tensorflow.ipynb
│   └── README.md                         # (optional) local README for this folder
├── LICENSE
└── README.md                             # (this file)
```

> If your notebook filenames are slightly different, just update the names in this README.

---

## 🧠 Dataset

This project expects the rice dataset in:

```text
Rice Image Classification/Data/Rice_Image_Dataset/
```

Recommended directory style (class-per-folder):

```text
Rice_Image_Dataset/
├── Arborio/
├── Basmati/
├── Ipsala/
├── Jasmine/
└── Karacadag/
```

If you are using a Kaggle dataset, download it and extract it into the path above.

---

## ⚙️ Setup

### 1) Create environment

```bash
python -m venv .venv

# Windows:
# .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2) Install dependencies

If you don’t have a `requirements.txt` yet, you can start with this minimal set:

```bash
pip install -U pip
pip install numpy pandas matplotlib scikit-learn pillow opencv-python tqdm
# Choose ONE framework (or install both):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tensorflow
```

> If you are on CPU-only, remove the CUDA PyTorch index-url line and install regular `torch/torchvision`.

---

## ▶️ Run (recommended)

Open one of the notebooks:

- `Rice Image Classification/rice-image-classification-by-pytorch.ipynb`
- `Rice Image Classification/rice-image-classification-by-tensorflow.ipynb`

Then run all cells.

### Optional: run in Google Colab

If you want Colab links, use this pattern:

```text
https://colab.research.google.com/github/Mirtaheri-ai/Image-Classification/blob/main/Rice%20Image%20Classification/<NOTEBOOK_NAME>.ipynb
```

Replace `<NOTEBOOK_NAME>` with the exact notebook filename.

---

## ✅ What the notebooks should do

### Common steps
- Load images from `Data/Rice_Image_Dataset`
- Train/val split
- Preprocessing + resizing (e.g., 224×224)
- Data augmentation (optional)
- Train a CNN (custom or transfer learning)
- Evaluate with:
  - **Accuracy**
  - **Precision / Recall / F1**
  - **Confusion Matrix**
- Save best model/checkpoint (optional)

### Nice-to-have outputs
- Training curves (loss/accuracy)
- Misclassified examples grid
- Inference demo on a few test images

---

## 📊 Results

Add your best metrics here after training:

| Framework | Model | Image Size | Val Accuracy | Val F1 |
|---|---|---:|---:|---:|
| PyTorch | CNN / Transfer | 224 | _fill_ | _fill_ |
| TensorFlow | CNN / Transfer | 224 | _fill_ | _fill_ |

You can also attach a confusion matrix image:

```md
![Confusion Matrix](Rice%20Image%20Classification/assets/confusion_matrix.png)
```

---

## 🔍 Inference (quick idea)

After training, you can add a small inference cell in the notebook:

- Load the saved model
- Read an image
- Apply the same preprocessing
- Predict class + probability

---

## 🧪 Reproducibility tips

- Fix seeds (Python/NumPy/Framework)
- Log hyperparameters (epochs, lr, batch_size)
- Save the best checkpoint
- Keep dataset splits stable (save split indices)

---

## 🧾 License

See `LICENSE`.

---

## 🙌 Acknowledgments

If you use a public rice dataset (e.g., Kaggle), add its reference here and respect its license/terms.
