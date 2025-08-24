# 🌊 SEA-ViT: Sea Surface Currents Forecasting with Vision Transformers and GRUs

> Official repository for:
> **"SEA-ViT: Forecasting Sea Surface Currents Using a Vision Transformer and GRU-Based Spatio-Temporal Covariance Model"**
> 📌 Accepted at **IEEE KST 2025** | 🔬 Developed by **Teerapong Panboonyuen (Kao)**

---

## 🚀 Highlights

* ✅ **Vision Transformer + BiGRU** for capturing spatio-temporal ocean dynamics.
* 🌐 **Forecasts U/V currents** from HF-radar & ENSO time series.
* 📦 **Modular PyTorch Code** for fast deployment and experimentation.
* 🧠 **30-Year Dataset Ready** — built for long-term environmental forecasting.

---

## 🗂️ Project Structure

```bash
sea-vit/
├── src/
│   ├── model.py         # SEA-ViT Model: ViT + BiGRU
│   ├── train.py         # Training Pipeline
│   ├── inference.py     # Run Inference on Radar Sequences
│   ├── metrics.py       # Evaluation (RMSE, MAE, Corr)
│   └── utils.py         # Dataset I/O, Preprocessing, Logging
├── Dockerfile           # Reproducible Environment
├── requirements.txt     # Python Dependencies
├── README.md            # Project Overview (this file)
└── data/                # Example .npz radar data (optional)
```

---

## 📥 Installation

Install dependencies via pip:

```bash
git clone https://github.com/kaopanboonyuen/GISTDA-SeaViT.git
cd GISTDA-SeaViT
pip install -r requirements.txt
```

Or build the full environment via Docker:

```bash
docker build -t sea-vit .
```

---

## 💻 How to Run the Code

### 🔧 1. Train SEA-ViT on your dataset

Assuming your `.npz` dataset includes radar sequences + targets:

```bash
python src/train.py \
  --data-path ./data/train.npz \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4
```

> Input shape: `[N, T, 2, H, W]` for sequences
> Target shape: `[N, forecast_steps, 2]` for (U, V)

---

### 🧠 2. Inference on a New Sequence

Run forecasting on unseen radar input:

```bash
python src/inference.py \
  --model-path ./checkpoints/sea-vit.pth \
  --input-path ./data/sample_input.npz \
  --output-path ./output/forecast.npy
```

---

### 📏 3. Evaluate Metrics (RMSE, MAE, Correlation)

```bash
python src/metrics.py \
  --predictions ./output/forecast.npy \
  --ground-truth ./data/sample_ground_truth.npy
```

---

## 📈 Results Summary

| Metric | Value (Example) |
| ------ | --------------- |
| RMSE   | **0.087**       |
| MAE    | **0.065**       |
| Corr   | **0.91**        |

SEA-ViT demonstrates high accuracy across diverse oceanic conditions, outperforming CNN-GRU and ConvLSTM baselines.

---

## 📚 Learn More

* 📖 **Blog**: [SEA-ViT Overview](https://kaopanboonyuen.github.io/blog/2024-09-15-sea-vit-sea-surface-currents-forecasting-with-vision-transformers-and-grus/)
* 📄 **Paper**: [arXiv](https://arxiv.org/abs/2409.16313)
* 📄 **Paper**: [IEEE](https://ieeexplore.ieee.org/document/11003320/)

---

## 🔬 Citation

```bibtex
@inproceedings{panboonyuen2025sea,
  title={SEA-ViT: Forecasting Sea Surface Currents Using a Vision Transformer and GRU-Based Spatio-Temporal Covariance Model},
  author={Panboonyuen, Teerapong},
  booktitle={2025 17th International Conference on Knowledge and Smart Technology (KST)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```

---

## 📫 Contact

**Teerapong Panboonyuen (Kao)**
Postdoctoral Researcher, Chulalongkorn University
Senior Research Scientist, MARSAIL
📧 [teerapong.panboonyuen@gmail.com](mailto:teerapong.panboonyuen@gmail.com)

---

## 🤝 Contributing

Pull requests welcome! For feature ideas or bug fixes, feel free to open an issue.

---

> “SEA-ViT doesn’t just see the ocean — it *foresees* it.” 🌊📡

---