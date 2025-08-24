# SEA-ViT: Sea Surface Currents Forecasting 🌊🚀

<!-- ![SEA-ViT Logo](https://example.com/sea-vit-logo.png) -->

Welcome to the official repository for **SEA-ViT** — a cutting-edge deep learning model designed for predicting sea surface currents using Vision Transformers and bidirectional Gated Recurrent Units. This repository contains code, pretrained models, and documentation for exploring our innovative approach to ocean forecasting.

## 📚 Overview

**SEA-ViT** integrates the Vision Transformer (ViT) architecture with bidirectional Gated Recurrent Units (GRUs) to forecast sea surface currents (U, V). Developed by **Teerapong Panboonyuen**, this model leverages high-frequency radar (HF) data to capture spatio-temporal dependencies and provide accurate predictions for maritime navigation, environmental monitoring, and climate analysis.

**Key Contributions:**
- **Vision Transformer Integration:** Utilizes ViT to capture complex spatial features in sea surface data.
- **Bidirectional GRUs:** Enhances temporal sequence learning to improve forecasting accuracy.
- **Long-Term Data Utilization:** Trained on a comprehensive 30-year dataset, including ENSO indices to account for climatic variations.

For a detailed discussion on the model and methodology, visit [SEA-ViT Overview](https://kaopanboonyuen.github.io/blog/2024-09-15-sea-vit-sea-surface-currents-forecasting-with-vision-transformers-and-grus/).

## 🔬 Key Features

- **High-Precision Forecasting:** Combines ViT and GRU technologies for superior accuracy in sea surface current predictions.
- **Long-Term Data Integration:** Incorporates a rich dataset spanning over 30 years, including ENSO indices.
- **Enhanced Capabilities for Thailand:** Tailored for maritime regions such as the Gulf of Thailand and the Andaman Sea.

## 📈 Results

Our experiments demonstrate SEA-ViT's efficacy in forecasting sea surface currents with high precision. The model is well-suited for applications requiring detailed and accurate oceanographic predictions.

**Results Summary:**
- **Model Performance:** Proven accuracy in capturing complex spatio-temporal patterns in sea surface data.

## 📥 Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/kaopanboonyuen/gistda-ai-sea-surface-currents.git
cd gistda-ai-sea-surface-currents
pip install -r requirements.txt
```

## 🚀 Usage

Detailed usage instructions and example code are provided in the `docs` directory. For questions or contributions, please refer to the [contributing guidelines](CONTRIBUTING.md).

## 📄 Citation

If you use SEA-ViT in your research, please cite our work:

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

## 📫 Contact

For further inquiries, reach out to:

- **Teerapong Panboonyuen**  
  Postdoctoral Researcher, Chulalongkorn University  
  Senior Research Scientist, MARS (Motor AI Recognition Solution)  
  Email: [teerapong.panboonyuen@gmail.com](mailto:teerapong.panboonyuen@gmail.com)

---

Thank you for visiting our repository! We hope SEA-ViT proves valuable in advancing sea surface current forecasting and oceanographic research.