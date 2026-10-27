# EB-UNet: Enhanced Bottleneck-UNet pour Dense Semantic Segmentation & Traffic Occupancy Estimation

Un *complete framework* pour l'urban semantic segmentation (évalué sur les datasets **CamVid** et **BDD100K**) basé sur une *6-stage encoder-decoder hierarchy* et une *hybrid CNN-SSM architecture* (**SSM2D + VMamba**). Ce repository intègre aussi le *morphological post-processing pipeline* ($N=5$) pour une *accurate traffic occupancy estimation*.

---

## 📦 Repository Structure (`BU_Unet`)

- `src/` – *Modular code* (CamVid/BDD100K dataloaders, 6-stage EB-UNet model, metrics, morphological post-processing)
- `scripts/` – Runners pour le training (*from scratch*) et l'evaluation
- `config.yaml` – *Centralized hyperparameters configuration*
- `outputs/` – *Saved weights*, courbes et *visual predictions*

---

## 🚀 Installation & Quick Start

```bash
git clone [https://github.com/ichebon/BU_Unet.git](https://github.com/ichebon/BU_Unet.git)
cd BU_Unet
pip install -r requirements.txt
