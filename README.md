# Pneumonia Detection, Medical Report Generation, and Semantic Image Retrieval (Tasks 1–3)

## Overview
This repository contains a three-part medical imaging pipeline built on chest X-ray images (PneumoniaMNIST):

- **Task 1:** Deep learning classification of pneumonia (comparative study)
- **Task 2:** Medical report generation using a Visual Language Model (MedGemma)
- **Task 3:** Semantic image retrieval (CBIR) using medical embeddings + FAISS

The project demonstrates end-to-end competence across **classification**, **multimodal generation**, and **embedding-based retrieval**, which are key components in modern medical AI applications.

---

## Task 1 – Pneumonia Detection (CNN / Transformer / Mamba)
### Goal
Train and evaluate multiple deep learning models for binary pneumonia detection with rigorous experimental methodology.

### Models
- Simple CNN
- ResNet-18
- EfficientNet-B0
- ViT-Tiny
- MambaNet

### Metrics
Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curves, and failure-case analysis.

**Outcome:** ViT-Tiny achieved the best overall discrimination performance and was used as a reference model for subsequent tasks.

---

## Task 2 – Medical Report Generation (VLM)
### Goal
Generate short radiology-style impressions from chest X-ray images using an open-source medical Visual Language Model.

### Model
- **MedGemma 1.5** (`google/medgemma-1.5-4b-it`) with prompt engineering
- Qualitative comparison against Ground Truth and ViT predictions, including difficult and misclassified cases.

Outputs include:
- Prompt strategy documentation
- Sample generated reports (≥10 images)
- Task-2 markdown report

---

## Task 3 – Semantic Image Retrieval (CBIR)
### Goal
Build a semantic image retrieval system using medical embeddings and a vector database/index.

### Features
- **Image→Image retrieval:** query X-ray → top-k similar X-rays
- **Text→Image retrieval:** clinical prompt → relevant images (CLIP-style shared embedding space)
- **Vector search:** FAISS (`IndexFlatIP`) with cosine similarity
- **Evaluation:** Precision@k for k ∈ {1,3,5,10}
- **Visualization:** query + retrieved results saved as figures





## Reproducibility
- Python, PyTorch, Hugging Face Transformers
- Task-2 requires Hugging Face access to MedGemma (public gated model)
- All experiments are reproducible using the provided scripts

---

## Disclaimer
This project is intended for academic and research purposes only.  
Generated predictions and reports are **not suitable for clinical diagnosis**.

---

## Author
**Bushara A. R.**  
Deep Learning | Medical Image Analysis | Explainable & Multimodal AI
