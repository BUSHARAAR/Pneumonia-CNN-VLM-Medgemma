# Task 3 – Semantic Image Retrieval System (CBIR) with Medical Embeddings + FAISS

## Objective
This task implements a content-based image retrieval (CBIR) system for chest X-ray images using medically meaningful embeddings and a vector search index. The system enables:
- **Image-to-image retrieval:** find visually similar X-rays given a query image
- **Text-to-image retrieval:** retrieve relevant images given a clinical text description (if supported by the embedding model)

## Embedding Model
- **Embedding backbone:** Medical CLIP-style encoder (CLIPModel medical variant)
- **Rationale:** CLIP-style medical models map **images and clinical text** into a shared embedding space, enabling both retrieval modes and capturing medically relevant patterns beyond raw pixel similarity.

**Embedding details**
- Image embeddings extracted from the vision encoder and projected into the shared embedding space
- Text embeddings extracted from the text encoder and projected into the same space
- Embeddings are **L2-normalized** to support cosine similarity search via inner product

## Vector Index (FAISS)
- **Library:** FAISS
- **Index type:** `IndexFlatIP` (inner product)
- **Similarity:** Cosine similarity (because vectors are normalized)
- Stored artifacts (for reproducibility):
  - `task3_outputs/embeddings.npy`
  - `task3_outputs/labels.npy`
  - `task3_outputs/faiss_index.bin`
  - `task3_outputs/retrieval_figures/`
  - `task3_outputs/task3 retrieval system.md`

## Retrieval Interfaces
### 1) Image-to-Image Search
Given a query image (by dataset index), the system retrieves top-k nearest neighbors from the FAISS index.

### 2) Text-to-Image Search
Given a text prompt (e.g., “pneumonia opacity in lungs”), the system embeds the text and retrieves the most relevant images from the same index.

## Evaluation
Retrieval quality is evaluated using **Precision@k**:
- For each query image, compute the fraction of retrieved top-k images sharing the same label as the query.
- Report Precision@k for k ∈ {1, 3, 5, 10}.

## Visualizations
Retrieval figures are saved under:
- `task3_outputs/retrieval_figures/`

Each visualization shows:
- Query image (or text prompt)
- Top-k retrieved images with ground-truth labels and similarity scores

## Notes / Limitations
- PneumoniaMNIST images are low-resolution (28×28), which can limit fine-grained radiological similarity.
- Precision@k uses binary labels; it measures class consistency, not full clinical equivalence.
- The system is intended for research and educational purposes only (not for clinical diagnosis).

