# llm-detection

This project proposes using an author attribution system for LLM detection. The system has the following structure: it mean-pools LLM sentence embeddings during training and then computes cosine similarity with test sentence embeddings to draw conclusions. The project employs this naive system to perform three failure case analyses in `llm_detection_contribution1.ipynb` and finds that this system fails when prompt engineering is involved and shifts the LLM's own style. The project then extends the system to include a mean-pooled human representation and projects all embeddings to two dimensions using PCA, using Euclidean distance as a similarity measure (`llm_detection_contribution2.ipynb`). This improvement enables the system to achieve 74% accuracy, even in use cases where prompt engineering is involved.

Figure 1: Naive Proposed LLM-Detection System:

![proposed system](https://github.com/jjz5463/llm-detection/assets/47905800/7f0f91b4-8a47-43cc-b0f2-e8255a38ee8c)

Figure 2: Extended LLM-Detection System to Generalize to Prompt-Engineered Text:

![Extended proposed system](https://github.com/jjz5463/llm-detection/assets/47905800/ce1edf48-3036-41f6-890f-8ea9fe3c1afd)
