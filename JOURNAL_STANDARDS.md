# Journal-Level Research Standards: Metrics & Parameters Checklist

This document outlines the standard metrics and parameters required for high-quality, publication-ready research papers in computer vision.

## 1. General Reporting Standards (Mandatory for All Types)
*   **Statistical Significance:** Report mean ± standard deviation over multiple runs (e.g., 3-5 random seeds).
*   **Computational Complexity:**
    *   **Parameters:** Trainable vs Non-trainable count.
    *   **FLOPS/MACs:** Floating Point Operations per second (measure of compute heaviness).
    *   **Model Size:** specific storage size (MB).
*   **Performance Benchmarks:**
    *   **Inference Latency:** Average time (ms) per image on target hardware (CPU/GPU/Edge).
    *   **Throughput:** Images per second (IPS) or Frames per second (FPS).
    *   **Training Time:** Total hours/epochs to convergence.
*   **Environment Specs:** Exact CPU/GPU models, RAM, Driver/CUDA versions, Framework versions.

---

## 2. Task-Specific Metrics

### A. Image Classification
*   **Primary Metrics:**
    *   **Top-1 Accuracy:** The standard correctness metric.
    *   **Top-5 Accuracy:** (For datasets with >10 classes) Did the right class appear in the top 5 guesses?
*   **Robustness Metrics (Crucial for imbalanced data):**
    *   **Macro-F1 Score:** Harmonic mean of precision and recall (balanced across classes).
    *   **Weighted F1 Score:** F1 score weighted by class support.
    *   **Precision & Recall:** Per-class breakdown.
*   **Visualizations:**
    *   **Confusion Matrix:** Normalized (percentages) and raw counts.
    *   **ROC Curves & AUC:** Sensitivity vs Specificity analysis.
    *   **t-SNE/PCA:** 2D visualization of high-dimensional feature embeddings.
    *   **Grad-CAM / Saliency Maps:** Heatmaps showing *where* the model is looking.

### B. Semantic Segmentation
*   **Primary Metrics:**
    *   **Mean Intersection over Union (mIoU):** The standard "accuracy" for segmentation.
*   **Secondary Metrics:**
    *   **Dice Coefficient:** (F1 Score for pixels) Better for small objects/imbalanced classes.
    *   **Pixel Accuracy:** Global % of correct pixels (can be misleading if background dominates).
    *   **Frequency Weighted IoU.**
*   **Visualizations:**
    *   Ground Truth vs Prediction overlays.
    *   Error Maps (highlighting pixels where prediction != truth).

### C. Object Detection
*   **Primary Metrics (COCO Standards):**
    *   **mAP@[0.5:0.95]:** Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 (Step 0.05). **This is the gold standard.**
*   **Secondary Metrics:**
    *   **mAP@0.5:** (PASCAL VOC standard) Less strict, good for comparison with older papers.
    *   **Average Recall (AR):** Max recall given a fixed number of detections per image.
*   **Visualizations:**
    *   Precision-Recall Curves.
    *   Bounding box qualitative examples (Best case, Worst case, Occlusion cases).

---

## 3. Recommended Ablation Studies
A journal paper must explain *why* the model works, not just *that* it works.

*   **Component Analysis:** "Leave-one-out" testing (remove one new feature at a time to prove its value).
*   **Hyperparameter Sensitivity:**
    *   Effect of **Batch Size** on convergence/generalization.
    *   Effect of **Learning Rate** schedules.
    *   Effect of **Input Resolution**.
*   **Robustness Tests:**
    *   **Noise Sensitivity:** Performance drop vs SNR (Signal-to-Noise Ratio).
    *   **Data Scarcity:** Performance with 10%, 50%, 100% of training data.
    *   **OOD (Out-of-Distribution):** Performance on "unseen" or slightly different dataset domains.
