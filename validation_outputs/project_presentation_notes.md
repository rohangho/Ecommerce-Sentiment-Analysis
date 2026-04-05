# E-Commerce Sentiment Analysis — Project Presentation Notes

> Use these bullet points as your slide-by-slide talking points for the professor presentation.

---

## Slide 1 — Project Overview

**Title:** E-Commerce Sentiment & Aspect-Based Analysis Pipeline

- Built a **production-ready, end-to-end NLP pipeline** for analyzing customer reviews from an Amazon e-commerce dataset
- The system predicts **3-class sentiment** (Positive / Neutral / Negative) AND identifies the **product aspect** being discussed (e.g., "battery life", "sound quality", "display")
- Covers the full ML lifecycle: data cleaning → multi-model training → evaluation → web deployment → automated retraining

---

## Slide 2 — Dataset & Class Imbalance Challenge

- **Source:** Amazon product reviews (tablets, Kindle, Echo, etc.)
- **Training set:** 4,003 rows
- **Severe class imbalance discovered:**

| Class | Count | % of Data |
|---|---|---|
| Positive | 3,751 | 93.7% |
| Neutral | 158 | 3.9% |
| Negative | 94 | 2.3% |

> [!IMPORTANT]
> This imbalance was the root cause of most model failures. Addressed using **balanced class weights** during training (`compute_class_weight` from sklearn applied to all Transformer models).

---

## Slide 3 — Multi-Model Sentiment Analysis (7 Models Trained)

Built and evaluated **7 sentiment analysis models** in a unified training pipeline (`train_and_validate.py`):

| # | Model | Type |
|---|---|---|
| 1 | **Logistic Regression** | Classical ML (TF-IDF bigrams + categorical features) |
| 2 | **SVM** | Classical ML (RBF kernel, `class_weight='balanced'`) |
| 3 | **RNN** | Deep Learning (Keras, LSTM, EarlyStopping + ReduceLR) |
| 4 | **DistilBERT** | Transformer (distilbert-base-uncased, fine-tuned) |
| 5 | **BERT** | Transformer (bert-base-uncased, fine-tuned) |
| 6 | **RoBERTa** | Transformer (siebert backbone, 3-class head) |
| 7 | **DeBERTa** | Transformer (microsoft/deberta-v3-base) ✅ **Final Model** |

- All models output **Negative / Neutral / Positive**
- Validated on a **held-out hidden test set** (`test_data_hidden.csv`)
- Comparison plots generated: confusion matrices, F1 heatmaps, accuracy bar charts

---

## Slide 4 — Primary Model: DeBERTa Fine-Tuning

**Why DeBERTa?**
- Microsoft's `deberta-v3-base` uses **disentangled attention** — handles word position and content separately → better contextual understanding than BERT/RoBERTa

**Fine-tuning approach:**
- Added a fresh **3-class classification head** (`ignore_mismatched_sizes=True`)
- 3 epochs, `lr=2e-5`, 12% warmup, `weight_decay=0.02`
- **Balanced class weights** to counter the 40:1 Positive-to-Negative ratio
- Apple M1 Max GPU used via **TensorFlow Metal** plugin

**Saved as:** `deploy_models/sentiment_model_deberta_deberta_tf/` (safetensors format)

---

## Slide 5 — Problem Discovered & Confidence-Based Fix

> [!WARNING]
> **Root cause identified:** The word "sucks" had **0 occurrences** in the training set. Short colloquial phrases like "the battery sucks" returned Neutral instead of Negative because the model had never learned this vocabulary.

**Solution — Confidence-Gated Dual-Model System:**

```
User Input
    ↓
DeBERTa (fine-tuned)
    ↓
Softmax confidence > 0.65?
   YES → Trust DeBERTa result
    NO → Fallback: Cardiff RoBERTa
              (cardiffnlp/twitter-roberta-base-sentiment)
              Trained on Twitter/social media (slang-aware)
```

- **Cardiff RoBERTa** was downloaded once and cached locally in `deploy_models/cardiffnlp_fallback/` — **no internet needed at container startup**
- "the battery sucks" → DeBERTa confidence ~0.50 → fallback triggered → **Negative** ✅

---

## Slide 6 — Aspect-Based Analysis: Guided BERTopic & Multi-Aspect Extraction

**Goal:** Identify *what specific product features* (aspects) the review discusses (e.g., RAM, Battery, Display).

**Previous Problem:** BERTopic assigned only **one** generic cluster per review, filled with irrelevant demographic words (e.g., "teenage", "daughter", "beautifully"). Reviews like *"ram is good but display is bad"* only returned "memory, downloading".

**Solution — Hybrid Multi-Aspect Architecture:**

1. **Externalized Configuration (`seed_topics.json`)**
   - Extracted all product aspect definitions into a standalone JSON file.
   - **Why?** Allows instant addition/editing of product aspects without altering Python code or rebuilding Docker containers.

2. **Guided BERTopic Retraining**
   - Transitioned from unsupervised clustering to **Semi-Supervised/Guided BERTopic**.
   - `train_bertopic.py` dynamically loads `seed_topics.json` and forces the UMAP/HDBSCAN pipeline to cluster around defined product features yielding much cleaner latent boundaries.

3. **Multi-Aspect Keyword Extractor (`app.py`)**
   - Replaced single-topic extraction with a multi-match engine.
   - Scans text against `seed_topics.json`. If a user mentions "RAM", "display", and "SSD", the system returns **all three** as separate tags `["RAM", "Display", "Storage"]`.
   - **Fallback:** If no keywords match, the system falls back to the Guided BERTopic model to assign a general related cluster.

**Result:** A single review now successfully maps to **multiple, highly relevant, and visually styled aspect tags** on the dashboard. ✅

---

## Slide 7 — Flask Web Application

Built a **Flask web dashboard** (`app.py`) with two views:

### View 1 — Aspect Dashboard (`/`)
- Real-time **sentiment prediction** for any typed text
- Shows predicted **Sentiment** + **Related Aspect** (BERTopic cluster name)
- Live bar charts of aspect sentiment distributions across the dataset

### View 2 — Product Dashboard (`/products`)
- **Doughnut charts** per product showing Positive/Neutral/Negative breakdown
- **Drill-down:** Click any chart segment → modal popup showing raw reviews for that sentiment + product
- Powered by `/api/product-reviews` endpoint

---

## Slide 8 — Docker Deployment & Automated Retraining

> [!NOTE]
> The entire system runs inside Docker — no local Python environment needed. Two containers work together.

### Container Architecture:

```
┌─────────────────────────────────────────────────────┐
│  docker-compose                                     │
│  ┌──────────────────┐    ┌───────────────────────┐  │
│  │  web container   │    │  trainer container    │  │
│  │  Flask app       │    │  watchdog watcher     │  │
│  │  Port 5001       │    │  watches incoming/    │  │
│  │  DeBERTa model   │    │  triggers retrain     │  │
│  │  BERTopic model  │    │  on new CSV drop      │  │
│  │  Cardiff model   │    │                       │  │
│  └──────────────────┘    └───────────────────────┘  │
│  Shared volume: deploy_models/, Ecommerce_dataset/  │
└─────────────────────────────────────────────────────┘
```

### Automated Retraining Pipeline (`watch_and_train.py`):
1. **File detected** → new CSV dropped into `Ecommerce_dataset/incoming/`
2. **Backup** → current `train_data.csv` backed up before merge
3. **Merge** → new data appended to training set
4. **Retrain DeBERTa** → `train_and_validate.py` runs
5. **Retrain BERTopic** → `train_bertopic.py` runs
6. **Error recovery** → if any step fails, rolls back to backed-up CSV
7. **Cleanup** → incoming file deleted on success

### Real-Time Progress Banner:
- Frontend polls `/training-status` every 2.5 seconds
- Animated progress bar shows current stage (0-100%)
- Page **auto-refreshes** when training completes

---

## Slide 9 — Technical Challenges Solved

| Challenge | Root Cause | Solution Applied |
|---|---|---|
| App hung on macOS | HDBSCAN/OpenMP thread conflict on Apple Silicon | `OMP_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false` |
| Model predicted Neutral for negative slang | "sucks" not in training vocab (0 occurrences) | Cardiff RoBERTa confidence fallback |
| Topic names had "teenage", "daughter" | Amazon reviews co-mingle demographics with features | `clean_topic_name()` + custom CountVectorizer stop words + seed_topic_list |
| Model path broken inside Docker | Model saved with absolute `/app/` path | `load_model()` patched to map `/app/` → relative path |
| Cardiff model downloading on every restart | No local caching | Downloaded once, saved to `deploy_models/cardiffnlp_fallback/` |
| Class imbalance (93% Positive) | Dataset distribution skew | `compute_class_weight('balanced')` on all transformer models |

---

## Slide 10 — Key Numbers

| Metric | Value |
|---|---|
| Models trained & evaluated | 7 |
| Training reviews | 4,003 |
| Total reviews (train + test + predicted) | 6,004 |
| BERTopic seed topic groups | 12 |
| Filler words filtered from topic names | 50+ |
| Cardiff fallback confidence threshold | 0.65 |
| Docker containers | 2 (web + trainer) |
| Disk freed by removing unused model artifacts | ~1.2 GB |
| Local Cardiff model size | ~480 MB |
| Lines of Python code written | ~1,155 |

---

## Slide 11 — Architecture Summary (One Slide Diagram)

```
                  ┌─────────────────────────────┐
                  │        User (Browser)       │
                  └──────────────┬──────────────┘
                                 │ HTTP :5001
                  ┌──────────────▼──────────────┐
                  │     Flask Web App (app.py)   │
                  │  GET /        → Aspect View  │
                  │  GET /products→ Product View │
                  │  POST /predict→ Inference    │
                  └──────┬───────────────┬───────┘
                         │               │
           ┌─────────────▼──┐   ┌────────▼──────────┐
           │ DeBERTa Model  │   │  BERTopic Model   │
           │ (fine-tuned)   │   │  (guided + seeds) │
           │ conf ≥ 0.65?   │   │  → aspect label   │
           └────┬───────────┘   └───────────────────┘
                │ NO
      ┌─────────▼──────────────┐
      │ Cardiff RoBERTa Model  │
      │ (Twitter-trained,      │
      │  slang-aware fallback) │
      └────────────────────────┘
```

---

## What Makes This Project Stand Out

1. **Not just a single model** — 7 models compared head-to-head with metrics
2. **Dual-model inference** — smart confidence gating, not just one model
3. **Unsupervised aspect detection** — BERTopic without needing labeled aspect data
4. **Guided topic modeling** — Semi-supervised BERTopic with product-domain seeds
5. **Production deployment** — Dockerized, auto-retraining, real-time UI
6. **Error resilience** — automatic rollback if retraining fails
7. **No internet required at runtime** — all models cached locally
