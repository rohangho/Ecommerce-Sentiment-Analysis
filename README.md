# Ecommerce Sentiment Analysis

Sentiment classification for e‑commerce reviews (**Negative / Neutral / Positive**) using classical ML, an RNN, and several Transformer models trained in TensorFlow.

## Project structure

| Path | Purpose |
|------|---------|
| `Ecommerce_dataset/train_data.csv` | Training data |
| `Ecommerce_dataset/test_data_hidden.csv` | Held‑out labeled test set for final metrics |
| `notebooks/` | Model implementations (`sentiment_analysis_*.py`), EDA (`CapstoneData.py`) |
| `deploy_models/` | Saved checkpoints after training (`.pkl` metadata + model weights directories) |
| `validation_outputs/` | **Generated** — metrics table and comparison plots (see below) |
| `train_and_validate.py` | Train all models, run hidden‑test evaluation, write plots |
| `validate_deploy_models.py` | **Validate only** — load `deploy_models/`, score `test_data_hidden`, refresh plots |
| `MasterRunner.ipynb` | End‑to‑end notebook (EDA → training → model load → sample predictions → reporter) |

## Models

Trained and evaluated in one pipeline:

1. **Logistic regression** (TF‑IDF + categorical features)  
2. **SVM**  
3. **RNN** (Keras)  
4. **DistilBERT**  
5. **BERT**  
6. **RoBERTa** (siebert backbone, 3‑class head)  
7. **DeBERTa** (microsoft/deberta‑v3‑base, 3‑class head)

Artifacts are written under `deploy_models/` (for example `sentiment_model_TFD_LR.pkl`, `sentiment_model_distilbert.pkl`, …).

## Training and validation

From the repository root (with your virtualenv activated and dependencies installed per `requirements.txt`):

```bash
python train_and_validate.py
```

This cleans `train_data.csv` the same way as the EDA notebook, trains every model, runs **validation on `test_data_hidden.csv`** using the saved deploy artifacts, prints per‑model classification reports, and writes comparison assets to **`validation_outputs/`**.

To **skip training** and only reload models from `deploy_models/`, re‑score the hidden test set, and regenerate plots:

```bash
python validate_deploy_models.py
```

Optional flags:

```bash
python validate_deploy_models.py --deploy-dir deploy_models --test-csv Ecommerce_dataset/test_data_hidden.csv --out-dir validation_outputs
```

### Outputs in `validation_outputs/`

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Per‑model accuracy, macro precision / recall / F1, weighted F1, per‑class precision / recall / F1 |
| `confusion_matrices_counts.png` | Confusion matrices (raw counts) for each model |
| `confusion_matrices_normalized.png` | Row‑normalized matrices (recall per true class) |
| `macro_precision_recall_f1.png` | Bar chart comparing macro precision, recall, and F1 |
| `f1_per_class_heatmap.png` | Models × sentiment class (F1) |
| `precision_per_class_heatmap.png` | Models × class (precision) |
| `recall_per_class_heatmap.png` | Models × class (recall) |
| `accuracy_by_model.png` | Accuracy bar chart |

The notebook **Section 2** (`train_and_validate.run_pipeline(eda.df)`) performs the same training + validation + plot export.

### Training hyperparameters (defaults)

Settings are tuned for the imbalanced three-class setup (many Positive, few Neutral/Negative). The pipeline in `train_and_validate.py` uses:

| Model | Notable settings |
|-------|-------------------|
| **Logistic** | Stratified split; TF–IDF bigrams, `min_df=2`, higher `max_features`; `saga` solver, `C=0.85`, `class_weight='balanced'` |
| **SVM** | Stratified split; larger TF–IDF vocab, `C=2.5`, RBF, `class_weight='balanced'` |
| **RNN** | Longer sequences (`max_len=256`), larger vocab/embedding; **EarlyStopping** + **ReduceLROnPlateau**; up to 24 epochs (early stop) |
| **DistilBERT** | `max_length=256`, batch 12, 3 epochs, AdamW `weight_decay=0.02`, ~12% warmup |
| **BERT** | Batch 12, 3 epochs, same decay/warmup style as DistilBERT |
| **RoBERTa** | Large backbone: slightly lower LR (`1.7e-5`), 3 epochs, 14% warmup, `weight_decay=0.02` |
| **DeBERTa** | 3 epochs, `2e-5` LR, 12% warmup, `weight_decay=0.02` |

Individual classes accept overrides (e.g. `epochs`, `learning_rate`) if you train from a notebook.

## Dependencies

See `requirements.txt` (TensorFlow, transformers, scikit‑learn, pandas, matplotlib, seaborn, etc.).

## Contribution guidelines

To keep the codebase stable, contributors should follow these rules.

### 1. Branching policy

- **No direct pushes to `main`.**  
- Use feature branches: `feature/your-feature-name` or `fix/issue-description`.  
- Open a pull request into `main`.

### 2. Review process

- PRs need review before merge.  
- Ensure checks/tests pass where applicable.

### 3. Issue reporting

- Do not patch ad hoc on `main`.  
- Open an issue with reproduction steps and expected behavior.

---

*Maintained by the Ecommerce Sentiment Analysis team.*
