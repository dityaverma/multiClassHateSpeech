# Multi-Modal Harmful Content Detection for Tamil & Telugu Memes

<p align="center">
  <img src="architecture.png" alt="Multimodal HASOC architecture" width="100%">
</p>

<p align="center">
  <b>Multimodal • Multi-Task • Cross-Modal Attention • Task-Aware Gating • Statistical Relationship Analysis</b>
</p>

---

## Abstract

We present a multimodal multi-task framework for harmful-content analysis in **Tamil and Telugu memes**. The system jointly models the visual content of a meme and the text extracted from it using [..[...]

The system predicts five related tasks:

- **Sentiment**
- **Sarcasm**
- **Vulgarity**
- **Abuse**
- **Target**

The training pipeline additionally addresses severe class imbalance using a **multi-task weighted sampler, class-balanced Focal Loss, task-level loss weighting, staged fine-tuning, and validation-[...]

A separate statistical analysis layer studies dependencies among the categorical labels using **Cramér's V and conditional probability**. Rather than blindly applying pairwise label relationships[...[...]

We also explored **few-shot GPT-based semantic analysis** for difficult multiclass cases, particularly Sentiment and Target. This component is treated as an auxiliary analysis/prediction strategy [...[...]

---

## 1. Task Definition

The problem is formulated as a five-task classification problem over Tamil and Telugu memes.

| Task | Problem type | Output |
|---|---|---|
| Sentiment | Multiclass | Negative / Neutral / Positive |
| Sarcasm | Binary | No / Yes |
| Vulgarity | Binary | Not Vulgar / Vulgar |
| Abuse | Binary | Non-Abusive / Abusive |
| Target | Multiclass | Gender / Individual / None / Others / Political / Social Sub-Groups |

The implementation defines these five task heads explicitly and maps the internal `vulgar` task to the official `vulgarity` submission column. fileciteturn36file0L244-L270

---

# 2. Architecture

The core architecture implemented in the notebook is:

```text
                         MEME IMAGE
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
          SigLIP           OCR Text      Context Text
          Vision              │              │
          Encoder             ▼              ▼
              │          IndicBERT       IndicBERT
              │              │              │
              └──────────────┼──────────────┘
                             │
                   Non-Linear Projection
                             │
                             ▼
                ┌─────────────────────────┐
                │ Tri-Modal Cross-Attention  │
                │ Image <-> Text <-> Context │
                └────────────┬────────────┘
                             │
                             ▼
                  2-Layer Transformer
                         Fusion
                             │
                             ▼
                 Learned Attention Pooling
                             │
                             ▼
                       Shared MLP
                             │
                             ▼
                   Task Interaction Gate
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
      Sentiment          Sarcasm            Vulgarity
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                           Abuse
                             │
                             ▼
                           Target
```

The actual notebook implements `TriModalCrossAttentionBlock`, `LearnedAttentionPooling`, `TaskInteractionGate`, and `SeniorMultimodalHASOCModel`. fileciteturn36file0L887-L1003

---

# 3. Modality Encoding

## 3.1 Visual Encoder

The visual branch uses:

```text
google/siglip-base-patch16-224
```

with a 768-dimensional hidden representation.

The visual tokens are projected to the common fusion dimension:


through a nonlinear projection block.

The notebook explicitly configures SigLIP with `siglip_hidden_dim=768` and `d_fusion=256`. fileciteturn36file0L196-L214

---

## 3.2 OCR Text Encoder

The OCR text branch uses:

```text
ai4bharat/IndicBERTv2-MLM-only
```

OCR text is Unicode-normalized and whitespace-cleaned while preserving the underlying Tamil/Telugu text.

The maximum sequence length is:

```text
128 tokens
```

The IndicBERT representation is projected:

The dataset implementation separately tokenizes the OCR text and contextual text. fileciteturn36file0L775-L800

---

## 3.3 Context Representation

A separate context sequence is encoded using the same IndicBERT backbone.

This gives the model three logical streams:

```text
Visual representation
+
OCR representation
+
Context representation
```

The three streams are not simply concatenated at the beginning. They first interact through cross-attention.

---

# 5. Tri-Modal Cross-Attention

The central fusion mechanism is a custom tri-modal cross-attention block.

### Image attends to text + context

### Text attends to image + context

### Context attends to image + text

This is implemented using three separate multi-head attention modules with **8 heads**, residual connections, dropout and LayerNorm. fileciteturn36file0L897-L920

The motivation is to allow the modalities to condition one another.

For example:

```text
Visual expression
       v
helps interpret
       v
OCR phrase
       v
helps interpret
       v
context
```

This is particularly relevant for sarcastic memes where the literal OCR text alone may not reveal the intended meaning.

---

# 6. Transformer Fusion

After cross-modal interaction, the image, text and context tokens are concatenated:

and processed using a Transformer encoder with:

| Parameter | Value |
|---|---:|
| Layers | 2 |
| Attention heads | 8 |
| Model dimension | 256 |
| Feed-forward dimension | 1024 |
| Activation | GELU |
| Dropout | 0.2 |

The implementation directly constructs this Transformer encoder after the cross-attention block. fileciteturn36file0L967-L970

---

# 7. Learned Attention Pooling

Instead of using a fixed CLS token or simple mean pooling, the architecture learns a scalar importance score for every fused token.

The pooled representation is then passed through a shared MLP with LayerNorm, GELU and dropout. fileciteturn36file0L922-L929

---

# 8. Task Interaction Gate

The model contains five learned task queries:

```text
Sentiment
Sarcasm
Vulgarity
Abuse
Target
```

The task queries interact using a 4-head multi-head attention module.

Each task representation is then gated against the shared multimodal representation:

where k denotes the task.

This allows the model to produce **task-specific views of the same multimodal representation**.

The implementation creates one learned query per task and produces five gated representations. fileciteturn36file0L931-L948

---

# 9. Task-Specific Heads

Each task has its own classification head:

```text
Dropout
   v
Linear(256 -> number of classes)
```

Therefore the model does not force all tasks to share the same final decision boundary.

The resulting structure is:

```text
                         Shared Multimodal Feature
                                  │
                         Task Interaction Gate
                                  │
             ┌────────────┬───────┼───────┬────────────┐
             ▼            ▼       ▼       ▼            ▼
        Sentiment      Sarcasm  Vulgar   Abuse       Target
```

The five heads are explicitly constructed in the model's `ModuleDict`. fileciteturn36file0L950-L980

---

# 10. Data Processing

The notebook includes a dynamic EDA and data-integrity stage.

It checks:

- CSV availability
- Sample counts
- Class distributions
- OCR statistics
- Duplicate OCR text
- Missing images
- Corrupt images

The EDA outputs are written under:

```text
reports/dataset_eda/
```

The official unlabeled test files are kept separate from training/validation splitting. fileciteturn36file0L24-L34

---

# 11. OCR Preprocessing

OCR text is normalized using Unicode NFKC normalization.

The preprocessing removes redundant repetitions of:

```text
!!!
???
...
```

and collapses repeated whitespace.

The implementation deliberately avoids aggressive language-specific normalization so that Tamil/Telugu characters, code-mixed text and semantic cues remain available to IndicBERT. fileciteturn36f[...]

---

# 12. Image Augmentation

Training images can undergo lightweight augmentations:

- Small random rotation
- Brightness variation
- Contrast variation
- JPEG quality degradation

The augmentation is intentionally moderate because aggressive transformations can alter meme text or visual semantics. fileciteturn36file0L618-L644

---

# 13. Handling Class Imbalance

The dataset is highly imbalanced, especially for:

```text
Vulgarity
Abuse
```

The pipeline therefore combines multiple mechanisms.

## 13.1 Multi-task weighted sampling

A `WeightedRandomSampler` is constructed using all task labels.

For each task:

where:

- N = number of training samples
- K = number of classes
- n_c = class frequency

Task-specific sample weights are normalized and accumulated across tasks.

This prevents the sampler from focusing on only one label distribution. fileciteturn36file0L688-L706

---

# 14. Class-Balanced Focal Loss

The core loss is Focal Loss:

with:

The class weight \alpha_t is computed from inverse class frequency on the **training split only**:

This gives greater influence to minority classes.

The implementation computes these weights dynamically for every task. fileciteturn37file0L135-L147

---

# 15. Task-Level Loss Weighting

The total multi-task objective is:

with:

| Task | Weight |
|---|---:|
| Sentiment | 1.0 |
| Sarcasm | 1.2 |
| Vulgarity | 2.0 |
| Abuse | 2.5 |
| Target | 1.0 |

Thus the training objective gives additional emphasis to the more difficult/imbalanced binary harmful-content tasks. fileciteturn36file0L227-L242

---

# 16. Staged Fine-Tuning

Training uses two stages.

## Stage 1 -- Fusion/head adaptation

The pretrained SigLIP and IndicBERT backbones are frozen.

Only the newly initialized multimodal fusion and task-specific layers are optimized.

```text
Epochs: 10
LR: 1e-4
```

## Stage 2 -- Selective backbone fine-tuning

The top two SigLIP layers and top two IndicBERT encoder layers are unfrozen.

Learning rates:

```text
Backbone: 1e-5
Task/fusion layers: 5e-5
```

This provides a conservative form of differential fine-tuning.

The staged freezing/unfreezing strategy is implemented explicitly in the training loop. fileciteturn37file0L185-L216

---

# 17. Optimization

The training configuration uses:

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Stage 1 LR | 1\times10^{-4} |
| Stage 2 backbone LR | 1\times10^{-5} |
| Stage 2 head LR | 5\times10^{-5} |
| Weight decay | 10^{-2} |
| Warm-up ratio | 0.1 |
| Gradient clipping | 1.0 |
| Gradient accumulation | 2 |
| Batch size | 8 |
| Effective batch size | 16 |

A cosine schedule with warm-up is used in both training stages. fileciteturn36file0L213-L226 fileciteturn37file0L219-L263

---

# 18. Evaluation Protocol

The primary metric is **Macro-F1**.

For each task, the pipeline reports:

- Macro-F1
- Accuracy
- Macro Precision
- Macro Recall

The overall score is:

The checkpoint saver tracks:

- Best overall mean Macro-F1
- Best Sentiment F1
- Best Sarcasm F1
- Best Vulgarity F1
- Best Abuse F1
- Best Target F1

The checkpoint contains the complete model, optimizer, scheduler, scaler, epoch, thresholds, validation metrics and configuration. fileciteturn37file0L63-L92

---

# 19. Threshold Optimization

The three binary tasks are not forced to use the default 0.5 threshold.

For:

```text
Sarcasm
Vulgarity
Abuse
```

and the threshold producing the highest validation Macro-F1 is retained.

This is performed **only on the validation set**. fileciteturn37file0L336-L354

Multiclass Sentiment and Target predictions use argmax.

---

# 20. Error Analysis

The pipeline exports:

```text
all_validation_predictions.csv
lowest_confidence.csv
worst_predictions.csv
```

and generates:

```text
confusion_matrices/
```

for every task.

This enables targeted inspection of:

- Minority-class failures
- Low-confidence examples
- Samples with multiple simultaneous task errors
- Systematic confusion between classes

The implementation records per-task target, prediction, confidence and error flags. fileciteturn37file0L356-L410

---

# 21. Statistical Feature Engineering with Cramér's V

The neural architecture is complemented by a separate categorical relationship analysis.

For two categorical variables X and Y:

where:

- N is the sample count
- r is the number of rows in the contingency table
- c is the number of columns
- ^2 is the Chi-square statistic

The analysis is run **separately for Tamil and Telugu**.

---

# 22. Why Cramér's V?

The five tasks are not necessarily independent.

For example, sentiment may provide information about sarcasm, while vulgarity may provide information about abuse.

However, direct pairwise rules are risky because the dataset is highly skewed.

Consider a highly imbalanced binary task:

```text
Not Vulgar     ████████████████████
Vulgar         █
```

A pairwise relationship can appear strong simply because the majority class dominates.

Therefore, Cramér's V is used as an **association-screening statistic**, not as a direct prediction rule.

---

# 23. Conditional Probability

After identifying potentially useful associations, we calculate directional conditional probabilities:

This is different from Cramér's V.

Cramér's V measures:

while conditional probability measures:

The direction therefore matters.

---

# 24. Multi-Category Relationship Search

Because of class skewness, we do not rely only on relationships such as:

```text
Sentiment -> Sarcasm
```

Instead, the analysis searches for:

```text
Sentiment + Sarcasm -> Abuse
```

```text
Sentiment + Vulgarity -> Abuse
```

```text
Sentiment + Sarcasm + Vulgarity -> Abuse
```

and other logically valid configurations.

The search is performed for:

1. Pair conditions
2. Triplet conditions
3. Four-way conditions

Each result stores:

- Condition
- Outcome
- Condition count
- Outcome count
- Conditional percentage

This allows a 100% relationship based on a small number of observations to be distinguished from a high-confidence relationship supported by hundreds of samples.

---

# 25. Tamil and Telugu Are Analyzed Independently

We do not assume that the same label relationships hold equally across both languages.

The pipeline therefore computes:

```text
Tamil
  v
Contingency analysis
  v
Cramér's V
  v
Conditional combinations
```

and independently:

```text
Telugu
  v
Contingency analysis
  v
Cramér's V
  v
Conditional combinations
```

The strongest relationships can therefore be language-specific.

---

# 26. Few-Shot GPT Analysis

We additionally used **few-shot GPT-based semantic analysis** for difficult multiclass cases, particularly **Sentiment and Target**.

The motivation was that meme interpretation depends not only on the OCR text but also on the **visual context of the meme**. Facial expressions, gestures, actions, characters, visual situations and other contextual cues can significantly change the interpretation of the text and influence both sentiment and the intended target.

A text-only model may therefore miss information such as:

- Facial expressions that indicate positive or negative sentiment
- Actions and gestures that change the meaning of the text
- Visual situations that provide sarcastic or emotional context
- The identity or role of the person or group being referred to
- Visual cues that help determine the intended target
- Contradictions between the literal OCR text and the visual meaning of the meme

The few-shot workflow is:

```text
Meme Image + OCR Text
          +
Few-Shot Examples
          |
          v
   GPT Multimodal Analysis
          |
          v
Visual Context + Textual Context
          |
          v
Candidate Sentiment / Target

The few-shot examples provide the model with representative labelled cases so that it can reason about the relationship between the meme image, its extracted text and the expected classification.
```

---

# 27. Why Multiclass GPT Prediction Was Not Used as the Core Model

Sentiment and Target are multiclass problems with substantial class imbalance.

Direct few-shot multiclass predictions can therefore be inconsistent across minority categories.

Rather than replacing the supervised model with GPT predictions, we use the few-shot model as:

- An auxiliary semantic signal
- A difficult-example analysis tool
- A source of hypotheses for multiclass errors

The supervised multimodal model remains the primary reproducible prediction architecture.

---

# 28. Relationship-Based Prediction Refinement

The statistical analysis can be used as a **post-prediction diagnostic/refinement layer**.

The workflow is:

```text
Model Predictions
       │
       ▼
Predicted task labels/probabilities
       │
       ▼
Check high-confidence
multi-category configurations
       │
       ▼
Compare with training-derived
conditional relationships
       │
       ▼
Flag or conservatively refine
inconsistent predictions
```

The important design principle is:

> We do not treat a pairwise correlation as a deterministic rule.

Instead, multiple category conditions are considered jointly, and the support count is inspected before a relationship is considered useful.

---

# 29. Avoiding Label Leakage

All statistical relationships must be learned from the training portion.

The validation set is used for:

- Model selection
- Threshold optimization
- Error analysis

The official unlabeled test set is reserved for final inference.

The notebook explicitly separates the official test CSV from the annotated training/validation data. fileciteturn36file0L477-L541

---

# 30. Reproducibility

The pipeline includes:

- Deterministic random seeding
- Local Hugging Face caching
- Dynamic path resolution
- Full optimizer/scheduler/scaler checkpoints
- Configuration export
- Threshold export
- Automatic Tamil/Telugu execution
- GPU memory cleanup between languages

The notebook runs:

```python
for lang in ['tamil', 'telugu']:
    ...
```

and explicitly releases the model and CUDA cache before moving to the next language. fileciteturn37file0L582-L665

---

# 31. Output Artifacts

A successful run produces:

```text
outputs/
├── tamil_predictions.csv
├── tamil_predictions_probs.csv
├── telugu_predictions.csv
└── telugu_predictions_probs.csv
```

alongside:

```text
models/
├── tamil/
│   ├── best_mean_macro_f1.pt
│   ├── best_sentiment.pt
│   ├── best_sarcasm.pt
│   ├── best_vulgar.pt
│   ├── best_abuse.pt
│   ├── best_target.pt
│   ├── thresholds.json
│   └── run_config.json
│
└── telugu/
    └── ...
```

and error-analysis reports under:

```text
reports/error_analysis/
```

The official inference routine writes both submission predictions and probability/confidence files. fileciteturn37file0L529-L580

---

# 32. Research Contributions

### Multimodal meme understanding

The system jointly models visual and textual information rather than treating OCR text as the complete meme representation.

### Tri-modal cross-modal reasoning

Image, OCR text and context interact through dedicated cross-attention blocks.

### Task-aware multi-task learning

A learned task interaction gate generates task-specific representations from a shared multimodal feature.

### Imbalance-aware optimization

The combination of weighted sampling, class-balanced Focal Loss and task-level weighting explicitly addresses the long-tail label distribution.

### Language-aware statistical analysis

Tamil and Telugu are analyzed independently rather than assuming identical task dependencies.

### Higher-order relationship discovery

The statistical layer searches for multi-category configurations rather than relying solely on pairwise relationships.

### Validation-aware decision thresholds

Binary task thresholds are optimized against validation Macro-F1.

### Auxiliary few-shot semantic analysis

Few-shot GPT analysis is used as an auxiliary mechanism for difficult semantic and multiclass cases rather than replacing the core supervised model.

---

# 33. Evaluation and Output Artifacts

The system evaluates all five tasks using Macro-F1, Accuracy, Macro Precision and Macro Recall. Macro-F1 is used as the primary metric because the task distributions are highly imbalanced.

For each language, the pipeline records:

- Per-task validation performance
- Mean Macro-F1
- Optimized binary thresholds
- Confusion matrices
- Classification reports
- Per-sample predictions
- Prediction confidence
- Error flags
- Official test predictions
- Prediction probability files

The final prediction outputs are written separately for Tamil and Telugu and are kept independent of the validation analysis.

# 34. Key Configuration

```python
    "siglip_model_name": "google/siglip-base-patch16-224",
    "indicbert_model_name": "ai4bharat/IndicBERTv2-MLM-only",

    "d_fusion": 256,
    "max_seq_len": 128,

    "batch_size": 8,
    "grad_accum_steps": 2,

    "stage1_epochs": 10,
    "stage2_epochs": 10,

    "lr_head_stage1": 1e-4,
    "lr_head_stage2": 5e-5,
    "lr_backbone_stage2": 1e-5,

    "weight_decay": 1e-2,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,

    "focal_gamma": 2.0
}
```

The task-level loss weights are:

```python
{
    "sentiment": 1.0,
    "sarcasm": 1.2,
    "vulgar": 2.0,
    "abuse": 2.5,
    "target": 1.0
}
```

These values are directly defined in the submitted notebook. fileciteturn36file0L196-L242

---

# 38. End-to-End Method

The complete method can be summarized as:

```text
                  Tamil / Telugu Meme
                          │
                          ▼
                ┌─────────────────┐
                │   Data / OCR    │
                └────────┬────────┘
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          SigLIP      IndicBERT    IndicBERT
          Image         OCR         Context
             │           │           │
             └───────────┼───────────┘
                         ▼
                 Shared 256-D Space
                         │
                         ▼
               Tri-Modal Cross Attention
                         │
                         ▼
                 Transformer Fusion
                         │
                         ▼
              Learned Attention Pooling
                         │
                         ▼
                      Shared MLP
                         │
                         ▼
                 Task Interaction Gate
                         │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
       Sentiment       Sarcasm        Vulgarity
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                        Abuse
                          │
                          ▼
                        Target
                          │
                          ▼
                  Validation Metrics
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
   Threshold Optimization        Error Analysis
           │                             │
           └──────────────┬──────────────┘
                          ▼
                   Official Inference
                          │
                          ▼
                  Submission CSVs
```

In parallel, the training labels are analyzed through:

```text
Training Labels
      │
      ▼
Contingency Tables
      │
      ▼
Cramér's V
      │
      ▼
Conditional Probability
      │
      ▼
Pair / Triplet / 4-Way Search
      │
      ▼
High-Support Logical Configurations
```
