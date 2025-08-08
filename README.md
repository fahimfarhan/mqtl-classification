# mqtl-classification

* To process long sequences, please use --model_type dnalong, and set the max sequence length as a multiple of 512 (e.g., 3072). Then the model should work well. [DNABERT github issues](https://github.com/jerryji1993/DNABERT/issues/18#issuecomment-823707084)

```bash
python3 main.py \
  --MODEL_NAME LongSafari/hyenadna-small-32k-seqlen-hf \
  --MODEL_VARIANT HyenaDNAWithDropoutAndNorm \
  --RUN_NAME_PREFIX hyena-dna-mqtl-classifier \
  --WINDOW 1024 \
  --NUM_EPOCHS 15 \
  --ENABLE_LOGGING \
  --LEARNING_RATE 5e-4 \
  --L1_LAMBDA_WEIGHT 1e-3 \
  --WEIGHT_DECAY 0.01 \
  --GRADIENT_CLIP 5.0 \
  --OPTIMIZER adam
```

## linear scheduler => accuracy auc 0.5, consant lr adam => opimizer overfiting

things to do next:
* cosine scheduler,

```python
def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    # Total steps = num_epochs * steps_per_epoch
    # You need to set this dynamically outside if you want exact control
    num_training_steps = self.trainer.estimated_stepping_batches
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    scheduler_config = {
        "scheduler": scheduler,
        "interval": "step",   # step-wise decay
        "frequency": 1,
        "name": "learning_rate",  # shows up in wandb as `learning_rate`
    }

    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler_config
    }
```
* use lr = 5e-4, and warm up = 0.01
* use l2 regularization, ie, weight decay = 0.01
* use drop out
* Create a baseline model (LSTM + linear layer)
* Plot confusion matrix: is it missing only one class? is it near the decision boundary
```python
# after evaluating test set
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
```

* Changing adamw to Adam with cosine scheduler. try lion, or adan as well.
```bash
python3 main.py \
  --MODEL_NAME LongSafari/hyenadna-small-32k-seqlen-hf \
  --MODEL_VARIANT HyenaDNAWithDropoutAndNorm \
  --RUN_NAME_PREFIX hyena-dna-mqtl-classifier \
  --WINDOW 1024 \
  --NUM_EPOCHS 10 \
  --LEARNING_RATE 5e-4 \
  # --L1_LAMBDA_WEIGHT 1e-3 \
  --WEIGHT_DECAY 0.01 \
  --GRADIENT_CLIP 5.0 \
  --OPTIMIZER adam \
#  --ENABLE_LOGGING

  #  --ENABLE_LOGGING \
  #  --L1_LAMBDA_WEIGHT 0 \
!python3 main.py \
  --MODEL_NAME LongSafari/hyenadna-small-32k-seqlen-hf \
  --MODEL_VARIANT HyenaDNAWithDropoutBatchNorm1d \
  --RUN_NAME_PREFIX hyena-dna-mqtl-classifier \
  --WINDOW 1024 \
  --NUM_EPOCHS 3 \
  --LEARNING_RATE 5e-5 \
  --WEIGHT_DECAY 0 \
  --GRADIENT_CLIP 5.0 \
  --OPTIMIZER adamw \
  --DROP_OUT_PROBABILITY 0.5 \
  --CRITERION_LABEL_SMOOTHENING 0.1
  
  
# dnabert 6
!python3 main.py \
  --MODEL_NAME zhihan1996/DNA_bert_6 \
  --MODEL_VARIANT default \
  --RUN_NAME_PREFIX dnabert6-mqtl-classifier \
  --WINDOW 512 \
  --NUM_EPOCHS 3 \
  --LEARNING_RATE 5e-5 \
  --WEIGHT_DECAY 0 \
  --GRADIENT_CLIP 5.0 \
  --OPTIMIZER adamw \
  --DROP_OUT_PROBABILITY 0.5 \
  --CRITERION_LABEL_SMOOTHENING 0.1
```

6.0s	0	0.00s - Debugger warning: It seems that frozen modules are being used, which may
6.0s	1	0.00s - make the debugger miss breakpoints. Please pass -Xfrozen_modules=off
6.0s	2	0.00s - to python to disable frozen modules.
6.0s	3	0.00s - Note: Debugging will proceed. Set PYDEVD_DISABLE_FILE_VALIDATION=1 to disable this validation.
6.8s	4	0.00s - Debugger warning: It seems that frozen modules are being used, which may
6.8s	5	0.00s - make the debugger miss breakpoints. Please pass -Xfrozen_modules=off
6.8s	6	0.00s - to python to disable frozen modules.
6.8s	7	0.00s - Note: Debugging will proceed. Set PYDEVD_DISABLE_FILE_VALIDATION=1 to disable this validation.


how to fix overfitting?
1. Add weight regularization, L1, and L2
2. Use the dropOut

| Technique                    | Purpose                      |
| ---------------------------- | ---------------------------- |
| ✅ Data sanity check          | Eliminate data leakage       |
| ✅ Tiny subset overfit        | Test model capacity          |
| ✅ Train longer w/ early stop | Allow generalization to show |
| ✅ Regularization tuning      | Reduce overfitting capacity  |
| ✅ Prediction analysis        | Diagnose model confusion     |  ok
| ✅ Data augmentation          | Enrich generalization        |
| ✅ Pretrained backbone        | Boost feature richness       |
| ✅ LR tuning                  | Improve convergence          |

```bash
!rm -rf mqtl-classification
!rm -rf models

!git clone https://github.com/fahimfarhan/mqtl-classification.git
# copying files to root. so output folders will be easily found at the root. No need to navigate 

!mkdir -p models
!cp mqtl-classification/src/experiment/models/*.py models/
!cp mqtl-classification/src/experiment/*.py .
!rm -rf mqtl-classification
```

Popular Pretrained Models for DNA/RNA Sequence Tasks
🔹 1. HyenaDNA — (you're using this)

    Pros: Handles long sequences (up to 32k tokens!), fast, low memory.
    Use case: Long-range genomic interactions, enhancers, etc.
    Hugging Face: LongSafari/hyenadna-small-32k-seqlen-hf

🔹 2. DNABERT / DNABERT-2

    Model: BERT adapted for k-mer tokenized DNA (e.g., 6-mers).
    Pros: Simple, widely used, pretrained on human genome (hg19).
    Use case: Promoter classification, TF binding, CpG island detection.

    Hugging Face:
        zhihan1996/DNABERT-6
        zhihan1996/DNAbert-2-117M (improved version)

    Note: Input is tokenized using overlapping k-mers (like ACGTAC → ACG, CGT, etc).

🔹 3. Enformer

    Model: Huge transformer trained to predict epigenomic profiles across 200k bp sequences.
    Pros: SOTA on functional genomics benchmarks.
    Cons: Very large (~700M+ parameters), hard to fine-tune on small data.
    Hugging Face (converted): Search for "Enformer" or use via DeepMind repo.

🔹 4. Basenji2 / BpNet

    Model: Deep CNN (not transformers), effective for functional genomics (TF binding, epigenetic marks).
    Use case: Works well on localized genomic patterns.
    Library: Kipoi
        Look for DeepSEA, BpNet, Basenji, etc.

🔹 5. GenFormer / GPN (Genomic Perceiver)

    Model: Perceiver-based model trained for genome-scale data.
    Pros: Handles very long contexts like Hyena.
    Cons: Research-level code.
    Paper: "Genomic Perceiver"
    Hugging Face (WIP): Some unofficial ports exist.

🔹 6. Genome-T5 (gT5)

    Model: T5-style encoder-decoder pre-trained on DNA language tasks.
    Good for: Sequence generation or masking tasks, but can be adapted for classification.
    Check: Papers with Code + GitHub.

🔹 7. DeepSEA / DeepBind / CpGenie

    Early CNN-based models for genomic feature prediction.
    Simpler and lightweight.
    Available via:
        Kipoi: https://kipoi.org/models/
        Code from their GitHub repos.


DeepSEA/variantEffects
DeepBind/Homo_sapiens/RBP_eIF4A3
CpGenie
DeepCpG
DanQ
Basenji
DeepSea/Begula
Expecto
BPNet



!rm -rf mqtl-classification
!git clone https://github.com/fahimfarhan/mqtl-classification.git
!mkdir -p models
!cp mqtl-classification/src/experiment/models/*.py models/
!cp mqtl-classification/src/experiment/*.py .
!rm -rf mqtl-classification