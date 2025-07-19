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
