# mqtl-classification

* To process long sequences, please use --model_type dnalong, and set the max sequence length as a multiple of 512 (e.g., 3072). Then the model should work well. [DNABERT github issues](https://github.com/jerryji1993/DNABERT/issues/18#issuecomment-823707084)

```bash
python3 main.py \
  --MODEL_NAME LongSafari/hyenadna-small-32k-seqlen-hf \
  --MODEL_VARIANT HyenaDNAWithDropoutAndNorm \
  --run_name_prefix hyena-dna-mqtl-classifier \
  --WINDOW 1024 \
  --NUM_EPOCHS 15 \
  --ENABLE_LOGGING \
  --LEARNING_RATE 5e-4 \
  --WEIGHT_DECAY 0.01 \
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

python3 main.py \
  --MODEL_NAME LongSafari/hyenadna-small-32k-seqlen-hf \
  --MODEL_VARIANT HyenaDNAWithDropoutAndNorm \
  --run_name_prefix hyena-dna-mqtl-classifier \
  --WINDOW 1024 \
  --NUM_EPOCHS 2 \
  --LEARNING_RATE 5e-4 \
  --OPTIMIZER adam