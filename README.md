# mqtl-classification

* To process long sequences, please use --model_type dnalong, and set the max sequence length as a multiple of 512 (e.g., 3072). Then the model should work well. [DNABERT github issues](https://github.com/jerryji1993/DNABERT/issues/18#issuecomment-823707084)

```bash
python3 main_hyenadna.py \
  --MODEL_NAME LongSafari/hyenadna-small-32k-seqlen-hf \
  --run_name_prefix hyena-dna-mqtl-classifier \
  --WINDOW 1024 \
  --NUM_EPOCHS 50
```
