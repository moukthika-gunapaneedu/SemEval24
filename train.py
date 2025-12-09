import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from scipy.stats import spearmanr

### ------------------------
### Load JSON as DataFrame
### ------------------------

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for k, v in data.items():
        rows.append({
            "id": k,
            "precontext": v.get("precontext", ""),
            "sentence": v.get("sentence", ""),
            "ending": v.get("ending", ""),
            "judged_meaning": v.get("judged_meaning", ""),
            "average": float(v.get("average", 0.0)),
        })
    return pd.DataFrame(rows)


### ------------------------
### Dataset Class
### ------------------------

class SemEvalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        text = (
            f"{r['precontext']} "
            f"{r['sentence']} "
            f"{r['ending']} "
            f"[MEANING] {r['judged_meaning']}"
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len
        )

        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(r["average"], dtype=torch.float32)
        return item


### ------------------------
### Metrics (MSE + Spearman + Accuracy within SD)
### ------------------------

def compute_metrics(eval_pred):
    """
    Computes:
      - MSE (mean squared error)
      - Spearman correlation between predictions and gold labels
      - Accuracy within ~1 std-dev (here approximated as |pred - gold| <= 0.75)
    """
    preds, labels = eval_pred

    # preds shape: (N, 1) for regression, labels shape: (N, 1)
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    # Mean Squared Error
    mse = float(np.mean((preds - labels) ** 2))

    # Spearman correlation
    sp, _ = spearmanr(preds, labels)
    if np.isnan(sp):
        sp = 0.0

    # "Accuracy within 1 SD" (approx): treat 0.75 as 1 std dev band
    sigma = 0.75
    acc_within_sd = float(np.mean(np.abs(preds - labels) <= sigma))

    return {
        "mse": mse,
        "spearman": sp,
        "acc_within_sd": acc_within_sd,
    }



### ------------------------
### Training Code
### ------------------------

def main():
    train_df = load_json("train.json")
    dev_df = load_json("dev.json")

    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = SemEvalDataset(train_df, tokenizer)
    dev_ds = SemEvalDataset(dev_df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression"
    )

    training_args = TrainingArguments(
    output_dir="./model",
    do_train=True,
    do_eval=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_metrics,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_metrics,
)

    trainer.train()

    # NEW: run evaluation on dev and save metrics
    eval_results = trainer.evaluate()
    print("Dev metrics:", eval_results)

    import json as _json
    with open("dev_metrics.json", "w") as f:
        _json.dump(eval_results, f, indent=2)

    trainer.save_model("./model")


if __name__ == "__main__":
    main()