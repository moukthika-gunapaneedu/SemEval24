import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            "judged_meaning": v.get("judged_meaning", "")
        })
    return pd.DataFrame(rows)

def format_text(r):
    return (
        f"{r['precontext']} "
        f"{r['sentence']} "
        f"{r['ending']} "
        f"[MEANING] {r['judged_meaning']}"
    )

def main():
    df = load_json("dev.json")  # change to test.json for final submission

    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForSequenceClassification.from_pretrained("./model").to(device)
    model.eval()

    preds = []

    for _, r in df.iterrows():
        text = format_text(r)
        enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            out = model(**enc).logits.squeeze().item()

        preds.append(out)

    df["prediction"] = preds
    df[["id", "prediction"]].to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()