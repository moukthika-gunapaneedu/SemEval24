import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load gold dev data
with open("dev.json", "r") as f:
    dev_data = json.load(f)

rows = []
for k, v in dev_data.items():
    rows.append({
        "id": str(k),                      
        "gold": float(v.get("average", 0.0)),
    })
gold_df = pd.DataFrame(rows)
gold_df["id"] = gold_df["id"].astype(str)

# 2. Load predictions from predict.py
pred_df = pd.read_csv("predictions.csv")      
pred_df["id"] = pred_df["id"].astype(str)  
pred_df = pred_df.rename(columns={"prediction": "pred"})

print("gold_df dtype:", gold_df["id"].dtype)
print("pred_df dtype:", pred_df["id"].dtype)

# 3. Merge on id
df = gold_df.merge(pred_df, on="id", how="inner")
print("Merged rows:", len(df))

# 4. Scatter plot: gold vs predicted
plt.figure()
plt.scatter(df["gold"], df["pred"], alpha=0.4)
plt.xlabel("Gold average plausibility")
plt.ylabel("Predicted plausibility")
plt.title("Dev set: gold vs predicted plausibility")
plt.tight_layout()
plt.savefig("figure_scatter_dev.png", dpi=300)

# 5. Error histogram
errors = df["pred"] - df["gold"]
plt.figure()
plt.hist(errors, bins=30)
plt.xlabel("Prediction - gold")
plt.ylabel("Count")
plt.title("Dev set error distribution")
plt.tight_layout()
plt.savefig("figure_error_hist_dev.png", dpi=300)
