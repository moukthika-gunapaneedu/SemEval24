# SemEval 2026 Task 5

**Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding**

This repository contains our team’s work for **SemEval 2026 Task 5**, which focuses on modeling human-like understanding of ambiguous word senses within short narratives.

Official task page:  
https://nlu-lab.github.io/semeval.html


# Team

| Name | GitHub Username |
|------|----------------|
| Collette | https://github.com/cohe3527-rgb |
| Jacqui | https://github.com/jafa4364 |
| Moukthika | https://github.com/moukthika-gunapaneedu |

GitHub repository:  
https://github.com/moukthika-gunapaneedu/SemEval24

Team Google Doc (notes + weekly updates):  
https://docs.google.com/document/d/1tH4uvhLNYkfRF5qeAqJjWVGmyQTr49YS6z71R_BDUko/edit?usp=sharing


# Task Overview

Traditional Word Sense Disambiguation assumes a single correct meaning for a word, but humans often find multiple meanings plausible depending on context.

**SemEval 2026 Task 5** evaluates a model’s ability to assign a *plausibility score (1–5)* to different word senses within a short narrative.

Dataset: **AmbiStory**, a five-sentence story:

1. Precontext (3 sentences)  
2. Ambiguous sentence  
3. Ending sentence  

Each story includes multiple human ratings for each possible sense (1–5).


# Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Spearman Correlation | Correlation between human mean ratings and model predictions. |
| Accuracy Within Standard Deviation | If predicted score is within ±1 SD of human ratings. |


# Repository Structure

| File/Folder | Purpose |
|-------------|----------|
| model/ | Saved model checkpoints (ignored in GitHub). |
| train.json | Training data. |
| dev.json | Development data. |
| sample_data.json | Small sample dataset for quick testing. |
| train.py | Full training script. |
| predict.py | Script for generating predictions. |
| requirements.txt | Python dependencies. |
| .gitignore | Prevents large files from uploading. |
| README.md | Documentation. |


# How to Run

1. Install dependencies

`pip install -r requirements.txt`


2. Train the model

`python train.py`

This trains a RoBERTa-based regression model on the training set.

3. Generate predictions

When the test set is released:
`python predict.py --input_file test.json --output_file predictions.csv`

To test using the development set:

`python predict.py --input_file dev.json --output_file dev_predictions.csv`



# Model Summary

- Base model: RoBERTa-base  
- Task type: regression  
- Outputs: one numerical score (1–5)  
- Includes:  
  - tokenization  
  - regression head  
  - MSE loss  
  - Spearman correlation metric  
  - Accuracy-within-SD metric  
  - checkpoint saving  


# Future Work

- Run predictions once the test dataset is released.  
- Add results tables, charts, and error analysis to the report.  


