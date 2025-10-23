# SemEval 2026 Task 5

**Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding**

This repository contains our team’s work for **SemEval 2026 Task 5**, which focuses on modeling human-like understanding of **ambiguous word senses** within short narratives.  
Official task page: [https://nlu-lab.github.io/semeval.html](https://nlu-lab.github.io/semeval.html)


## Team 

| Name | GitHub Username |
|:--|:--|
| Collette | [@cohe3527-rgb](https://github.com/cohe3527-rgb) |
| Jacqui | [@jafa4364](https://github.com/jafa4364) |
| Moukthika | [@moukthika-gunapaneedu](https://github.com/moukthika-gunapaneedu) |

**GitHub Repository:** [https://github.com/moukthika-gunapaneedu/SemEval24](https://github.com/moukthika-gunapaneedu/SemEval24)


## Task Overview

Traditional **Word Sense Disambiguation (WSD)** assumes a single “correct” sense for a word, but real-world understanding is far more nuanced.  
**SemEval 2026 Task 5** aims to predict **human-perceived plausibility** of different word senses in context, modeling how humans interpret ambiguous meanings within a story.

The dataset, **AmbiStory**, consists of **five-sentence short stories**, each divided into:
1. **Precontext (3 sentences)** — sets up the narrative  
2. **Ambiguous Sentence (1 sentence)** — contains a homonym with multiple senses  
3. **Ending (optional)** — may imply one interpretation over another  

Each story is annotated by human participants who rate each possible word sense on a **scale from 1 to 5**, resulting in multiple samples per story.


## Task and Evaluation Metrics

The model must output a **plausibility score (1–5)** for a given word sense within a story.

| Metric | Description |
|:--|:--|
| **Spearman Correlation** | Measures correlation between model predictions and human mean ratings. |
| **Accuracy Within Standard Deviation** | Percentage of predictions within ±1 standard deviation of human scores. |

This encourages both **correlation with human intuition** and **robustness to ambiguity**.
