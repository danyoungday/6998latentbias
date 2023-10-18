# 6998latentbias

This is my (Daniel Young's) project for CS E6998 002 Natural Language Generation and Summarization. It is exploring underlying biases in large language models via. [Contrast Consistent Search](https://arxiv.org/abs/2212.03827).

## Method
1. Hidden states are generated via. LLM in `hs.ipynb`
2. CCS model and logistic regression models are trained on hidden states in `train.ipynb`

## Data
- Professions data is from [Man is to Computer Programmer as Woman is to Homemaker](https://arxiv.org/abs/1607.06520) saved in `professions.json`
- Hidden states are stored in `results/`
- F1 score outputs are stored in `saved/`