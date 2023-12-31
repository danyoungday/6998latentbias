# 6998latentbias

This is my (Daniel Young's) project for COMS E6998 002 Natural Language Generation and Summarization. It is exploring underlying biases in large language models via. [Contrast Consistent Search](https://arxiv.org/abs/2212.03827).

## Method
1. Hidden states are generated via. LLM in `hs.ipynb`
2. CCS model and logistic regression models are trained on hidden states in `train.ipynb`
3. Perplexities can be generated using `msp.ipynb`
4. Cross-result analysis can be done with `analysis.ipynb`

## Results
- Results are saved in structure `<Bias Type>/<Model Name>`
- Hidden states from `hs.ipynb` are stored in `saved/`
- Results for CCS and perplexity analysis from `train.ipynb` and `msp.ipynb` are stored in `results/crowspairs` and `results/msp` & `results/perp`

## Data
- Professions data is from [Man is to Computer Programmer as Woman is to Homemaker](https://arxiv.org/abs/1607.06520) saved in `professions.json`
- CrowS-Pairs data is from [CrowS-Pairs paper](https://aclanthology.org/2020.emnlp-main.154.pdf) We filter out antistereos.
    - The CrowS-Pairs dataset has a lot of errors that needs to be fixed.