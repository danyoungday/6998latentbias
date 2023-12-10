import os

EPS = 1e-4
RESULTS_DIR = os.path.join(os.getcwd(), "results")
CROWSPAIRS_RESULTS_DIR = os.path.join(RESULTS_DIR, "crowspairs")
DATA_DIR = os.path.join(os.getcwd(), "data")
CROWSPAIRS_PATH = os.path.join(DATA_DIR, "crows_pairs_anonymized.csv")

GPT2_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
ROBERTA_MODELS = ["roberta-base", "roberta-large"]
FLAN_T5_MODELS = ["flan-t5-small", "flan-t5-base", "flan-t5-large"]
T5_MODELS = ["t5-small-enc", "t5-base-enc", "t5-large-enc"]
FILTERS = ["all", "age", "disability", "gender", "nationality", "physical-appearance", "race-color", "religion", "sexual-orientation", "socioeconomic"]