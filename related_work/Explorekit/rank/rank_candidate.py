from related_work.Explorekit.evaluation.evalution import evaluate_a_candidate, evaluate_a_dataset
import pandas as pd
from tqdm import tqdm


def rank_candidate(original_dataset: pd.DataFrame, candidate: pd.DataFrame, label):
    base_score = evaluate_a_dataset(original_dataset, label)
    rank = {}
    bar = tqdm(total=len(candidate.columns), desc='rank')
    for col in candidate.columns:
        score = evaluate_a_candidate(original_dataset, candidate[col], label)
        rank[col] = score - base_score
        bar.update()
    bar.close()

    return sorted(rank.items(), key=lambda item: item[1], reverse=True)
