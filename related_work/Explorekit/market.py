import sys

sys.path.append('/GPUFS/ecnu_cqjin_caipeng/AutoFE')
sys.path.append('C:\\Users\\ZFY\\PycharmProjects\\AutoFE')
from related_work.Explorekit.dataset import dataset
from utils.init_seed import init_seed
from related_work.Explorekit.generater.generate_candidates import \
    generate_candidate_features_from_binary_group_by_transform, init_dataset_and_candidate_features
from related_work.Explorekit.evaluation.evalution import evaluate_a_regress_dataset
from related_work.Explorekit.rank.rank_candidate import rank_regress_candidate
from dataprocessing import dataset
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    init_seed()
    original_dataset, label = dataset.get_market_sales()
    maxIterations = 10
    threshold_w = 0.1

    print('init_dataset_and_candidate_features')
    dataset, candidate = init_dataset_and_candidate_features(original_dataset)
    print(f'init success. dataset shape {dataset.shape}, candidate shape {candidate.shape}, '
          f'original data shape{original_dataset.shape}')
    for i in range(maxIterations):
        rank_list = rank_regress_candidate(original_dataset, candidate, label)
        chosen_candidate_name = rank_list[0][0]
        improvement = rank_list[0][1]
        print(f'iteration {i}.............. {chosen_candidate_name}..........{improvement}')
        if improvement > threshold_w:
            original_dataset[chosen_candidate_name] = candidate[chosen_candidate_name]
            dataset[chosen_candidate_name] = candidate[chosen_candidate_name]
            candidate.drop(chosen_candidate_name, axis=1, inplace=True)
            new_candidate = generate_candidate_features_from_binary_group_by_transform(dataset, chosen_candidate_name)
            candidate[new_candidate.columns] = new_candidate
        else:
            break

    print(f'score:{evaluate_a_regress_dataset(original_dataset, label)}')
    print(original_dataset.columns)
