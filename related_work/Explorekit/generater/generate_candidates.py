from related_work.Explorekit.generater.operator_list import unary_operator_list, binary_operator_list, \
    consider_order, group_by_operator_list
import pandas as pd
from itertools import combinations, permutations

threshold = 50


def generate_candidate_features_from_unary_transform(df: pd.DataFrame):
    columns = df.columns
    candidates = pd.DataFrame()

    # unary operator
    for col in columns:
        # if  discrete, do not perform unary transform
        # if len(df[col].unique()) <= threshold:
        #     continue
        # perform unary transform
        for operator in unary_operator_list:
            candidates[f'{col}_{operator}'] = unary_operator_list[operator](df[[col]])

    return candidates


def generate_candidate_features_from_binary_group_by_transform(df: pd.DataFrame, new_feature: str):
    columns = [i for i in df.columns if i != new_feature]
    candidates = pd.DataFrame()
    # perform binary transform
    for col in columns:
        # preform group by transform
        if len(df[col].unique()) <= threshold:
            for operator in group_by_operator_list:
                candidates[f'{col}_{new_feature}_{operator}'] = group_by_operator_list[operator](df, col, new_feature)
        elif len(df[new_feature].unique()) <= threshold:
            for operator in group_by_operator_list:
                candidates[f'{new_feature}_{col}_{operator}'] = group_by_operator_list[operator](df, new_feature, col)

        # perform binary transform
        else:
            for operator in binary_operator_list:
                if not (operator == 'division' and 0 in df[new_feature].values):
                    candidates[f'{col}_{new_feature}_{operator}'] \
                        = binary_operator_list[operator](df[col], df[new_feature])
                if operator in consider_order:
                    if not (operator == 'division' and 0 in df[col].values):
                        candidates[f'{new_feature}_{col}_{operator}'] \
                            = binary_operator_list[operator](df[new_feature], df[col])

    return candidates


def init_dataset_and_candidate_features(original_dataset: pd.DataFrame):
    dataset = original_dataset.copy()
    candidate = generate_candidate_features_from_unary_transform(dataset)
    dataset[candidate.columns] = candidate
    columns = dataset.columns
    new_candidate = {}
    for col in columns:
        new_candidate[col] = generate_candidate_features_from_binary_group_by_transform(dataset, col)
    for col in columns:
        new = new_candidate[col]
        candidate[new.columns] = new
    return dataset, candidate
    # combination = list(combinations(columns, 2))

    # for i in combination:
    #     # if discrete, do not perform
    #     if len(df[i[0]].unique()) <= threshold or len(df[i[1]].unique()) <= threshold:
    #         continue
    #
    #     for operator in binary_operator_list:
    #         if operator == 'division' and 0 in df[i[1]].values:
    #             continue
    #         candidates[f'{i[0]}_{i[1]}_{operator}'] \
    #             = binary_operator_list[operator](df[i[0]], df[i[1]])
    #         if operator in consider_order:
    #             if operator == 'division' and 0 in df[i[0]].values:
    #                 continue
    #             candidates[f'{i[1]}_{i[0]}_{operator}'] \
    #                 = binary_operator_list[operator](df[i[1]], df[i[0]])
    #
    # # perform group by transform
    # for i in combination:
    #     for operator in group_by_operator_list:
    #         if len(df[i[0]].unique()) <= threshold < len(df[i[1]].unique()):
    #             candidates[f'{i[0]}_{i[1]}_{operator}'] = group_by_operator_list[operator](df, i[0], i[1])
    #         if len(df[i[1]].unique()) <= threshold < len(df[i[0]].unique()):
    #             candidates[f'{i[1]}_{i[0]}_{operator}'] = group_by_operator_list[operator](df, i[1], i[0])