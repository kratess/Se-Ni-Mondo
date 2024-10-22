import os
import re
import pandas as pd

dataset = pd.read_csv('data/train.csv', usecols=['ID', 'Text', 'Label'])

labels = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0
}

ADJUSTMENT_FACTOR = 0.75


def get_label(value):
    if value > 0.334:
        return "positive"
    elif value > -0.334:
        return "neutral"
    else:
        return "negative"


def get_current_epoch() -> int:
    epoch_files = [f for f in os.listdir(
        'epochs') if re.match(r'epoch_\d+\.csv', f)]

    if epoch_files:
        epoch_numbers = [int(re.findall(r'\d+', f)[0]) for f in epoch_files]
        latest_epoch_number = max(epoch_numbers)

        return latest_epoch_number
    else:
        return -1


def predict(document_tokens: list[str], token_weight_dict: dict[str, float]):
    total_weight = 0
    token_count = 0

    for token in document_tokens:
        if token in token_weight_dict:
            total_weight += token_weight_dict[token]
            token_count += 1

    return total_weight / token_count if token_count > 0 else 0.5


def get_processed_dataset_data(csv_file: str) -> tuple[dict[any, list[str]], dict[any, str]]:
    dataset = pd.read_csv(f'data/{csv_file}__processed.csv')  # Col Label is optional

    dataset_id_tokens_dict = dict(
        zip(dataset['ID'], dataset['Tokens'].apply(lambda x: x.split(",") if pd.notnull(x) else []))
    )

    if 'Label' in dataset.columns:
        dataset_id_label_dict = dict(zip(dataset['ID'], dataset['Label']))
    else:
        dataset_id_label_dict = None

    return dataset_id_tokens_dict, dataset_id_label_dict


def get_dataset_data(csv_file: str) -> tuple[dict[any, str], dict[any, str] | None]:
    dataset = pd.read_csv(f'data/{csv_file}.csv')  # Col Label is optional

    dataset_id_text_dict = dict(zip(dataset['ID'], dataset['Text']))

    if 'Label' in dataset.columns:
        dataset_id_label_dict = dict(zip(dataset['ID'], dataset['Label']))
    else:
        dataset_id_label_dict = None

    return dataset_id_text_dict, dataset_id_label_dict


def get_epoch_data(epoch: int) -> dict[str, float]:
    epoch_data = pd.read_csv(
        f'epochs/epoch_{epoch}.csv', usecols=['Token', 'Weight'])
    token_weight_dict = pd.Series(
        epoch_data.Weight.values, index=epoch_data.Token).to_dict()

    return token_weight_dict


def calculate_scores(
        token_weight_dict: dict[any, float],
        id_tokens_dict: dict[any, list[str]],
        id_label_dict: dict[any, str]
) -> tuple[float, float]:
    true_positive, false_positive = 0, 0
    true_neutral, false_neutral = 0, 0
    true_negative, false_negative = 0, 0

    for id, tokens in id_tokens_dict.items():
        value = predict(tokens, token_weight_dict)
        label = get_label(value)
        true_label = id_label_dict[id]

        if label == "positive":
            if label == true_label:
                true_positive += 1
            else:
                false_positive += 1
        elif label == "neutral":
            if label == true_label:
                true_neutral += 1
            else:
                false_neutral += 1
        elif label == "negative":
            if label == true_label:
                true_negative += 1
            else:
                false_negative += 1

    total_correct = true_positive + true_negative + true_neutral
    total_predictions = (
        true_positive + false_positive +
        true_negative + false_negative +
        true_neutral + false_neutral
    )

    total_accuracy = total_correct / total_predictions

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0

    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0

    if recall + precision > 0:
        f1score = 2 * ((recall * precision) / (recall + precision))
    else:
        f1score = 0

    return total_accuracy, f1score
