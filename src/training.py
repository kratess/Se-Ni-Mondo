import pandas as pd
from src.utils import labels, ADJUSTMENT_FACTOR, get_processed_dataset_data, get_current_epoch, get_epoch_data, predict, calculate_scores


def _setup_epoch_zero(dataset_id_tokens_dict: dict[any, list[str]]):
    tokens_weight_dict = {token: 0.0 for tokens in dataset_id_tokens_dict.values()
                          for token in tokens}
    tokens_data = pd.DataFrame(
        list(tokens_weight_dict.items()), columns=['Token', 'Weight'])
    tokens_data.to_csv('epochs/epoch_0.csv', index=False)


def _compute_mse(y_true: dict[str, float], y_pred: dict[str, float]) -> float:
    total_error = 0
    num = len(y_true)

    for key in y_true:
        if key in y_pred:
            true_value = y_true[key]
            pred_value = y_pred[key]
            total_error += (true_value - pred_value) ** 2

    return total_error / num


def _compute_gradients(
    token_weight_dict: dict[str, float],
    y_true: dict[str, float],
    y_pred: dict[str, float],
    dataset_id_tokens_dict: dict[any, list[str]]
) -> dict[str, float]:
    gradients = {token: 0 for token in token_weight_dict.keys()}
    num = len(y_true)

    for i, tweet_tokens in dataset_id_tokens_dict.items():
        if i in y_true and i in y_pred:
            error = y_pred[i] - y_true[i]

            for token in tweet_tokens:
                if token in token_weight_dict:
                    gradients[token] += (2 / num) * error

    return gradients


def _update_weights(
    token_weight_dict: dict[str, float],
    gradients: dict[str, float]
) -> dict[str, float]:
    for token in token_weight_dict:
        if token in gradients:
            token_weight_dict[token] -= ADJUSTMENT_FACTOR * gradients[token]

    return token_weight_dict


def generate_epoch(csv_file: str = None):
    csv_file = csv_file or "train"

    dataset_id_tokens_dict, dataset_id_label_dict = get_processed_dataset_data(csv_file)
    current_epoch = get_current_epoch()

    if current_epoch == -1:
        _setup_epoch_zero(dataset_id_tokens_dict)
        current_epoch = 0

    token_weight_dict = get_epoch_data(current_epoch)

    y_true = {id: labels[label] for id, label in dataset_id_label_dict.items()}
    y_pred = {id: predict(tokens, token_weight_dict)
              for id, tokens in dataset_id_tokens_dict.items()}

    mse = _compute_mse(y_true, y_pred)
    gradients = _compute_gradients(token_weight_dict, y_true, y_pred, dataset_id_tokens_dict)
    weights = _update_weights(token_weight_dict, gradients)

    current_epoch = current_epoch+1

    accuracy, f1score = calculate_scores(
        weights, dataset_id_tokens_dict, dataset_id_label_dict)
    print(
        f"Epoch {current_epoch} finished! [Acc: {accuracy:.4f} | F1: {f1score:.4f} | MSE: {mse:.4f}]")

    # Save epoch

    token_weight_df = pd.DataFrame(weights.items(), columns=['Token', 'Weight'])

    token_weight_df.to_csv(
        f'epochs/epoch_{current_epoch}.csv', index=False)
