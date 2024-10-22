import pandas as pd
from tqdm import tqdm
from src.utils import get_processed_dataset_data, get_current_epoch, get_epoch_data, get_label, predict, calculate_scores


def test(csv_file: str = None):
    csv_file = csv_file or "test"

    dataset_id_tokens_dict, dataset_id_label_dict = get_processed_dataset_data(csv_file)
    current_epoch = get_current_epoch()

    token_weight_dict = get_epoch_data(current_epoch)

    if dataset_id_label_dict:
        accuracy, f1score = calculate_scores(
            token_weight_dict, dataset_id_tokens_dict, dataset_id_label_dict)
        print(f"Based on epoch {current_epoch}: [Acc: {accuracy:.4f} | F1: {f1score:.4f}]")
    else:
        print("Results are finished!")

    # Save results

    predicts = {id: get_label(predict(tokens, token_weight_dict))
                for id, tokens in dataset_id_tokens_dict.items()}

    predicts = {id: get_label(predict(tokens, token_weight_dict))
                for id, tokens in tqdm(
                    dataset_id_tokens_dict.items(),
                    desc="Predicting labels",
                    unit="predict")
                }

    token_weight_df = pd.DataFrame(predicts.items(), columns=['ID', 'Label'])

    token_weight_df.to_csv(
        f'data/{csv_file}__results.csv', index=False)
