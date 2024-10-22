import argparse
from tqdm import tqdm
from src.preprocessing import preprocess_corpus
from src.training import generate_epoch
from src.testing import test


def main():
    parser = argparse.ArgumentParser(description="Script with custom arguments")

    parser.add_argument('--preprocess', action='store_true', help='Preprocess the train_csv file')
    parser.add_argument('--train', action='store_true', help='Train the train_csv file')
    parser.add_argument('--test', action='store_true', help='Test the train_csv file')

    parser.add_argument('--csv_file', help='Use a custom csv_file')
    parser.add_argument('--epochs', help='Number of epochs')

    args = parser.parse_args()

    if args.preprocess:
        preprocess_corpus(csv_file=args.csv_file)

    if args.train:
        for i in tqdm(range(int(args.epochs or 1)), desc="Generating epochs", unit="epoch"):
            generate_epoch(csv_file=args.csv_file)

    if args.test:
        test(csv_file=args.csv_file)


if __name__ == "__main__":
    main()
