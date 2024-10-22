import re
import pandas as pd
from tqdm import tqdm
from src.utils import get_dataset_data

df_stopwords = pd.read_csv('_data/stopwords.txt', header=None)
stopwords = df_stopwords[0].tolist()

df_firstnames = pd.read_csv('_data/italian_firstnames.txt', header=None)
firstnames = df_firstnames[0].tolist()

df_surnames = pd.read_csv('_data/italian_surnames.txt', header=None)
surnames = df_surnames[0].tolist()

df_lemmatization = pd.read_csv('_data/italian_lemmatization.csv', usecols=['Token', 'Lemma'])
token_lemma_dict = dict(zip(df_lemmatization['Token'], df_lemmatization['Lemma']))


def _reform_italian_words(document: str) -> str:
    document = document.replace("á", "à")
    document = document.replace("Á", "À")

    # no worry for "perché". it is removed afterwards as stopword
    document = document.replace("é", "è")
    document = document.replace("É", "È")

    document = document.replace("ú", "ù")
    document = document.replace("Ú", "Ù")

    document = document.replace("pk", "perché")
    document = document.replace("xche", "perché")
    document = document.replace("xché", "perché")
    document = document.replace("xchè", "perché")
    document = document.replace("nn", "non")

    return document


def _convert_emoticons_to_emojis(text):
    emoticon_to_emoji = {
        ":)": "😊",
        ":')": "😊",
        ":-)": "😊",
        ":(": "😢",
        ":'(": "😢",
        ":-(": "😢",
        ":D": "😀",
        ":-D": "😀",
        ";)": "😉",
        ";-)": "😉",
        ":'(": "😢",
        ":'-)": "😢",
        ":O": "😮",
        ":-O": "😮",
        ":/": "😕",
        ":-/": "😕",
        ">:(": "😠",
        ":*": "😘",
        ":-*": "😘",
        "8-)": "😎",
        "B-)": "😎",
        ":P": "😜",
        ":-P": "😜",
        "|-O": "😴",
        ":-O": "😮",
        ":S": "😖",
        "<3": "❤️",
        ";P": "😜",
        ";-)P": "😜",
        ">:O": "😡",
        ">:-O": "😡",
        ">:)": "😏",
        ":S": "😕",
        "xD": "😂",
        "XD": "😂"
    }

    # Replace emoticons with emojis
    for emoticon, emoji in emoticon_to_emoji.items():
        text = text.replace(emoticon, emoji)

    return text


def _remove_unwanted(document: str) -> str:
    document = re.sub("@[A-Za-z0-9_]+", "", document)  # remove user mentions
    document = re.sub("#[A-Za-z0-9_]+", "", document)  # remove hashtags
    document = re.sub(r'http\S+', ' ', document)  # remove URLS
    document = re.sub(r'\d+', ' ', document)  # remove digits
    # remove punctuation but keep italian letters and emojis
    document = re.sub(r"[^A-Za-z\sÀ-ÿ\u263a-\U0001F999]", " ", document)

    return document.strip()


def _add_space_before_non_alphanumeric(document: str) -> str:
    """
    Adds a space before each character that is not alphanumeric.
    This function is used because at this points all non-alpha chars are emojis.
    """
    return re.sub(r'([^A-Za-zÀ-ÿ])', r' \1', document).strip()


def _remove_multiple_spaces(document: str) -> str:
    # transform multiple spaces into one
    return re.sub(r'\s+', ' ', document).strip()


def _remove_names(document_texts: list[str]) -> list[str]:
    filtered_texts = [
        text for text in document_texts if text not in firstnames and text not in surnames]

    return filtered_texts


def _remove_stop_words(document_texts: list[str]) -> list[str]:
    filtered_texts = [
        text for text in document_texts if text not in stopwords]

    return filtered_texts


def _replace_laughter(document_texts: list[str]) -> list[str]:
    pattern = r'^[ha]+$'
    document_texts = [re.sub(pattern, "ahah", word) if word != "ha" and re.match(
        pattern, word) else word for word in document_texts]

    return document_texts


def _remove_single_chars(document_texts: list[str]) -> list[str]:
    return [text for text in document_texts if len(text) > 1 or not text.isalpha()]


def _get_lemmas(document_texts: list[str]) -> list[str]:
    lemmas = [token_lemma_dict.get(text, text) for text in document_texts]

    return lemmas


def _save_corpus_file(csv_file: str, lemmas_corpus: dict[any, list[str]], dataset_id_label_dict: dict[any, str] | None):
    if dataset_id_label_dict is not None:
        corpus_data = pd.DataFrame(
            [
                (k, ','.join(v), dataset_id_label_dict.get(k, ''))
                for k, v in lemmas_corpus.items()
            ], columns=['ID', 'Tokens', 'Label']
        )
    else:
        corpus_data = pd.DataFrame(
            [
                (k, ','.join(v))
                for k, v in lemmas_corpus.items()
            ], columns=['ID', 'Tokens']
        )

    corpus_data.to_csv(f'data/{csv_file}__processed.csv', index=False)


def preprocess_corpus(csv_file: str = None) -> tuple[dict[any, list[str]], set[str]]:
    csv_file = csv_file or "train"
    dataset_id_text_dict, dataset_id_label_dict = get_dataset_data(csv_file)

    lemmas_corpus: dict[any, list[str]] = {}
    tokens = set()

    for id, document in tqdm(dataset_id_text_dict.items(), desc="Preprocessing corpus", unit="document"):
        if not isinstance(document, str):
            lemmas_corpus[id] = []
            continue

        document = _reform_italian_words(document)
        document = _convert_emoticons_to_emojis(document)
        document = _remove_unwanted(document)
        document = _add_space_before_non_alphanumeric(document)
        document = _remove_multiple_spaces(document)

        document_texts = document.split()
        document_texts = _remove_names(document_texts)
        document_texts = [text.lower() for text in document_texts]
        document_texts = _remove_stop_words(document_texts)
        document_texts = _replace_laughter(document_texts)
        document_texts = _remove_single_chars(document_texts)
        lemmas = _get_lemmas(document_texts)
        lemmas_corpus[id] = lemmas

        tokens.update(lemmas)

    _save_corpus_file(csv_file, lemmas_corpus, dataset_id_label_dict)

    print(f"Preprocessing finished! Number of tokens: {len(tokens)}")

    return lemmas_corpus, tokens
