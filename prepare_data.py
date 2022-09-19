import pandas as pd
import swifter
import numpy as np
import spacy
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

# import kaggle

from authors import divide_author
from themes import bayes_ngrams, clean, to_lemma

swifter.set_defaults(force_parallel=True)
# nlp = spacy.load("en_core_web_lg")


def download_data():
    kaggle.api.competition_download_files("fake-news", "data/")


def _max_author(auth_list):
    return max([authors[author] for author in auth_list])


global authors
authors = None


def author_features(data: pd.DataFrame):
    data = data[data["author"].notna()]
    data["list_authors"] = data["author"].copy().swifter.apply(divide_author)
    data["qt_authors"] = data["list_authors"].copy().swifter.apply(len)

    global authors
    if authors is None:
        authors = defaultdict(lambda: 0)
        for author_list in data["list_authors"]:
            for author in author_list:
                authors[author] += 1
    data["best_at_art"] = data["list_authors"].copy().swifter.apply(_max_author)
    data.drop(columns=["list_authors"], inplace=True)
    print(data.tail())
    return data, ["qt_authors", "best_at_art"]


global vec, prob
vec, prob = None, None


def text_features(texts: pd.Series, labels: pd.Series, skip=False):
    print(texts.head())
    if texts.name == "title":
        print("TITLE")
        min_df = 15
    else:
        print("TEXT")
        min_df = 150
    texts = texts.swifter.apply(clean)
    texts = texts.swifter.apply(to_lemma)
    global vec
    if vec is None:
        vec = CountVectorizer(
            strip_accents=None,
            lowercase=True,
            ngram_range=(1, 3),
            min_df=min_df,
            stop_words="english",
        )
        print(texts)
        print(texts.map(" ".join).to_numpy())
        counts = vec.fit_transform(texts.map(" ".join).to_numpy())
    else:
        counts = vec.transform(texts.map(" ".join).to_numpy())
    counts = np.asarray(counts.toarray(), dtype=np.int_)
    global prob
    if prob is None:
        prob, _ = bayes_ngrams(counts, labels)
    if skip:
        return None, None
    res = []
    for line in counts:
        tmp = [[p] * qt for qt, p in zip(line, prob) if qt > 0]
        res.append([i for l in tmp for i in l])

    return pd.Series(res), texts.copy().swifter.apply(len)


def vec_features():
    pass


def prepare_base():
    data = pd.read_csv("data/train.csv", index_col="id")
    print(data.shape[0])
    data = data[data["title"].notna()]
    data = data[data["text"].notna()]
    data_train = data.sample(frac=0.7, random_state=2137)
    data_test = data.copy()
    data_test.drop(data_train.index)
    _, indexes = author_features(data_train)
    data_test, _ = author_features(data_test)

    tmp_prob, tmp_len = text_features(
        data_train["title"], data_train["label"], skip=True
    )
    # data_train["title_ngram_prob"], data_train["title_len"] = tmp_prob, tmp_len
    tmp_prob, tmp_len = text_features(data_test["title"], data_test["label"])
    data_test["title_ngram_prob"], data_test["title_len"] = tmp_prob, tmp_len
    # global vec
    # global prob
    vec, prob = None, None
    tmp_prob, tmp_len = text_features(
        data_train["text"], data_train["label"], skip=True
    )
    # data_train["text_ngram_prob"], data_train["text_len"] = tmp_prob, tmp_len
    tmp_prob, tmp_len = text_features(data_test["text"], data_test["label"])
    data_test["text_ngram_prob"], data_test["text_len"] = tmp_prob, tmp_len

    print(data_train.tail())
    print(data_train.shape[0] + data_test.shape[0])

    # data_train.to_csv("data/train_prob.csv")
    data_test.to_csv("data/test_prob.csv")


if __name__ == "__main__":
    # prepare_base()
    # exit()
    N_BEST = 20
    name = "train"
    data = pd.read_csv(f"data/{name}_prob_reduced.csv")
    whole = []
    for row in zip(
        data["qt_authors"],
        data["best_at_art"],
        data["title_ngram_prob"],
        data["title_len"],
        data["text_ngram_prob"],
        data["text_len"],
        data["label"],
    ):
        tmp = []
        for _id in (
            0,
            1,
            3,
            5,
        ):
            # print(row)
            if type(row[_id]) != int and type(row[_id]) != float:
                print(type(row[_id]), row[_id])
                raise Exception
            tmp.append(row[_id])
        for _id in (2, 4):
            if type(row[_id]) == str:
                r = [float(x) for x in row[_id][1:-1].split(", ") if len(x) > 0]
            else:
                r = row[_id]
            if r is None:
                tmp += [0.5] * N_BEST
            elif type(r) != list:
                if np.isnan(r):
                    tmp += [0.5] * N_BEST
                else:
                    if r > 0.5:
                        tmp += [r] + [0.5] * (N_BEST - 1)
                    else:
                        tmp += [0.5] * (N_BEST - 1) + [r]
            else:
                s = sorted(r, reverse=True)
                sb, se = s[: N_BEST // 2], s[-N_BEST // 2 :]
                d = N_BEST - len(sb) - len(se)
                tmp += sb + [0.5] * d + se
        tmp.append(row[6])
        whole.append(tmp)
    data = np.array(whole, np.float_)
    print(data.shape)
    np.save(f"data/{name}2.npy", data)

    exit()
    data_test = data_test[
        [
            "id",
            "qt_authors",
            "best_at_art",
            "title_ngram_prob",
            "title_len",
            "text_ngram_prob",
            "text_len",
            "label",
        ]
    ]
    data_test.to_csv("data/test_prob_reduced.csv")
