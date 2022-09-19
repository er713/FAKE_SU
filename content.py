import numpy as np
from themes import (
    clean,
    create_word_cloud,
    create_bar_chart,
    get_grams_with_thresh,
    bayes_ngrams,
    to_lemma,
)
import pandas as pd
import swifter
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

swifter.set_defaults(force_parallel=True)

bins = [
    0,
    100,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    8000,
    10000,
    15000,
    20000,
    25000,
    # 50000,
    # 85000,
    # 130000,
]


def not_str(x):
    if not isinstance(x, str):
        return -1
    else:
        return len(x)


def check_logistic(data: pd.DataFrame):
    reg = LogisticRegression()
    lens = data["text"].to_numpy(np.int_).reshape(-1, 1)
    reg.fit(lens, data["label"])
    print("Logistic Reg accuracy:", accuracy_score(data["label"], reg.predict(lens)))
    print("Balans:", sum(data["label"]) / len(data["label"]))


if __name__ == "__main__":
    data = pd.read_csv("data/train.csv", index_col="id")
    data = data.sample(frac=0.7, random_state=2137)
    con_lab = data[["text", "label"]]
    print("Artykuły bez zawartości:", con_lab[con_lab["text"].isna()].shape)
    # print(data[con_lab['text'].isna()]['title'])
    con_lab = con_lab[con_lab["text"].notna()]

    len_lab = con_lab.copy()
    len_lab["text"] = con_lab["text"].map(len)

    len_lab = len_lab.sort_values("text")
    # print(len_lab.tail())
    print(len_lab.shape)
    print("Korelacja\t\t\t", len_lab["text"].corr(len_lab["label"]))
    print(
        "Korelacja spearmana\t",
        len_lab["text"].corr(len_lab["label"], method="spearman"),
    )

    con_lab["text"].swifter.apply(clean)
    len_lab["text"] = con_lab["text"].map(len)
    check_logistic(len_lab)
    len_lab = len_lab[len_lab["text"] > 10]
    check_logistic(len_lab)
    # exit()
    print("=======Po Clean=======")
    print("Korelacja\t\t\t", len_lab["text"].corr(len_lab["label"]))
    print(
        "Korelacja spearmana\t",
        len_lab["text"].corr(len_lab["label"], method="spearman"),
    )
    print(
        "Korelacja kendalla\t", len_lab["text"].corr(len_lab["label"], method="kendall")
    )
    len_lab["text"].hist(
        bins=bins,
        rwidth=0.9,
    )
    plt.savefig("results/context_len_symbols.png")
    plt.clf()
    print(len_lab["text"].describe())
    # plt.show()

    # con_lab["text"].swifter.progress_bar(enable=True).apply(to_lemma)
    # con_lab.to_csv("data/context_lemma_train.csv")
    con_lab = pd.read_csv("data/context_lemma_train.csv")
    len_lab["text"] = con_lab["text"].map(len)
    # len_lab = len_lab[len_lab["text"] > 10]
    print("=======Po Clean=======")
    print("Korelacja\t\t\t", len_lab["text"].corr(len_lab["label"]))
    print(
        "Korelacja spearmana\t",
        len_lab["text"].corr(len_lab["label"], method="spearman"),
    )
    print(
        "Korelacja kendalla\t", len_lab["text"].corr(len_lab["label"], method="kendall")
    )
    len_lab["text"].hist(
        bins=bins,
        rwidth=0.9,
    )
    plt.savefig("results/context_len_words.png")
    print(len_lab["text"].describe())

    # word cloud
    ngram = 3
    vec = CountVectorizer(
        strip_accents=None,
        lowercase=True,
        ngram_range=(ngram, ngram),
        min_df=150,
        stop_words=None if ngram == 1 else "english",
    )
    count = vec.fit_transform(con_lab["text"].to_numpy())
    tab_count = np.array(count.toarray(), dtype=np.int_)

    prob, quantity = bayes_ngrams(tab_count, con_lab["label"])
    names = np.array(vec.get_feature_names_out())
    su = np.sum(tab_count, axis=0)
    print(names, np.max(su), np.min(su), np.mean(su), np.median(su))
    print(
        quantity[prob > 0.99],
        np.mean(quantity[prob > 0.99]),
        np.median(quantity[prob > 0.99]),
    )
    print(names[prob > 0.99])

    th = 0.6
    w = get_grams_with_thresh(names, prob, threshold=th)
    print(w)
    print(len(w))
    ss = sorted(w.items(), key=lambda x: x[1], reverse=True)
    print(ss)
    create_word_cloud(w, save_file=f"results/context_wordcloud{ngram}")
    create_bar_chart(
        ss,
        max_items=11,
        threshold=th,
        label_size=20,
        save_file=f"results/context_bar_ngram{ngram}",
    )
