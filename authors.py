from typing import List, Tuple, Iterable
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# import spacy


def divide_author(authors: str) -> Iterable[str]:
    aut = authors.split(' and ')
    if len(aut) > 1:
        return *aut[0].split(', '), aut[1]
    else:
        return tuple(aut[0].split(', '))


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv', index_col='id')
    aut_lab = data[['author', 'label']]
    aut_lab_nan = aut_lab[aut_lab['author'].isna()]
    aut_lab_clean = aut_lab[aut_lab['author'].notna()]
    # print(aut_lab_nan)
    y = aut_lab_clean['label'].to_numpy()
    print("Część fałszywych przy znanym autorze:", np.mean(y))
    y_nan = aut_lab_nan['label'].to_numpy()
    print("Część fałszywych przy nieznanym autorze:", np.mean(y_nan))
    authors = [divide_author(ats) for ats in aut_lab_clean['author']]
    qn = [len(ats) for ats in authors]
    print("Max, średnia i mediana liczby autorów dla jednego artykułu")
    print(np.max(qn), np.mean(qn), np.median(qn))

    a2n = dict()
    curr = 0
    for at in authors:
        for a in at:
            if a not in a2n.keys():
                a2n[a] = curr
                curr += 1

    wyn = np.zeros((curr, 2), dtype=np.uint16)
    for at, l in zip(authors, y):
        for a in at:
            inx = a2n[a]
            wyn[inx, 0] += l
            wyn[inx, 1] += 1

    n2a = [''] * curr
    for k, v in a2n.items():
        n2a[v] = k

    print("max, min i mediana liczy napisanych fałszywych artykułów przez jednego autora")
    print(np.max(wyn[:, 0]), np.min(wyn[:, 0]), np.median(wyn[:, 0]))
    print(np.argmax(wyn[:, 0]), np.argmin(wyn[:, 0]))
    p_wyn = wyn[:, 0] / wyn[:, 1]
    print("max, min i mediana części napisanych fałszywych artykułów przez jednego autora")
    print(np.max(p_wyn), np.min(p_wyn), np.median(p_wyn))
    print(np.argmax(p_wyn), np.argmin(p_wyn))
    # print(p_wyn)
    # print(wyn[:,1])
    # print(wyn[np.argmax(wyn[:, 0]), 1], wyn[np.argmin(wyn[:, 0]), 1], wyn[np.argmax(p_wyn), 1],
    #       wyn[np.argmin(p_wyn), 1])
    print("Największa liczba napisanych bezbłędnie i błędnie przez jednego autora")
    print(np.max(wyn[:, 1][wyn[:, 0] == 0]), np.max(wyn[:, 1][wyn[:, 0] == wyn[:, 1]]))

    # Śmieszki jakieś/debile
    for n in (1, 2, 3, 4, 5):
        print("================ >", n, "==============")
        tf = np.logical_and(wyn[:, 1] > n, wyn[:, 0] == wyn[:, 1])
        print("Wrodzeni kłamcy", np.sum(tf), np.sum(tf) / curr)

        tf = np.logical_and(wyn[:, 1] > n, wyn[:, 0] == 0)
        print("Idealni", np.sum(tf), np.sum(tf) / curr)


    # Multi
    mul = []
    mul_a = []
    wyn = np.zeros((curr, 2), dtype=np.uint16)
    for at, l in zip(authors, y):
        if len(at) == 1:
            continue
        mul.append(l)
        tmp = [0] * len(at)
        for i, a in enumerate(at):
            tmp[i] = a2n[a]
        mul_a.append(tmp)

    print("Średnia fałszywych pisanych w grupie", np.mean(mul))

    # nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    #
    # tokens = nlp("This is a horrible sentence.")
    # for t in tokens:
    #     print(len(t.vector))
    #     print(t.text, t.lemma_, t.sentiment)
