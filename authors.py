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
    f_q = [np.mean(y_nan)]
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
    for n in (0, 1, 2, 3, 4, 5):
        print("================ >", n, "==============")
        tf = np.logical_and(wyn[:, 1] > n, wyn[:, 0] == wyn[:, 1])
        print("Wrodzeni kłamcy", np.sum(tf), np.sum(tf) / curr)

        tf = np.logical_and(wyn[:, 1] > n, wyn[:, 0] == 0)
        print("Idealni", np.sum(tf), np.sum(tf) / curr)

    # Multi
    for n in range(1, 7):
        mul = []
        mul_a = []
        # wyn = np.zeros((curr, 2), dtype=np.uint16)
        for at, l in zip(authors, y):
            if len(at) == n:
                mul.append(l)
                tmp = [0] * len(at)
                for i, a in enumerate(at):
                    tmp[i] = a2n[a]
                mul_a.append(tmp)

        # print(len(mul))
        f_q.append(np.mean(mul))
        print(f"Średnia fałszywych pisanych w grupie {n} autorów:", np.mean(mul), len(mul))

    plt.bar(np.arange(0, 7), f_q, width=0.7)
    plt.ylim([0., 1.])
    plt.title("Procent fałszywych artykułów z daną liczbą autorów")
    plt.xlabel("Liczba autorów")
    plt.show()

    # Proc od ilości napisanych artykułów
    maq = np.max(wyn[:, 1])+1
    p_qa = np.zeros((maq, ))
    p_qa[0] = np.nan
    i_pa = np.zeros((maq, ), dtype=np.int32)
    for n in range(1, maq):
        # print(n, np.mean(p_wyn[wyn[:, 1] == n]))
        p_qa[n] = np.mean(p_wyn[wyn[:, 1] == n])
        i_pa[n] = p_wyn[wyn[:, 1] == n].shape[0]
    # print(p_qa, np.max(wyn[:, 1]))
    print(p_qa[193], p_qa[240:249])
    wybie = np.logical_not(np.isnan(p_qa))
    X = np.arange(maq)[wybie]
    Y = p_qa[wybie]
    p_qa[np.isnan(p_qa)] = 0.0
    www = i_pa[i_pa > 0] > 9
    print(i_pa[i_pa > 0])
    plt.plot(X[www][:50], Y[www][:50])  # , width=0.6)
    plt.title("Średni procent fałszywych artykułów napisanych\nprzez autorów z konkretną liczbą artykułów")
    plt.ylim([0.0, np.max(Y[www][:50]) + 0.05])
    plt.xlabel("Liczba napisanych artykułów")
    plt.show()

    # plt.hist(X, bins=[1,2,3,4,5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 243])
    # plt.show()
    # print(i_pa)

    # nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    #
    # tokens = nlp("This is a horrible sentence.")
    # for t in tokens:
    #     print(len(t.vector))
    #     print(t.text, t.lemma_, t.sentiment)
