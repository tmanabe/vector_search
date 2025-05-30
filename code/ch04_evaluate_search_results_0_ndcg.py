#!/usr/bin/env python

from sklearn.metrics import ndcg_score


# ESCIラベルとnDCGの利得の対応
ESCI_LABEL_TO_NDCG_GAIN = {"E": 1.0, "S": 0.01, "C": 0.1, "I": 0.0}


# ESCIラベルをnDCGの利得に変換する関数
def esci_label_to_ndcg_gain(esci_label):
    return ESCI_LABEL_TO_NDCG_GAIN[esci_label]


# nDCGを計算し表示する関数
def print_ndcg(data, k=None):
    total, query_count = 0.0, 0

    # クエリごとに計算する
    for _, group_by_query in data.groupby("query_id"):
        # 内部的には、scikit-learnの定義を使う
        total += ndcg_score(
            [group_by_query["esci_label"].apply(esci_label_to_ndcg_gain)],
            [group_by_query["score"]],
            k=k,
        )
        query_count += 1

    # クエリ間の平均を取り、小数点以下3桁まで表示する
    print(f"Mean nDCG: {total / query_count:.03f}")


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from ch02_basic_vectorization import read_basic_vectorized_data
    from ch03_vector_search_engines_0_numpy import cos

    # ベクトル化したデータセットをメモリに読み込む
    jp_data = read_basic_vectorized_data()

    # スコア（ここではコサイン類似度）を計算する
    jp_data["score"] = jp_data[["query_vector", "title_vector"]].apply(cos, axis=1)

    # nDCGを計算し表示する
    print_ndcg(jp_data)
