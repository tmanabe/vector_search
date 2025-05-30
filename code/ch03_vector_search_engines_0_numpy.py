#!/usr/bin/env python

import numpy as np


# データの行 (row) に対してコサイン類似度 (cos) を計算する関数
def cos(row):
    # コサイン類似度は2ベクトルの間に定義される。2ベクトルが2列として渡されているはず。
    vector0, vector1 = row

    # 計算して結果を返す
    return np.dot(vector0, vector1) / (
        np.linalg.norm(vector0) * np.linalg.norm(vector1)
    )


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from ch02_basic_vectorization import read_basic_vectorized_data

    # ベクトル化したデータセットをメモリに読み込む
    jp_data = read_basic_vectorized_data()

    # クエリとドキュメントの両ベクトルのコサイン類似度を計算しスコアとする
    jp_data["score"] = jp_data[["query_vector", "title_vector"]].apply(cos, axis=1)

    # クエリ (ID) ごとにドキュメント（データセットの各行）をスコアの降順ソートする。
    # まれにあるスコアが等しいものはデータセットの順序のままにする（安定ソートする）。
    jp_data.sort_values(
        inplace=True,
        by=["query_id", "score"],
        ascending=False,
        kind="stable",
    )

    # 例として、特定のクエリに対するランキング結果を表示する
    print(
        jp_data[jp_data.query_id == 119300].to_string(
            columns=["esci_label", "query", "score", "product_title"],
            index=False,
            max_colwidth=25,
        )
    )
