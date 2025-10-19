#!/usr/bin/env python

from ch02_basic_vectorization import split_into_query_and_document
from ch04_evaluate_search_results_0_ndcg import print_ndcg

import numpy as np
import pandas as pd
import time


# Faissが返したスコアをpandas（pd）DataFrameにまとめる関数
def format_score_data(k, query_ids, product_ids, score_matrix, index_matrix):
    # Faissはプレースホルダとして-1を返す。これはPythonでは最後の要素を指すので、
    # ユニークな製品ID列の最後の要素としてもプレースホルダ（None）を結合する
    product_ids = pd.concat([product_ids, pd.Series([None])])

    # スコアと、製品IDの通し番号それぞれ「リストのリスト」を単一のリストに結合する
    scores, indices = score_matrix.ravel(), index_matrix.ravel()

    return pd.DataFrame(
        {
            # Faissがクエリあたりk件のドキュメントを返すなら、クエリIDをk回ずつ繰り返す
            "query_id": np.repeat(list(query_ids), k),
            # 製品IDの通し番号のリスト（indices）を対応する製品ID列に変換する
            "product_id": product_ids.iloc[indices],
            # のちのソートを考慮して、プレースホルダに対するスコアを低い値に置換する
            "score": [
                -1e10 if index < 0 else score for score, index in zip(scores, indices)
            ],
        }
    )


# ラベルとFaissが返したスコアを単一のDataFrameにまとめる関数
def merge_label_and_score_data(label_data, score_data):
    return pd.merge(
        label_data,
        score_data,
        # いずれかのDataFrameにしか存在しない（クエリID、製品ID）のペアも残す（外部結合）
        on=["query_id", "product_id"],
        how="outer",
        # 不明なラベルはI（無関連）、のちのソートを考慮して不明なスコアは最低の値に置換する
    ).fillna({"esci_label": "I", "score": -2e10})


# Faissの検索結果の平均nDCGを計算し表示する関数
def print_ndcg_for_faiss(
    k, label_data, query_ids, product_ids, score_matrix, index_matrix
):
    # Faissが返したスコアをDataFrameにまとめる
    score_data = format_score_data(
        k, query_ids, product_ids, score_matrix, index_matrix
    )

    # nDCGの利得とFaissが返したスコアを単一のDataFrameにまとめる
    merged_data = merge_label_and_score_data(label_data, score_data)

    # 平均nDCGを計算し表示する
    print_ndcg(merged_data, k)


# Faissのインデックスをテストする関数。今後よく使うので関数にまとめた
def test_faiss(faiss_index, label_data, k, returns_distance=False):
    # わかりやすいようにFaissインデックスのクラス名を表示する
    print(faiss_index.__class__.__name__)

    # データセットをクエリとドキュメントに分割する
    query_data, document_data = split_into_query_and_document(label_data)

    # ドキュメントベクトルを整形し入力する
    faiss_index.add(np.vstack(document_data["title_vector"]))

    # クエリベクトルを整形し入力（つまり検索）する。また処理にかかった時間も測定する
    search_started_at = time.perf_counter()
    score_matrix, index_matrix = faiss_index.search(
        np.vstack(query_data["query_vector"]), k
    )
    search_finished_at = time.perf_counter()

    # 処理にかかった時間を計算し、マイクロ秒（小数点以下6桁）まで表示する
    print(f"Took: {search_finished_at - search_started_at:.06f} s")

    # 距離を返すインデックスの場合は、距離の符号を反転させてスコアとする
    if returns_distance:
        score_matrix *= -1

    # 平均nDCGを計算し表示する
    print_ndcg_for_faiss(
        k,
        label_data,
        query_data["query_id"],
        document_data["product_id"],
        score_matrix,
        index_matrix,
    )


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from ch02_basic_vectorization import (
        get_dimension_number_of,
        read_basic_vectorized_data,
    )

    import faiss

    # ベクトル化したデータセットをメモリに読み込む
    jp_data = read_basic_vectorized_data()

    # データからベクトルの次元数を取得し、総当たりのFaissインデックスを作成する
    faiss_index = faiss.IndexFlatIP(get_dimension_number_of(jp_data))

    # Faissインデックスをテストする
    test_faiss(faiss_index, jp_data, k=10)
