#!/usr/bin/env python

from ch02_basic_vectorization import (
    get_dimension_number_of,
    read_basic_vectorized_data,
    split_into_query_and_document,
)

import faiss
import numpy as np


# ベクトル化したデータセットをメモリに読み込み、クエリとドキュメントに分割する
query_data, document_data = split_into_query_and_document(
    read_basic_vectorized_data()
)

# データからベクトルの次元数を取得し、Faissインデックスを作成する
faiss_index = faiss.IndexFlatIP(get_dimension_number_of(query_data))

# ドキュメントを整形し入力する
faiss_index.add(np.vstack(document_data["title_vector"]))

# 例として、単一のクエリを取り出し、整形し、表示する
query_data = query_data[query_data.query_id == 119300]
print(query_data.to_string(columns=["query_id", "query"], index=False))

# クエリを整形し入力（つまり検索）する
ip_matrix, index_matrix = faiss_index.search(
    np.vstack(query_data["query_vector"]), 10
)

# 結果を、そのまま表示する
print(ip_matrix)
print(index_matrix)

# 結果を整形し表示する
for indices, ips in zip(index_matrix, ip_matrix):
    # ドキュメントのインデックスのリストを、ランキング結果のDataFrameに変換する
    ranking = document_data.iloc[indices].copy()
    # ランキング結果にスコア列を追加する
    ranking["score"] = ips
    print(
        ranking.to_string(
            columns=["score", "product_title"], index=False, max_colwidth=35
        )
    )
