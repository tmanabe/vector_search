#!/usr/bin/env python

from argparse import ArgumentParser
from ch02_basic_vectorization import (
    get_dimension_number_of,
    split_into_query_and_document,
)
from ch03_vector_search_engines_2_os import (
    get_default_index_options,
    OpenSearchTester,
)
from ch07_vector_compression_0_data import read_tuned_vectorized_data

import faiss
import numpy as np


# コマンドライン引数を読み込む
argument_parser = ArgumentParser()
# ドキュメントを分別するセントロイド数（クラスタ数でもある）
argument_parser.add_argument("--number-of-centroids", default=40, type=int)
# ドキュメントあたりOpenSearchに紐づけさせるセントロイド数
argument_parser.add_argument(
    "--number-of-centroids-per-document", default=2, type=int
)
# クエリあたりOpenSearchに引き当てさせるセントロイド数
argument_parser.add_argument("--number-of-centroids-per-query", default=2, type=int)
args = argument_parser.parse_args()

# ベクトル化したデータセットをメモリに読み込む
jp_data = read_tuned_vectorized_data()

# データからベクトルの次元数を取得する
dimension_number = get_dimension_number_of(jp_data)

# クラスタリング対象は訓練データ上のドキュメント（ここでは製品タイトル）ベクトル列とする
vectors_to_cluster = np.vstack(jp_data[jp_data.split == "train"]["title_vector"])

# ベクトル列をクラスタリングし、セントロイドのインデックスを作成する
faiss_k_means = faiss.Kmeans(
    dimension_number, args.number_of_centroids, spherical=True, seed=0
)
faiss_k_means.train(vectors_to_cluster)

# 各クエリベクトルに最寄りのセントロイドIDのリストを紐づける
_, index_matrix = faiss_k_means.index.search(
    np.vstack(jp_data.query_vector), args.number_of_centroids_per_query
)
jp_data["query_centroids"] = list(index_matrix)

# 各ドキュメントベクトルに最寄りのセントロイドIDのリストを紐づける
_, index_matrix = faiss_k_means.index.search(
    np.vstack(jp_data.title_vector), args.number_of_centroids_per_document
)
jp_data["title_centroids"] = list(index_matrix)

# データセットをクエリとドキュメントに分割する
query_data, document_data = split_into_query_and_document(jp_data)

# OpenSearchを扱うためのインスタンスを作成する
tester = OpenSearchTester(index_name="ch09")

# OpenSearchインデックスを作成する
tester.create_index(get_default_index_options(dimension_number))

# ドキュメントを整形し入力する
tester.input_documents(
    document_data,
    formatter=lambda row: {
        # とくにドキュメントベクトル（製品タイトルベクトル）のセントロイドIDのリストも入力する
        "title_centroids": row.title_centroids,
        # 以降は、本書デフォルトと同じ
        "product_title": row.product_title,
        "title_vector": row.title_vector,
    },
)

# クエリを整形し入力（つまり検索）する
tester.input_queries(
    query_data,
    size=10,
    formatter=lambda row: {
        "script_score": {
            # とくに、製品タイトルとクエリのセントロイドIDが一致するドキュメントのみスコア計算する
            "query": {
                "bool": {
                    "should": [
                        {"match": {"title_centroids": c}} for c in row.query_centroids
                    ]
                }
            },
            # 以降は、本書デフォルトと同じ
            "script": {
                "source": "knn_score",
                "lang": "knn",
                "params": {
                    "field": "title_vector",
                    "query_value": row.query_vector,
                    "space_type": "innerproduct",
                },
            },
        }
    },
)
