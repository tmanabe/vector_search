#!/usr/bin/env python

from ch02_basic_vectorization import (
    get_dimension_number_of,
    split_into_query_and_document,
)
from ch03_vector_search_engines_2_os import OpenSearchTester
from ch07_vector_compression_0_data import read_tuned_vectorized_data


# ベクトル化したデータセットをメモリに読み込む
jp_data = read_tuned_vectorized_data()

# データセットをクエリとドキュメントに分割する
query_data, document_data = split_into_query_and_document(jp_data)

# OpenSearchを扱うためのインスタンスを作成する
tester = OpenSearchTester(index_name="ch10")

# データからベクトルの次元数を取得し、OpenSearchインデックスを作成する
tester.create_index(
    {
        # 高速なベクトル検索のためのインデックスを、OpenSearch内部で構築する設定
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "title_vector": {
                    "type": "knn_vector",
                    "dimension": get_dimension_number_of(jp_data),
                    # HNSWでは、ドキュメントの入力の時点でも類似度や距離を計算する
                    "space_type": "innerproduct",  # ここでは内積（ドット積）
                    # LuceneのHNSWの実装を使う設定
                    "method": {"engine": "lucene", "name": "hnsw"},
                }
            }
        },
    }
)

# ドキュメントを整形し入力する
tester.input_documents(document_data)

# クエリを整形し入力（つまり検索）する
tester.input_queries(
    query_data,
    size=10,
    formatter=lambda row: {
        # HNSWにより対象を絞り込みつつスコア計算する指定
        "knn": {
            # ドキュメントベクトルのフィールド名
            "title_vector": {
                # クエリベクトル
                "vector": row.query_vector,
                # HNSWで引き当てさせるベクトルの数なのでsizeと同じにする
                "k": 10,
            }
        }
    },
)
