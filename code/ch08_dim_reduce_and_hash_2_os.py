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
from ch08_dim_reduce_and_hash_0_numpy import randomly_rotate


# コマンドライン引数から、LSHのビット数を読み込む
argument_parser = ArgumentParser()
argument_parser.add_argument("--dimensions-of-output", default=8, type=int)
args = argument_parser.parse_args()

# ベクトル化したデータセットをメモリに読み込む
jp_data = read_tuned_vectorized_data()

# データからベクトルの次元数を取得する
dimensions = get_dimension_number_of(jp_data)


# ハッシュ関数（ベクトルの要素の符号を取り、冗長だが見やすい文字列とする）
def hashed(vector):
    return "".join(["0" if element < 0 else "1" for element in vector])


# LSHを実行する
jp_data["query_hash"] = [
    hashed(vector)
    for vector in randomly_rotate(
        jp_data["query_vector"], dimensions, args.dimensions_of_output
    )
]
jp_data["title_hash"] = [
    hashed(vector)
    for vector in randomly_rotate(
        jp_data["title_vector"], dimensions, args.dimensions_of_output
    )
]

# データセットをクエリとドキュメントに分割する
query_data, document_data = split_into_query_and_document(jp_data)

# OpenSearchを扱うためのインスタンスを作成する
tester = OpenSearchTester(index_name="ch08")

# OpenSearchインデックスを作成する
tester.create_index(get_default_index_options(dimensions))

# ドキュメントを整形し入力する
tester.input_documents(
    document_data,
    formatter=lambda row: {
        # とくに、製品タイトル（のベクトル）のハッシュ値も入力する
        "title_hash": row.title_hash,
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
            # とくに、製品タイトルとクエリのハッシュ値が一致するドキュメントのみスコア計算する
            "query": {"bool": {"must": {"match": {"title_hash": row.query_hash}}}},
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
