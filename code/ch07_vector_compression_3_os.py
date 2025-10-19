#!/usr/bin/env python

from argparse import ArgumentParser
from ch02_basic_vectorization import (
    get_dimension_number_of,
    split_into_query_and_document,
)
from ch03_vector_search_engines_2_os import OpenSearchTester
from ch07_vector_compression_0_data import read_tuned_vectorized_data
from ch07_vector_compression_1_numpy import quantize


# コマンドライン引数で、スカラ量子化なしを指定できる
argument_parser = ArgumentParser()
argument_parser.add_argument("--skip-quantize", action="store_true")
args = argument_parser.parse_args()

# ベクトル化したデータセットをメモリに読み込む
jp_data = read_tuned_vectorized_data()

# 必要に応じてベクトルをスカラ量子化する
if not args.skip_quantize:
    quantize(jp_data)

# データセットをクエリとドキュメントに分割する
query_data, document_data = split_into_query_and_document(jp_data)

# OpenSearchを扱うためのインスタンスを作成する
tester = OpenSearchTester(index_name="ch07")

# OpenSearchインデックスを作成する
tester.create_index(
    {
        "mappings": {
            "properties": {
                "title_vector": {
                    "type": "knn_vector",
                    # データからベクトルの次元数を取得する
                    "dimension": get_dimension_number_of(jp_data),
                    # スカラ量子化ありなら、表現は byte（INT8）とする
                    "data_type": "float" if args.skip_quantize else "byte",
                }
            }
        }
    }
)

# ドキュメントを整形し入力する
tester.input_documents(document_data)

# クエリを整形し入力（つまり検索）する
tester.input_queries(query_data, size=10)
