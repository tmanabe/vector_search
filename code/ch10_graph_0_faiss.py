#!/usr/bin/env python

from argparse import ArgumentParser
from ch02_basic_vectorization import get_dimension_number_of
from ch04_evaluate_search_results_1_faiss import test_faiss
from ch07_vector_compression_0_data import read_tuned_vectorized_data

import faiss


# コマンドライン引数から節点あたりの枝の最大数を読み込む
argument_parser = ArgumentParser()
argument_parser.add_argument("--number-of-neighbors", default=8, type=int)
args = argument_parser.parse_args()

# ベクトル化したデータセットをメモリに読み込む
jp_data = read_tuned_vectorized_data()

# データからベクトルの次元数を取得し、Faissインデックスを作成する
faiss_index = faiss.IndexHNSWFlat(
    get_dimension_number_of(jp_data),
    args.number_of_neighbors,
    faiss.METRIC_INNER_PRODUCT,
)

# テストする
test_faiss(faiss_index, jp_data, k=10)
