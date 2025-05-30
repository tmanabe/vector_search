#!/usr/bin/env python

from argparse import ArgumentParser
from ch02_basic_vectorization import get_dimension_number_of
from ch04_evaluate_search_results_1_faiss import test_faiss
from ch07_vector_compression_0_data import read_tuned_vectorized_data

import faiss


# コマンドライン引数から、LSHのビット数を読み込む
argument_parser = ArgumentParser()
argument_parser.add_argument("--dimensions-of-output", default=96, type=int)
args = argument_parser.parse_args()

# ベクトル化したデータセットをメモリに読み込む
jp_data = read_tuned_vectorized_data()

# データからベクトルの次元数を取得し、Faissインデックスを作成する
faiss_index = faiss.IndexLSH(
    get_dimension_number_of(jp_data),
    args.dimensions_of_output,
)

# テストする。とくにIndexLSHは距離を返す (returns_distance=True) ことに注意する。
test_faiss(faiss_index, jp_data, k=10, returns_distance=True)
