#!/usr/bin/env python

from argparse import ArgumentParser
from ch02_basic_vectorization import get_dimension_number_of
from ch04_evaluate_search_results_1_faiss import test_faiss
from ch07_vector_compression_0_data import read_tuned_vectorized_data

import faiss
import numpy as np


# コマンドライン引数を読み込む
argument_parser = ArgumentParser()
# ドキュメントを分別するセントロイド数（クラスタ数でもある）
argument_parser.add_argument("--number-of-centroids", default=4, type=int)
# クエリあたりFaissに引き当てさせるセントロイド数（プローブ数）
argument_parser.add_argument("--number-of-probes", default=1, type=int)
args = argument_parser.parse_args()

# ベクトル化したデータセットをメモリに読み込む
jp_data = read_tuned_vectorized_data()

# データからベクトルの次元数を取得する
dimension_number = get_dimension_number_of(jp_data)

# IVFでは、Faissインデックスを二重に作成する。外側はドキュメントのインデックス
faiss_index = faiss.IndexIVFFlat(
    # 内側はIVF特有のセントロイドのインデックス
    faiss.IndexFlatIP(dimension_number),
    dimension_number,
    args.number_of_centroids,
    faiss.METRIC_INNER_PRODUCT,
)

# クラスタリング対象は訓練データ上のドキュメント（ここでは製品タイトル）ベクトル列とする
vectors_to_cluster = np.vstack(jp_data[jp_data.split == "train"]["title_vector"])

# ベクトル列をクラスタリングする
faiss_index.train(vectors_to_cluster)

# プローブ数を設定する
faiss_index.nprobe = args.number_of_probes

# テストする
test_faiss(faiss_index, jp_data, k=10)
