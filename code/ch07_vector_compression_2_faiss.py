#!/usr/bin/env python

from ch02_basic_vectorization import get_dimension_number_of
from ch04_evaluate_search_results_1_faiss import test_faiss
from ch07_vector_compression_0_data import read_tuned_vectorized_data

import faiss
import numpy as np


# ベクトル化したデータセットをメモリに読み込む
jp_data = read_tuned_vectorized_data()

# データからベクトルの次元数を取得する
dimension_number = get_dimension_number_of(jp_data)

# スカラ量子化なし・あり、それぞれ実行して平均nDCGを比較する
for with_quantize in [False, True]:
    # Faissインデックスを作成する
    if with_quantize:
        # スカラ量子化あり
        faiss_index = faiss.IndexScalarQuantizer(
            dimension_number,
            # 表現はINT8とする
            faiss.ScalarQuantizer.QT_8bit,
            # 計算式はINNER_PRODUCT（内積もしくはドット積）とする
            faiss.METRIC_INNER_PRODUCT,
        )

        # キャリブレーションデータは訓練データ上の
        # ドキュメント（ここでは製品タイトル）ベクトル列とする
        calibration_vectors = np.vstack(
            jp_data[jp_data.split == "train"]["title_vector"]
        )

        # キャリブレーションする
        faiss_index.train(calibration_vectors)
    else:
        # スカラ量子化なし
        faiss_index = faiss.IndexFlatIP(dimension_number)

    # テストする
    test_faiss(faiss_index, jp_data, k=10)
