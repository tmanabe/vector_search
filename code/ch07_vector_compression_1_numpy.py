#!/usr/bin/env python

from sentence_transformers.quantization import quantize_embeddings

import numpy as np


# スカラ量子化を実行する関数
def quantize(data):
    # キャリブレーション対象は訓練データ上のタイトルベクトル列とする
    calibration_vectors = np.vstack(data[data.split == "train"]["title_vector"])

    # クエリベクトル列・タイトルベクトル列それぞれ実行する
    for vector_column in ["query_vector", "title_vector"]:
        data[vector_column] = quantize_embeddings(
            np.vstack(data[vector_column]),
            # 表現はINT8とする
            precision="int8",
            # キャリブレーション対象は共通で訓練データ上のタイトルベクトル列とする
            calibration_embeddings=calibration_vectors,
        ).tolist()


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from ch03_vector_search_engines_0_numpy import cos
    from ch04_evaluate_search_results_0_ndcg import print_ndcg
    from ch07_vector_compression_0_data import read_tuned_vectorized_data

    # スカラ量子化なし・あり、それぞれ実行して平均nDCGを比較する
    for with_quantize in [False, True]:
        print(f"with_quantize: {with_quantize}")

        # ベクトル化したデータセットをメモリに読み込む
        jp_data = read_tuned_vectorized_data()

        # 必要に応じてスカラ量子化を実行する
        if with_quantize:
            quantize(jp_data)

        # スコア（ここではコサイン類似度）を計算する
        jp_data["score"] = jp_data[["query_vector", "title_vector"]].apply(
            cos, axis=1
        )

        # テストデータ上の平均nDCGを計算し表示する
        print_ndcg(jp_data[jp_data.split == "test"])
