#!/usr/bin/env python

import faiss
import numpy as np


# ランダム回転を実行する関数
def randomly_rotate(vectors, dimensions_of_input, dimensions_of_output, seed=0):
    # FaissのRandomRotationMatrixの作成
    random_rotation_matrix = faiss.RandomRotationMatrix(
        dimensions_of_input, dimensions_of_output
    )

    # 線形変換の重みを乱数によって決める
    random_rotation_matrix.init(seed)

    # 線形変換を実行し結果を返す
    return list(
        random_rotation_matrix.apply_py(
            np.vstack(vectors),
        )
    )


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from ch02_basic_vectorization import get_dimension_number_of
    from ch03_vector_search_engines_0_numpy import cos
    from ch04_evaluate_search_results_0_ndcg import print_ndcg
    from ch07_vector_compression_0_data import read_tuned_vectorized_data

    # ベクトル化したデータセットをメモリに読み込む
    jp_data = read_tuned_vectorized_data()

    # データからベクトルの次元数を取得する
    dimensions = get_dimension_number_of(jp_data)

    # ランダム回転を実行する。クエリとドキュメントの両ベクトル列それぞれ行う。
    for vector_column in ["query_vector", "title_vector"]:
        jp_data[vector_column] = randomly_rotate(
            jp_data[vector_column], dimensions, 64
        )

    # スコア（ここではコサイン類似度）を計算する
    jp_data["score"] = jp_data[["query_vector", "title_vector"]].apply(cos, axis=1)

    # テストデータ上の平均nDCGを計算し表示する
    print_ndcg(jp_data[jp_data.split == "test"])
