#!/usr/bin/env python

from ch02_basic_vectorization import vectorize_with
from ch03_vector_search_engines_0_numpy import cos
from ch04_evaluate_search_results_0_ndcg import print_ndcg


# ベクトル化、スコアの計算、平均nDCGの計算と表示を一気に行う関数
def evaluate(model, data):
    # クエリとドキュメント（ここでは製品タイトル）をベクトル化する
    vectorize = vectorize_with(
        model=model,
        # とくに、大きめのバッチサイズを指定する
        args={"batch_size": 64, "show_progress_bar": True},
    )
    data["query_vector"] = vectorize(data["query"])
    data["title_vector"] = vectorize(data["product_title"])

    # スコア（ここではコサイン類似度）を計算する
    data["score"] = data[["query_vector", "title_vector"]].apply(cos, axis=1)

    # 平均nDCGを計算し表示する
    print_ndcg(data)


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from argparse import ArgumentParser
    from ch01_data_preparation import read_jp_data
    from ch05_advanced_vectorization_0_tune import get_tuned_vectorization_model

    # コマンドライン引数から、テストデータのサンプル率を読み込む
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--sample-rate", default=0.01, type=float, choices=[0.01, 1.0]
    )
    args = argument_parser.parse_args()

    # テストデータを読み込む
    jp_test_data = read_jp_data(split="test", sample_rate=args.sample_rate)

    # ファインチューニング後のモデルをBF16に量子化し、平均nDCGを表示する
    print("After fine-tuning (BF16)")
    evaluate(get_tuned_vectorization_model().bfloat16(), jp_test_data)

    # ファインチューニング後のモデルをFP16に量子化し、平均nDCGを表示する
    print("After fine-tuning (FP16)")
    evaluate(get_tuned_vectorization_model().half(), jp_test_data)
