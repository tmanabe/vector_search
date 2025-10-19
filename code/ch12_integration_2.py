#!/usr/bin/env python

from ch02_basic_vectorization import vectorize_with
from ch03_vector_search_engines_0_numpy import cos
from ch04_evaluate_search_results_0_ndcg import print_ndcg


# 整形の対象とする列名と対応するプレフィックスの定義
COLUMNS_TO_PREFIX = {
    "product_title": "タイトル",
    "product_brand": "ブランド名",
    "product_color": "色",
    "product_description": "説明文",
    "product_bullet_point": "説明文（箇条書き）",
}


# データセットの行をドキュメントに整形する関数
def format_document(row):
    results = []
    for column, prefix in COLUMNS_TO_PREFIX.items():
        if row[column] is not None:
            results.append(f"{prefix}：{row[column]}")
    return "、".join(results)


# データセットの行をクエリに整形する関数
def format_query(row):
    return f"クエリ：{row.query}"


# ベクトル化、スコアの計算、平均nDCGの計算と表示を一気に行う関数
def evaluate(model, data):
    # クエリとドキュメントをベクトル化する
    vectorize = vectorize_with(model=model)
    data["query_vector"] = vectorize(data.apply(format_query, axis=1))
    data["document_vector"] = vectorize(data.apply(format_document, axis=1))

    # スコア（ここではコサイン類似度）を計算する
    data["score"] = data[["query_vector", "document_vector"]].apply(cos, axis=1)

    # nDCGを計算し表示する
    print_ndcg(data)


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from argparse import ArgumentParser
    from ch01_data_preparation import read_jp_data
    from ch04_evaluate_search_results_0_ndcg import esci_label_to_ndcg_gain
    from ch05_advanced_vectorization_0_tune import (
        DEFAULT_ARGS,
        get_sentence_transformer,
    )
    from datasets import Dataset
    from sentence_transformers import losses, SentenceTransformerTrainer

    # コマンドライン引数を読み込む
    argument_parser = ArgumentParser()
    # 最大シーケンス長（ベクトル化モデルに入力できる最大のトークン数）
    argument_parser.add_argument("--max-seq-length", default=128, type=int)
    # データのサンプル率
    argument_parser.add_argument(
        "--sample-rate", default=0.01, type=float, choices=[0.01, 1.0]
    )
    args = argument_parser.parse_args()

    # ファインチューニング前のベクトル化モデルを組み、最大シーケンス長を設定する
    model = get_sentence_transformer()
    model.max_seq_length = args.max_seq_length

    # テストデータを読み込む。とくに、製品の詳細も読み込む
    jp_test_data = read_jp_data(
        split="test", sample_rate=args.sample_rate, read_product_detail=True
    )

    # ファインチューニング前のモデルの平均nDCGを計算し表示する
    print("Before fine-tuning")
    evaluate(model, jp_test_data)

    # 訓練データを読み込む。とくに、製品の詳細も読み込む
    jp_train_data = read_jp_data(
        split="train", sample_rate=args.sample_rate, read_product_detail=True
    )

    # ファインチューニングを実行する
    SentenceTransformerTrainer(
        # クエリ、ドキュメント、正解のスコア（nDCGの利得）を与える
        train_dataset=Dataset.from_dict(
            {
                "query": jp_train_data.apply(format_query, axis=1),
                "document": jp_train_data.apply(format_document, axis=1),
                "score": jp_train_data["esci_label"].apply(esci_label_to_ndcg_gain),
            }
        ),
        # もちろん対象のモデルも与える
        model=model,
        # パラメータも与える
        args=DEFAULT_ARGS,
        # 損失関数も与える。ここではコサイン類似度と正解のスコアの大小関係が一致するように指定
        loss=losses.CoSENTLoss(model),
    ).train()

    # ファインチューニング後のモデルの平均nDCGを計算し表示する
    print("After fine-tuning")
    evaluate(model, jp_test_data)
