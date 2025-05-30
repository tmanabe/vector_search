#!/usr/bin/env python

from ch05_advanced_vectorization_0_tune import DEFAULT_ARGS
from chaa_image_0_prec import (
    evaluate,
    get_text_or_image_vectorization_model,
    IMAGE_LABEL_TO_QUERY,
    read_image_data,
)
from datasets import Dataset
from sentence_transformers import losses, SentenceTransformerTrainer


# 題材の画像訓練データを1パーセントサンプリングと画像をデコードしつつ読み込む
image_train_data = read_image_data(split="train", sample_percent=1)

# ファインチューニング前の、画像にも対応するベクトル化モデルを読み込む
model = get_text_or_image_vectorization_model()

# ファインチューニングを実行する
SentenceTransformerTrainer(
    # クエリと正解の画像を与える
    train_dataset=Dataset.from_dict(
        {
            "query": image_train_data["label"].apply(
                lambda label: IMAGE_LABEL_TO_QUERY[label]
            ),
            "image": image_train_data["image"],
        }
    ),
    # もちろん対象のモデルも与える
    model=model,
    # パラメータも与える
    args=DEFAULT_ARGS,
    # 損失関数も与える。ここではクエリと正解の画像とのコサイン類似度は高くし、
    # 同じバッチ中で不正解の画像とのコサイン類似度は相対的に低くするように指定する。
    loss=losses.MultipleNegativesRankingLoss(model),
).train()

# 題材の画像テストデータを1パーセントサンプリングと画像のデコードをしつつ読み込む
image_test_data = read_image_data(split="test", sample_percent=1)

# ファインチューニング後のモデルの、適合率のクエリ間平均を表示する
evaluate(model, image_test_data)
