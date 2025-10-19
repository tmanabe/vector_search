#!/usr/bin/env python

from ch04_evaluate_search_results_0_ndcg import (
    esci_label_to_ndcg_gain,
    print_ndcg,
)
from collections import Counter
from fugashi import Tagger

import xgboost as xgb


# ある列をスコアとした場合のテストデータ上の平均nDCGを計算し、表示する関数
def print_ndcg_by(column, data):
    data["score"] = data[column]
    print(column)
    print_ndcg(data[data.split == "test"])


# 形態素解析器MeCabを使うための、そのPythonラッパーfugashiのインスタンス
FUGASHI_TAGGER = Tagger("-Owakati")


# クエリをトークナイズする関数。キーワードから出現回数への連想配列にする
def tokenize_query(query):
    tokens = FUGASHI_TAGGER.parse(query).split()
    return Counter(tokens)


# ドキュメントをトークナイズする関数。トークンのリストにする
def tokenize_document(text):
    return FUGASHI_TAGGER.parse(text).split()


# キーワード検索の特徴量として、クエリ中のトークンが製品タイトル中に出現する回数（TF,
# Term Frequency）と、クエリ・製品タイトルそれぞれの長さ（トークン数）を抽出する関数
def calc_tf_and_len(row):
    query_tokens, title_tokens = row.query_tokens, row.title_tokens
    row["title_tf"] = sum([query_tokens[token] for token in title_tokens])
    row["query_len"] = sum(query_tokens.values())
    row["title_len"] = len(title_tokens)
    return row


# XGBoostのGBDTを準備し、訓練して返す関数
def train_xgb_ranker(data, features):
    # GBDTを準備する
    xgb_ranker = xgb.XGBRanker(
        n_estimators=8,
        max_depth=2,
        objective="rank:ndcg",
        ndcg_exp_gain=False,
        random_state=0,
    )

    # 訓練する
    train_data = data[data.split == "train"]
    xgb_ranker.fit(
        # 特徴量の列（一般には複数）を与える
        train_data[features],
        # nDCGの利得の列も与える
        train_data["esci_label"].apply(esci_label_to_ndcg_gain),
        # どの行がどのクエリ（ID）に紐づくドキュメントかも与える
        qid=train_data["query_id"],
    )

    return xgb_ranker


# クエリIDごとに、ある列をスコアとした場合の順位を計算する関数
def calc_rank_by(column, data):
    return data.groupby("query_id")[column].rank(ascending=False, method="first")


# RRFで順位に加算する定数
RRF_K = 60


# 複数の列から成る行を受け取り、それぞれ順位とみなしてRRFを実行する関数
def rrf(row):
    return sum([1.0 / (RRF_K + rank) for rank in row])


# このコードを直に実行した場合のみ、以下のコードを実行する
if __name__ == "__main__":
    from ch03_vector_search_engines_0_numpy import cos
    from ch07_vector_compression_0_data import read_tuned_vectorized_data

    # ベクトル化したデータをメモリに読み込む
    jp_data = read_tuned_vectorized_data()

    # コサイン類似度を計算する
    jp_data["cos"] = jp_data[["query_vector", "title_vector"]].apply(cos, axis=1)
    # 平均nDCGを計算し表示する
    print_ndcg_by("cos", jp_data)

    # クエリをトークナイズする
    jp_data["query_tokens"] = jp_data["query"].apply(tokenize_query)
    # 製品タイトルをトークナイズする
    jp_data["title_tokens"] = jp_data["product_title"].apply(tokenize_document)
    # キーワード検索の特徴量を抽出する
    jp_data = jp_data.apply(calc_tf_and_len, axis=1)

    # GBDTを訓練する（コサイン類似度を使わない、3特徴）
    xgb_features = ["title_tf", "query_len", "title_len"]
    xgb_ranker = train_xgb_ranker(jp_data, xgb_features)
    # GBDTを推論する
    jp_data["xgb3"] = xgb_ranker.predict(jp_data[xgb_features])
    # 平均nDCGを計算し表示する
    print_ndcg_by("xgb3", jp_data)

    # コサイン類似度によるランキング結果の順位を計算する
    jp_data["cos_rank"] = calc_rank_by("cos", jp_data)
    # GBDTの出力によるランキング結果の順位を計算する
    jp_data["xgb3_rank"] = calc_rank_by("xgb3", jp_data)
    # Reciplocal Rank Fusion（RRF）を実行する
    jp_data["rrf"] = jp_data[["cos_rank", "xgb3_rank"]].apply(rrf, axis=1)
    # 平均nDCGを計算し表示する
    print_ndcg_by("rrf", jp_data)

    # GBDTを訓練する（コサイン類似度を使う、4特徴）
    xgb_features += ["cos"]
    xgb_ranker = train_xgb_ranker(jp_data, xgb_features)
    # GBDTを推論する
    jp_data["xgb4"] = xgb_ranker.predict(jp_data[xgb_features])
    # 平均nDCGを計算し表示する
    print_ndcg_by("xgb4", jp_data)
