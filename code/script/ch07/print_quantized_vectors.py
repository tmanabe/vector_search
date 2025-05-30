#!/usr/bin/env python

from sentence_transformers import SentenceTransformer

import numpy as np


# ベクトルは表示すると非常に長いので、前後2要素ずつ表示する
np.set_printoptions(edgeitems=2, threshold=5)


# それぞれの表現について表示する
for floating_point_format in ("FP32", "BF16", "FP16"):
    print(floating_point_format)

    # ここではベクトルの質は関係ないので、軽量な基本的なベクトル化モデルを読み込む
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 必要に応じてモデルを量子化する
    if floating_point_format == "BF16":
        model.bfloat16()
    if floating_point_format == "FP16":
        model.half()

    # 適当なテキストをベクトル化し、表示する
    vector = model.encode("Vector search")
    print(vector, vector.dtype, vector.nbytes)
    vector_bytes = vector.tobytes()
    print(vector_bytes[:16].hex(), vector_bytes[-16:].hex())
