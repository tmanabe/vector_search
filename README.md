# 「ベクトル検索実践入門」サポートページ

## リポジトリの構成

- `code`: サンプルコード本体
    - `esci-data`: 本文で題材とするデータセットへの参照（Gitサブモジュール）
    - `output`: 章別の出力例
    - `script`: 章別の細かいサンプルコード（コマンドラインなど）
    - `tmp`: 一時的なデータやモデルの保存先
    - `Dockerfile`: サンプルコードの実行環境の設定
    - `*.py`: 主要なサンプルコード
    - `docker-compose.yaml`: 実行環境やOpenSearchを立ち上げるための設定
    - `docker-compose.yaml.gpu.patch`: その設定にあて、実行環境からGPUを利用するためのパッチ
    - `requirements.txt`: サンプルコードで利用するライブラリの一覧
- `.gitmodules`: Gitサブモジュールの設定
- `README.md`: このファイル

## リリースノート

### 1.0.0

- 出版時の状態
