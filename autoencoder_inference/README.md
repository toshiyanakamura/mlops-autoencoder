# リアルタイム推論ツール (autoencoder_inference)

学習済みのオートエンコーダーモデルを読み込み、カメラ映像に対してリアルタイムで異常検知を行うツールです。
MLflowのModel Registryからモデルを取得し、設定した閾値に基づいて「正常(NORMAL)」か「異常(ANOMALY)」かを判定します。


## 運用環境(Jetson)での動作イメージ

![Image](https://github.com/user-attachments/assets/8cdc6607-ecd7-4aca-9f3d-7d0c2646d103)

## 前提条件

1.  MLflowサーバーが起動していること (`mlflow ui ...`)。
2.  `autoencoder_poc` での学習が完了し、MLflow上でモデルが登録（Register）されていること。
    *   デフォルトのモデル名は `kinoko` です。

## 実行方法

プロジェクトのルートディレクトリで以下のコマンドを実行してください。

```bash
python autoencoder_inference/main.py
```

## 操作ガイド

ツールが起動すると、カメラ映像と推論結果が表示されます。

### 1. 異常検知の確認
画面上には以下の情報が表示されます。
*   **Status**: 判定結果。`NORMAL` (緑) または `ANOMALY` (赤)。
*   **Score**: 現在の画像の異常スコア（再構成誤差）。この値が大きいほど異常度が高いことを意味します。
*   **Threshold**: 現在設定されている判定閾値。Scoreがこの値を超えると「異常」と判定されます。

### 2. 閾値（Threshold）の調整
画面上部（または下部）にある **"Threshold" トラックバー（スライダー）** をマウスで動かすことで、リアルタイムに閾値を変更できます。
*   正常な対象を映している時に `NORMAL` になり、異常な対象を映した時に `ANOMALY` になるように調整してください。

### 3. 終了
| キー | 動作 |
| :--- | :--- |
| **`q`** | ツールを終了します。 |

## トラブルシューティング

### モデルが読み込めない場合
エラーメッセージ: `Error loading model: ...`
*   MLflowサーバーが起動しているか確認してください。
*   MLflow UI上でモデルが正しく登録されているか確認してください。
*   `conf/config.yaml` の `model_name` が、MLflow上の登録名（例: `kinoko`）と一致しているか確認してください。

### カメラが開かない場合
エラーメッセージ: `Cannot open camera ...`
*   `conf/config.yaml` の `camera_id` を変更してみてください（0, 1, 2...）。

## 設定の変更

`config/config.yaml` ファイルで推論パラメータを変更できます。

*   **mlflow**:
    *   `model_name`: 読み込むモデルの名前（MLflowの登録名）
    *   `model_version`: モデルのバージョン（`latest` または `1`, `2` などの番号）
*   **inference**:
    *   `camera_id`: 使用するカメラID
    *   `initial_threshold`: 起動時の初期閾値
    *   `crop_height`, `crop_width`: 画像の中央から切り抜くサイズ（学習時と同じサイズにするのが基本ですが、推論時に範囲を絞ることも可能です）
