# モデル学習・評価ツール (autoencoder_poc)

作成したデータセットを使用してオートエンコーダーの学習を行い、異常検知の精度検証（PoC）を行うツールです。
実験管理には **MLflow** を使用しており、学習経過（Loss）、評価スコア、再構成画像の結果などをWebブラウザ上で確認・管理できます。

## 前提条件

1.  `create_traindata` ツールなどで、データセット（`train/good`, `test/good`, `test/anomaly`）が用意されていること。
2.  Python環境に `mlflow` がインストールされていること。

## 実行手順

### 1. MLflowサーバーの起動

学習ログを記録するために、まずMLflowサーバーをコンテナで起動してください。

> **注意**: すでに起動している場合はこの手順は不要です。

### 2. 学習の実行

新しいターミナルを開き（プロジェクトルートで）、以下のコマンドを実行して学習を開始します。

```bash
python autoencoder_poc/main.py
```

プログラムは自動的に以下の処理を行います：
1.  データセットの読み込み
2.  オートエンコーダーの学習（Train）
3.  検証（Validation）とEarly Stopping判定
4.  テストデータを用いた異常検知スコアの算出
5.  結果画像の生成（箱ひげ図、ヒストグラム、再構成画像の比較）

### 3. 結果の確認

ブラウザで [http://localhost:8080](http://localhost:8080) にアクセスします。
実験名 `Autoencoder_Anomaly_Detection` の中に、今回の実行結果（Run）が記録されています。

**確認できる項目:**
*   **Metrics**: `train_loss`, `val_loss`, 各クラスの異常スコア（平均・分散など）
*   **Artifacts**:
    *   `reconstruction_comparison.png`: 元画像と再構成画像の比較
    *   `boxplot.png`: クラスごとの異常スコアの分布（箱ひげ図）
    *   `histogram.png`: 異常スコアのヒストグラム

## 次のステップへ（重要：モデルの登録）

学習したモデルを次のステップ「推論 (autoencoder_inference)」で使用するためには、MLflow上でモデルを登録する必要があります。

1.  MLflow UIで、結果が良かった Run の詳細画面を開きます。
2.  **Artifacts** セクションにある `autoencoder_model` (または `best_model`) フォルダをクリックします。
3.  右側に表示される **"Register Model"** ボタンをクリックします。
4.  **Model Name** に、推論ツールの設定 (`autoencoder_inference/config/config.yaml`) と一致する名前（例: `kinoko`）を入力して **Register** します。

これで推論ツールからモデルを読み込めるようになります。

## 設定の変更

`config/config.yaml` ファイルで学習パラメータを変更できます。

*   **dataset**: データセットのパスや画像サイズ
*   **training**:
    *   `epochs`: 学習回数
    *   `learning_rate`: 学習率
    *   `early_stopping_patience`: 検証ロスが改善しなくなった時に停止するまでの回数
*   **model**:
    *   `enc_channels`: エンコーダーの層構造
    *   `latent_dim`: 潜在変数の次元数
