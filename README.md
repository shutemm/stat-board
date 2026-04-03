# keiba-ai 予想公開ページ

競馬AI予測結果の公開表示アプリ。

## デプロイ手順

1. GitHub に新規リポジトリを作成（例: `keiba-ai-public`）
2. `public_prediction/` の内容を push
3. [Streamlit Community Cloud](https://share.streamlit.io/) でデプロイ
   - Repository: 上記リポジトリ
   - Branch: `main`
   - Main file path: `app.py`

## データ更新

```bash
# keiba-ai-org ディレクトリで実行
python public_prediction/export_predictions.py

# 公開リポジトリに push
cd public_prediction
git add data/
git commit -m "update predictions"
git push
```

Streamlit Cloud が自動で再デプロイする。
