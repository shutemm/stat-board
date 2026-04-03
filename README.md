# keiba-ai 予想公開ページ

競馬AI予測結果の公開表示アプリ。

## デプロイ済み

- Repository: [shutemm/keiba-ai-public](https://github.com/shutemm/keiba-ai-public)
- Streamlit Cloud: デプロイ設定
  - Branch: `main`
  - Main file path: `app.py`

## データ更新

```bash
# keiba-ai-org ディレクトリで実行
bash public_prediction/update_data.sh [YYYY-MM-DD]
```

Streamlit Cloud が自動で再デプロイする。
