#!/bin/bash
# keiba-ai 予測データ更新スクリプト
# 使い方: bash public_prediction/update_data.sh [YYYY-MM-DD]
#
# export_predictions.py でJSON生成 → 公開リポジトリに push

set -e

ORG_DIR="/c/Users/okusa/Desktop/keiba-ai-org"
PUBLIC_DIR="/c/Users/okusa/Desktop/keiba-ai-public"
DATE="${1:-}"

echo "=== 予測データ更新 ==="

# Step 1: JSON生成
echo "[1/3] 予測結果エクスポート..."
if [ -n "$DATE" ]; then
    python "$ORG_DIR/public_prediction/export_predictions.py" --date "$DATE"
else
    python "$ORG_DIR/public_prediction/export_predictions.py"
fi
echo ""

# Step 2: 公開リポジトリにコピー
echo "[2/3] 公開リポジトリにコピー..."
cp "$ORG_DIR/public_prediction/data/"*.json "$PUBLIC_DIR/data/"
echo "  -> コピー完了"
echo ""

# Step 3: commit & push
echo "[3/3] commit & push..."
cd "$PUBLIC_DIR"
git add data/
if git diff --cached --quiet; then
    echo "  -> 変更なし（スキップ）"
else
    git commit -m "update predictions $(date +%Y-%m-%d)"
    git push origin main
    echo "  -> push完了（Streamlit Cloud が自動再デプロイ）"
fi
echo ""
echo "=== 完了 ==="
