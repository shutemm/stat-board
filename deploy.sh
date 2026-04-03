#!/bin/bash
# keiba-ai 予想ページ デプロイスクリプト
# 使い方: bash public_prediction/deploy.sh
#
# 1コマンドで GitHub認証 → リポジトリ作成 → push まで完了する

set -e

GH="/c/Users/okusa/tools/gh/bin/gh.exe"
if [ ! -f "$GH" ]; then
    GH="gh"
fi

REPO_NAME="keiba-ai-public"
GH_USER="shutemm"
REPO_DIR="/c/Users/okusa/Desktop/keiba-ai-public"

echo "=== keiba-ai 予想ページ デプロイ ==="
echo ""

# Step 1: gh 認証確認
echo "[1/4] GitHub認証確認..."
if ! $GH auth status &>/dev/null; then
    echo "GitHub認証が必要です。ブラウザが開きます。"
    $GH auth login --hostname github.com --git-protocol https --web
fi
echo "  -> 認証OK"
echo ""

# Step 2: リポジトリ作成
echo "[2/4] パブリックリポジトリ作成..."
if $GH repo view "$GH_USER/$REPO_NAME" &>/dev/null; then
    echo "  -> リポジトリ $GH_USER/$REPO_NAME は既に存在します"
else
    $GH repo create "$REPO_NAME" --public --description "keiba-ai 競馬予測公開ページ"
    echo "  -> リポジトリ作成完了"
fi
echo ""

# Step 3: Push
echo "[3/4] コード push..."
cd "$REPO_DIR"
if ! git remote | grep -q origin; then
    git remote add origin "https://github.com/$GH_USER/$REPO_NAME.git"
fi

# gh auth で git credentials を設定
$GH auth setup-git 2>/dev/null || true

git push -u origin main
echo "  -> push完了"
echo ""

# Step 4: デプロイ情報
echo "[4/4] 完了"
echo ""
echo "=========================================="
echo "GitHub: https://github.com/$GH_USER/$REPO_NAME"
echo "=========================================="
echo ""
echo "次のステップ: Streamlit Community Cloud でデプロイ"
echo "  1. https://share.streamlit.io/ にアクセス（GitHubアカウントでログイン）"
echo "  2. 「New app」をクリック"
echo "  3. 設定:"
echo "     - Repository: $GH_USER/$REPO_NAME"
echo "     - Branch: main"
echo "     - Main file path: app.py"
echo "  4. 「Deploy!」をクリック"
echo ""
echo "数分後に https://<app-name>.streamlit.app でアクセス可能になります"
