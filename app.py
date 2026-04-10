"""keiba-ai 予想公開ページ

予測結果JSONを読み込んで表示するStreamlitアプリ。
モデルコード・特徴量計算・学習ロジックは一切含まない。

データ更新: export_predictions.py で生成した JSON を data/ に配置する。
レートデータ: export_ratings.py で生成した ratings.json を data/ に配置する。
"""

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 設定
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(
    page_title="keiba-ai 予想",
    page_icon="🏇",
    layout="wide",
)

# ============================================================
# カラーパレット（ダークモード専用）
# ============================================================

BG_GREEN_STRONG = "#1a5c2a"
BG_GREEN_MEDIUM = "#1a4d2a"
BG_GREEN_LIGHT = "#1a3d2a"
BG_RED_STRONG = "#5a1a1a"
BG_RED_MEDIUM = "#4d1a1a"
BG_RED_LIGHT = "#3d1a1a"
BG_ORANGE_LIGHT = "#3d2a1a"
BG_ORANGE = "#4d2a1a"
BG_YELLOW_LIGHT = "#3d3d1a"
TEXT_GREEN = "#4caf50"
TEXT_GREEN_MED = "#66bb6a"
TEXT_RED = "#ef5350"
TEXT_RED_STRONG = "#ff1744"
TEXT_ORANGE = "#ff9800"
TEXT_ORANGE_LIGHT = "#ffb74d"
TEXT_YELLOW = "#ffee58"
TEXT_LIGHT = "#e0e0e0"
TEXT_MUTED = "#b0b0b0"
PACE_COLORS = {"H": "#ef5350", "M": "#ffa726", "S": "#42a5f5"}

# AI評価ランク表示（内部ティア → ユーザー向けラベル）
TIER_TO_LABEL = {
    "130%": "S",
    "120%": "A",
    "110%": "B",
    "100%": "C",
    "90%": "C-",
    "80%": "D",
    "70%": "D-",
    "60%": "E",
    "50%": "E-",
    "40%": "F",
}

# AI評価ランクの色設定
TIER_STYLES = {
    "S": {
        "bg": BG_GREEN_STRONG,
        "color": TEXT_GREEN,
        "font_weight": "bold",
        "font_size": "1.1em",
    },
    "A": {
        "bg": BG_GREEN_MEDIUM,
        "color": TEXT_GREEN_MED,
        "font_weight": "bold",
        "font_size": "1.05em",
    },
    "B": {
        "bg": BG_GREEN_LIGHT,
        "color": TEXT_GREEN_MED,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "C": {
        "bg": "",
        "color": TEXT_LIGHT,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "C-": {
        "bg": BG_YELLOW_LIGHT,
        "color": TEXT_YELLOW,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "D": {
        "bg": BG_ORANGE_LIGHT,
        "color": TEXT_ORANGE_LIGHT,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "D-": {
        "bg": BG_ORANGE,
        "color": TEXT_ORANGE,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "E": {
        "bg": BG_RED_LIGHT,
        "color": TEXT_RED,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "E-": {
        "bg": BG_RED_MEDIUM,
        "color": TEXT_RED,
        "font_weight": "bold",
        "font_size": "1em",
    },
    "F": {
        "bg": BG_RED_STRONG,
        "color": TEXT_RED_STRONG,
        "font_weight": "bold",
        "font_size": "1.1em",
    },
}

# AI評価ランクのソート順（高い方が上位）
TIER_SORT_ORDER = {
    "S": 0, "A": 1, "B": 2, "C": 3, "C-": 4,
    "D": 6, "D-": 7, "E": 8, "E-": 9, "F": 10,
    "-": 5,
}

# 内部ティア値 → ソート順（JSON上のand_filter_tier用）
INTERNAL_TIER_SORT = {
    "130%": 0, "120%": 1, "110%": 2, "100%": 3, "90%": 4,
    "80%": 6, "70%": 7, "60%": 8, "50%": 9, "40%": 10,
    "-": 5, None: 5,
}


# ============================================================
# CSS注入
# ============================================================

st.markdown("""
<style>
    .stMetric label { color: #b0b0b0 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e0e0e0 !important; }
    div[data-testid="stExpander"] details summary p { color: #e0e0e0 !important; }
    .race-header { color: #e0e0e0; font-size: 1.1em; font-weight: bold; }
    .buy-tag {
        background-color: #1a5c2a; color: #4caf50;
        padding: 2px 8px; border-radius: 4px; font-weight: bold;
    }
    .avoid-tag {
        background-color: #5a1a1a; color: #ef5350;
        padding: 2px 8px; border-radius: 4px; font-weight: bold;
    }
    .update-info {
        color: #888888; font-size: 0.85em; text-align: right;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# データ読み込み
# ============================================================

@st.cache_data(ttl=120)
def load_prediction_files():
    """data/ 配下のJSONファイル一覧を返す（日付降順）"""
    if not DATA_DIR.exists():
        return []
    files = sorted(DATA_DIR.glob("predictions_*.json"), reverse=True)
    result = []
    for f in files:
        stem = f.stem
        date_str = stem.replace("predictions_", "")
        try:
            d = datetime.strptime(date_str, "%Y%m%d").date()
            result.append({"path": f, "date": d, "label": d.strftime("%Y-%m-%d (%a)")})
        except ValueError:
            continue
    return result


@st.cache_data(ttl=120)
def load_prediction_data(file_path: str):
    """JSONファイルを読み込む"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_ratings_data():
    """ratings.json を読み込む"""
    ratings_path = DATA_DIR / "ratings.json"
    if not ratings_path.exists():
        return None
    with open(ratings_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ヘルパー
# ============================================================

def _tier_to_label(internal_tier: str | None) -> str:
    """内部ティア値（"130%"等）をユーザー向けラベル（"S"等）に変換"""
    if not internal_tier:
        return "-"
    return TIER_TO_LABEL.get(internal_tier, "-")


def _roi_to_display(val_str: str) -> str:
    """ROI文字列をそのまま返す（表示用）。'-'はそのまま。"""
    if not val_str or val_str == "-":
        return "-"
    return val_str


def _extract_roi_number(val_str: str) -> float | None:
    """ROI文字列から数値部分を抽出する。"""
    if not val_str or val_str == "-" or "%" not in val_str:
        return None
    try:
        s = val_str.replace("~", "").replace("(", "").replace(")", "")
        s = s.replace("芝", "").replace("ダ", "")
        s = s.split("走")[0]
        return float(s.replace("%", ""))
    except (ValueError, TypeError):
        return None


def roi_style(val_str: str) -> str:
    """ROI文字列からセルスタイルを返す"""
    v = _extract_roi_number(val_str)
    if v is None:
        return ""
    if v >= 150:
        return f"background-color: {BG_GREEN_STRONG}; color: {TEXT_GREEN}; font-weight: bold"
    if v >= 130:
        return f"background-color: {BG_GREEN_MEDIUM}; color: {TEXT_GREEN_MED}; font-weight: bold"
    if v >= 110:
        return f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
    if v >= 100:
        return f"color: {TEXT_LIGHT}"
    if v >= 80:
        return f"background-color: {BG_RED_LIGHT}; color: {TEXT_RED}"
    return f"background-color: {BG_RED_STRONG}; color: {TEXT_RED}"


def tier_style(val_str: str) -> str:
    """AI評価ラベルからセルスタイルを返す"""
    ts = TIER_STYLES.get(val_str)
    if ts is None:
        return f"color: {TEXT_MUTED}"
    parts = []
    if ts["bg"]:
        parts.append(f"background-color: {ts['bg']}")
    parts.append(f"color: {ts['color']}")
    parts.append(f"font-weight: {ts['font_weight']}")
    parts.append(f"font-size: {ts['font_size']}")
    return "; ".join(parts)


def _format_context_bias(val_str: str) -> str:
    """条件バイアス値をわかりやすい表現に変換"""
    if not val_str or val_str == "-":
        return "-"
    if "有利転換" in val_str:
        return "今回有利"
    if "不利転換" in val_str:
        return "今回不利"
    # 数値のみの場合
    try:
        v = float(val_str.split()[0])
        if v > 1.10:
            return "やや有利"
        elif v < 0.90:
            return "やや不利"
        else:
            return "-"
    except (ValueError, IndexError):
        return val_str


def _simplify_signal_tags(tags_str: str) -> str:
    """signal_tagsを一般向けに簡略化する。
    内部名（rank_dev系, cushion×rank_dev等）は除去し、
    有利/不利のカウントだけ表示する。"""
    if not tags_str:
        return "-"
    parts = [t.strip() for t in tags_str.split(",") if t.strip()]
    adv_count = sum(1 for p in parts if p.startswith("有利:"))
    dis_count = sum(1 for p in parts if p.startswith("不利:"))
    if adv_count == 0 and dis_count == 0:
        return "-"
    result_parts = []
    if adv_count > 0:
        result_parts.append(f"好材料{adv_count}")
    if dis_count > 0:
        result_parts.append(f"懸念{dis_count}")
    return " / ".join(result_parts)


def highlight_row(row):
    """テーブル行のハイライト"""
    styles = [""] * len(row)
    cols = list(row.index)

    # 推奨ハイライト
    if "推奨" in cols:
        idx = cols.index("推奨")
        v = row["推奨"]
        if v == "買":
            styles = [f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_LIGHT}"] * len(row)
            styles[idx] = f"background-color: {BG_GREEN_STRONG}; color: {TEXT_GREEN}; font-weight: bold"
        elif v == "避":
            styles = [f"background-color: {BG_RED_LIGHT}; color: {TEXT_LIGHT}"] * len(row)
            styles[idx] = f"background-color: {BG_RED_STRONG}; color: {TEXT_RED}; font-weight: bold"

    # 順位ハイライト
    if "順位" in cols:
        idx = cols.index("順位")
        rank = row["順位"]
        if rank == 1:
            styles[idx] = f"background-color: {BG_GREEN_STRONG}; color: #ffffff; font-weight: bold"
        elif rank <= 3:
            styles[idx] = f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_LIGHT}"

    # AI評価列ハイライト
    if "AI評価" in cols:
        idx = cols.index("AI評価")
        s = tier_style(str(row["AI評価"]))
        if s:
            styles[idx] = s

    # 馬場適性列ハイライト
    if "馬場適性" in cols:
        idx = cols.index("馬場適性")
        s = roi_style(str(row["馬場適性"]))
        if s:
            styles[idx] = s

    # 馬場×血統列ハイライト
    if "馬場×血統" in cols:
        idx = cols.index("馬場×血統")
        s = roi_style(str(row["馬場×血統"]))
        if s:
            styles[idx] = s

    # 血統適性列ハイライト
    if "血統適性" in cols:
        idx = cols.index("血統適性")
        s = roi_style(str(row["血統適性"]))
        if s:
            styles[idx] = s

    # 勝率ハイライト
    if "勝率" in cols:
        idx = cols.index("勝率")
        win_str = str(row["勝率"])
        try:
            win_val = float(win_str.replace("%", "")) if "%" in win_str else None
        except (ValueError, TypeError):
            win_val = None
        if win_val is not None:
            if win_val >= 20:
                styles[idx] = f"background-color: {BG_GREEN_STRONG}; color: {TEXT_GREEN}; font-weight: bold"
            elif win_val >= 10:
                styles[idx] = f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
            elif win_val < 5:
                styles[idx] = f"color: {TEXT_MUTED}"

    # 複勝率ハイライト
    if "複勝率" in cols:
        idx = cols.index("複勝率")
        place_str = str(row["複勝率"])
        try:
            place_val = float(place_str.replace("%", "")) if "%" in place_str else None
        except (ValueError, TypeError):
            place_val = None
        if place_val is not None:
            if place_val >= 50:
                styles[idx] = f"background-color: {BG_GREEN_STRONG}; color: {TEXT_GREEN}; font-weight: bold"
            elif place_val >= 30:
                styles[idx] = f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
            elif place_val < 15:
                styles[idx] = f"color: {TEXT_MUTED}"

    # オッズハイライト
    if "オッズ" in cols:
        idx = cols.index("オッズ")
        odds_str = str(row["オッズ"])
        try:
            odds_val = float(odds_str) if odds_str != "-" else None
        except (ValueError, TypeError):
            odds_val = None
        if odds_val is not None:
            if odds_val <= 3.0:
                styles[idx] = f"color: {TEXT_ORANGE}; font-weight: bold"
            elif odds_val <= 10.0:
                styles[idx] = f"color: {TEXT_LIGHT}"
            elif odds_val <= 30.0:
                styles[idx] = f"color: {TEXT_MUTED}"
            else:
                styles[idx] = f"color: {TEXT_MUTED}"

    # 展開列ハイライト
    if "展開" in cols:
        idx = cols.index("展開")
        v = row["展開"]
        if v == "有利":
            styles[idx] = f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
        elif v == "不利":
            styles[idx] = f"background-color: {BG_RED_LIGHT}; color: {TEXT_RED}"

    # 前走比較列ハイライト
    if "前走比較" in cols:
        idx = cols.index("前走比較")
        v = str(row["前走比較"])
        if "今回有利" in v:
            styles[idx] = f"background-color: {BG_GREEN_MEDIUM}; color: {TEXT_GREEN}; font-weight: bold"
        elif "今回不利" in v:
            styles[idx] = f"background-color: {BG_RED_MEDIUM}; color: {TEXT_RED}; font-weight: bold"
        elif "やや有利" in v:
            styles[idx] = f"color: {TEXT_GREEN_MED}"
        elif "やや不利" in v:
            styles[idx] = f"color: {TEXT_ORANGE}"

    return styles


def render_pace_prediction(pace_data: dict):
    """展開予測を表示"""
    if not pace_data:
        return

    pred = pace_data.get("predicted_pace", "?")
    label = pace_data.get("pace_label", pred)
    probs = pace_data.get("pace_probs", {})
    color = PACE_COLORS.get(pred, TEXT_LIGHT)

    # ペース確率をわかりやすく表示
    pace_label_map = {"H": "ハイ", "M": "ミドル", "S": "スロー"}
    prob_str = " / ".join(
        f"{pace_label_map.get(k, k)}:{v:.0%}" for k, v in probs.items()
    )
    st.markdown(
        f"**ペース予想**: <span style='color:{color}; font-size:1.2em; "
        f"font-weight:bold;'>{label}</span> ({prob_str})",
        unsafe_allow_html=True,
    )

    adv = pace_data.get("advantaged_styles", [])
    if adv:
        st.caption(f"有利な脚質: {', '.join(adv)}")


# ============================================================
# コース分析ページ
# ============================================================

@st.cache_data(ttl=300)
def load_section_coefficients():
    """section_coefficients.json を読み込む"""
    coeff_path = DATA_DIR / "section_coefficients.json"
    if not coeff_path.exists():
        return None
    with open(coeff_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_structural_model():
    """structural_load_model.json を読み込む"""
    model_path = DATA_DIR / "structural_load_model.json"
    if not model_path.exists():
        return None
    with open(model_path, "r", encoding="utf-8") as f:
        return json.load(f)


# 係数の日本語ラベル
COEFF_LABELS = {
    "distance_loss": "距離ロス",
    "position": "位置(前後)",
    "pos_category": "位置(内外)",
    "acceleration": "加速度",
    "cumulative_load": "蓄積負荷",
    "prev_speed": "前区間速度",
    "cushion_value": "クッション値",
    "moisture": "含水率",
    "pace_ratio": "ペース比",
    "dist_x_load": "距離ロスx負荷",
    "dist_x_pace": "距離ロスxペース",
    "cushion_x_moisture": "クッションx含水",
    "pos_x_load": "位置x負荷",
}

# 統一構造モデルの係数ラベル (旧Ridgeモデル用)
STRUCTURAL_COEFF_LABELS_RIDGE = {
    "is_corner": "コーナー区間",
    "corner_curvature": "コーナー曲率",
    "gradient": "勾配(%)",
    "gradient_abs": "勾配絶対値",
    "distance_loss": "距離ロス(m)",
    "cumulative_distance": "累積距離(m)",
    "cushion_value": "クッション値",
    "moisture": "含水率(%)",
    "track_condition_code": "馬場状態",
    "corner_x_dist_loss": "曲率x距離ロス",
    "cum_dist_x_gradient_abs": "累積距離x勾配絶対値",
    "moisture_x_gradient_abs": "含水率x勾配絶対値",
}

# 物理モデルのパラメータラベル
STRUCTURAL_COEFF_LABELS_PHYSICS = {
    "a": "a: コーナー遠心力スケール",
    "b": "b: 含水率xコーナー",
    "c": "c: 距離ロスxコーナー",
    "d": "d: 勾配スケール",
    "e": "e: 累積負荷x坂",
    "f": "f: 含水率x坂",
    "g": "g: コーナー距離ロス追加",
    "h": "h: 含水率->馬場負荷",
    "i": "i: 馬場状態->馬場負荷",
    "j": "j: クッション->馬場負荷",
    "k": "k: 勾配x馬場",
    "l": "l: 累積距離x馬場",
}

# 互換性エイリアス (page_course_analysis で使用)
STRUCTURAL_COEFF_LABELS = {**STRUCTURAL_COEFF_LABELS_RIDGE, **STRUCTURAL_COEFF_LABELS_PHYSICS}

# 構造タイプの日本語ラベル
STRUCTURE_LABELS = {
    "straight": "直線",
    "straight_uphill": "直線(上り)",
    "straight_downhill": "直線(下り)",
    "corner": "コーナー",
    "corner_uphill": "コーナー(上り)",
    "corner_downhill": "コーナー(下り)",
    "uphill_uphill": "急坂(上り)",
    "unknown": "-",
}

# フラグの日本語ラベルと色
FLAG_STYLES = {
    "distance_loss_negative": {"label": "距離ロス負", "color": TEXT_RED},
    "acceleration_positive": {"label": "加速度正", "color": TEXT_ORANGE},
    "very_low_r_squared": {"label": "R2極低", "color": TEXT_RED},
    "low_r_squared": {"label": "R2低", "color": TEXT_YELLOW},
    "low_sample_size": {"label": "N少", "color": TEXT_MUTED},
    "cushion_negative": {"label": "クッション負", "color": TEXT_ORANGE},
}


def _coeff_cell_style(value, p_value):
    """係数値+p値に応じたセルスタイル"""
    if p_value is not None and p_value > 0.1:
        return f"color: {TEXT_MUTED}"
    if value > 0:
        return f"color: {TEXT_RED}"
    elif value < 0:
        return f"color: #42a5f5"
    return f"color: {TEXT_LIGHT}"


def _r2_style(r2):
    """R2値に応じたスタイル"""
    if r2 >= 0.5:
        return f"background-color: {BG_GREEN_MEDIUM}; color: {TEXT_GREEN_MED}"
    elif r2 >= 0.2:
        return f"color: {TEXT_LIGHT}"
    elif r2 >= 0.05:
        return f"color: {TEXT_YELLOW}"
    else:
        return f"background-color: {BG_RED_LIGHT}; color: {TEXT_RED}"


def page_course_analysis():
    """コース分析ページ"""

    # --- 統一構造負荷モデル表示 ---
    structural_model = load_structural_model()
    if structural_model:
        model_type = structural_model.get("model", "ridge")
        is_physics = model_type == "physics"

        if is_physics:
            st.header("物理法則ベース構造負荷モデル")
            st.caption(
                f"非線形最小二乗法 (遠心力+位置エネルギー+距離ロス+馬場抵抗) / "
                f"更新: {structural_model.get('created_at', '?')[:16]} / "
                f"v{structural_model.get('version', '?')}"
            )
        else:
            st.header("統一構造負荷モデル")
            st.caption(
                f"全コース横断Ridge回帰 / "
                f"更新: {structural_model.get('created_at', '?')[:16]} / "
                f"v{structural_model.get('version', '?')}"
            )

        # メトリクス
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("R\u00b2", f"{structural_model.get('r_squared', 0):.4f}")
        with m2:
            st.metric("OOS R\u00b2", f"{structural_model.get('oos_r_squared', 0):.4f}")
        with m3:
            st.metric("OOS RMSE", f"{structural_model.get('oos_rmse', 0):.4f}s")
        with m4:
            st.metric("N", f"{structural_model.get('n_samples', 0):,}")

        if is_physics:
            # --- 物理モデル: 数式表示 ---
            st.subheader("数式")
            formula_details = structural_model.get("formula_details", {})
            st.markdown(f"**全体**: `{structural_model.get('formula', '')}`")
            for fname, fexpr in formula_details.items():
                st.markdown(f"- `{fname}` = `{fexpr}`")

            # パラメータテーブル
            st.subheader("フィットされたパラメータ")
            params = structural_model.get("parameters", {})
            validity = structural_model.get("physical_validity", {})
            label_map = STRUCTURAL_COEFF_LABELS_PHYSICS

            param_rows = []
            for pname in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]:
                p = params.get(pname, {})
                param_rows.append({
                    "パラメータ": label_map.get(pname, pname),
                    "値": p.get("value", 0),
                    "SE": p.get("se", 0),
                    "t値": p.get("t_stat", 0),
                    "p値": p.get("p_value", 1),
                    "有意": "***" if p.get("p_value", 1) < 0.001 else "**" if p.get("p_value", 1) < 0.01 else "*" if p.get("p_value", 1) < 0.05 else "",
                    "説明": p.get("description", ""),
                })

            df_params = pd.DataFrame(param_rows)

            def style_physics_param(row):
                styles = [""] * len(row)
                cols = list(row.index)
                if "値" in cols:
                    idx = cols.index("値")
                    try:
                        val = float(row["値"])
                        if val > 0:
                            styles[idx] = f"color: {TEXT_RED}; font-weight: bold"
                        elif val < 0:
                            styles[idx] = f"color: #42a5f5; font-weight: bold"
                    except (ValueError, TypeError):
                        pass
                if "p値" in cols:
                    idx = cols.index("p値")
                    try:
                        pv = float(row["p値"])
                        if pv < 0.001:
                            styles[idx] = f"color: {TEXT_GREEN}"
                        elif pv < 0.05:
                            styles[idx] = f"color: {TEXT_GREEN_MED}"
                        else:
                            styles[idx] = f"color: {TEXT_MUTED}"
                    except (ValueError, TypeError):
                        pass
                return styles

            st.dataframe(
                df_params.style.apply(style_physics_param, axis=1).format({
                    "値": "{:.8f}",
                    "SE": "{:.8f}",
                    "t値": "{:.2f}",
                    "p値": "{:.6f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=min(len(df_params) * 38 + 40, 600),
            )

            # 各項の寄与統計
            contrib = structural_model.get("contribution_stats", {})
            if contrib:
                st.subheader("各項の寄与 (秒)")
                contrib_rows = []
                for cname, cdata in contrib.items():
                    contrib_rows.append({
                        "項": cname,
                        "平均": cdata.get("mean", 0),
                        "標準偏差": cdata.get("std", 0),
                        "最小": cdata.get("min", 0),
                        "最大": cdata.get("max", 0),
                    })
                df_contrib = pd.DataFrame(contrib_rows)
                st.dataframe(
                    df_contrib.style.format({
                        "平均": "{:.4f}",
                        "標準偏差": "{:.4f}",
                        "最小": "{:.4f}",
                        "最大": "{:.4f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            # 物理妥当性チェック
            with st.expander("物理妥当性チェック", expanded=False):
                for fn, check in validity.items():
                    status_color = TEXT_GREEN if check.get("valid") else TEXT_RED
                    status_icon = "OK" if check.get("valid") else "NG"
                    st.markdown(
                        f"<span style='color: {status_color}; font-weight: bold'>[{status_icon}]</span> "
                        f"**{fn}**: {check.get('actual', '?')} "
                        f"(expected: {check.get('expected', '?')}) - {check.get('note', '')}",
                        unsafe_allow_html=True,
                    )

        else:
            # --- 旧Ridgeモデル表示 ---
            st.subheader("構造要素の係数")

            coefficients = structural_model.get("coefficients", {})
            interp = structural_model.get("interpretation", [])
            interp_map = {item["feature"]: item for item in interp}
            validity = structural_model.get("physical_validity", {})

            coeff_rows = []
            for fn in structural_model.get("feature_names", []):
                c = coefficients.get(fn, {})
                v = validity.get(fn, {})
                i = interp_map.get(fn, {})
                coeff_rows.append({
                    "変数": STRUCTURAL_COEFF_LABELS.get(fn, fn),
                    "係数": c.get("value", 0),
                    "SE": c.get("se", 0),
                    "t値": c.get("t_stat", 0),
                    "p値": c.get("p_value", 1),
                    "有意": "***" if c.get("p_value", 1) < 0.001 else "**" if c.get("p_value", 1) < 0.01 else "*" if c.get("p_value", 1) < 0.05 else "",
                    "妥当性": "OK" if v.get("valid", True) else "NG",
                    "解釈": i.get("description", ""),
                })

            df_coeff = pd.DataFrame(coeff_rows)

            def style_structural_coeff(row):
                styles = [""] * len(row)
                cols = list(row.index)
                if "係数" in cols:
                    idx = cols.index("係数")
                    try:
                        val = float(row["係数"])
                        if val > 0:
                            styles[idx] = f"color: {TEXT_RED}; font-weight: bold"
                        elif val < 0:
                            styles[idx] = f"color: #42a5f5; font-weight: bold"
                    except (ValueError, TypeError):
                        pass
                if "p値" in cols:
                    idx = cols.index("p値")
                    try:
                        p = float(row["p値"])
                        if p < 0.001:
                            styles[idx] = f"color: {TEXT_GREEN}"
                        elif p < 0.05:
                            styles[idx] = f"color: {TEXT_GREEN_MED}"
                        else:
                            styles[idx] = f"color: {TEXT_MUTED}"
                    except (ValueError, TypeError):
                        pass
                if "妥当性" in cols:
                    idx = cols.index("妥当性")
                    if row["妥当性"] == "NG":
                        styles[idx] = f"background-color: {BG_RED_LIGHT}; color: {TEXT_RED}; font-weight: bold"
                    else:
                        styles[idx] = f"color: {TEXT_GREEN}"
                return styles

            st.dataframe(
                df_coeff.style.apply(style_structural_coeff, axis=1).format({
                    "係数": "{:.6f}",
                    "SE": "{:.6f}",
                    "t値": "{:.2f}",
                    "p値": "{:.6f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=min(len(df_coeff) * 38 + 40, 700),
            )

            st.caption(f"intercept = {structural_model.get('intercept', 0):.4f}秒")

            # 物理妥当性チェック
            with st.expander("物理妥当性チェック", expanded=False):
                for fn, check in validity.items():
                    status_color = TEXT_GREEN if check.get("valid") else TEXT_RED
                    status_icon = "OK" if check.get("valid") else "NG"
                    st.markdown(
                        f"<span style='color: {status_color}; font-weight: bold'>[{status_icon}]</span> "
                        f"**{STRUCTURAL_COEFF_LABELS.get(fn, fn)}**: {check.get('actual', '?')} "
                        f"(expected: {check.get('expected', '?')}) - {check.get('note', '')}",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

    # --- 従来のコース別分析 ---
    coeff_data = load_section_coefficients()
    if not coeff_data:
        if not structural_model:
            st.warning("コース分析データがありません。")
        return

    courses = coeff_data.get("courses", {})
    metadata = coeff_data.get("metadata", {})

    st.header("コース別 区間係数")
    if metadata.get("generated_at"):
        try:
            gen_dt = datetime.fromisoformat(metadata["generated_at"])
            st.caption(f"更新: {gen_dt.strftime('%Y-%m-%d %H:%M')} / "
                       f"{metadata.get('n_courses', 0)}コース / "
                       f"{metadata.get('total_samples', 0):,}サンプル")
        except ValueError:
            pass

    # サイドバーでコース選択
    with st.sidebar:
        st.subheader("コース分析設定")
        course_keys = sorted(courses.keys())
        # 芝/ダートでグループ分け
        turf_courses = [k for k in course_keys if "_芝_" in k]
        dirt_courses = [k for k in course_keys if "_ダート_" in k]

        surface_filter = st.radio("馬場", ["全て", "芝", "ダート"], horizontal=True)
        if surface_filter == "芝":
            filtered_courses = turf_courses
        elif surface_filter == "ダート":
            filtered_courses = dirt_courses
        else:
            filtered_courses = course_keys

        if not filtered_courses:
            st.warning("該当コースがありません。")
            return

        selected_course = st.selectbox(
            "コース選択",
            filtered_courses,
            format_func=lambda x: f"{x} ({courses[x]['n_races']}R)",
        )

        # 有意水準フィルタ
        p_threshold = st.slider("有意水準 (p値)", 0.01, 0.50, 0.10, 0.01)

        # 表示する係数の選択
        all_feature_names = metadata.get("features", list(COEFF_LABELS.keys()))
        show_features = st.multiselect(
            "表示する係数",
            all_feature_names,
            default=["distance_loss", "position", "pos_category",
                     "acceleration", "cumulative_load", "prev_speed",
                     "cushion_value", "moisture", "pace_ratio"],
            format_func=lambda x: COEFF_LABELS.get(x, x),
        )

    if not selected_course:
        return

    course_data = courses[selected_course]
    sections = course_data.get("sections", [])

    if not sections:
        st.info("このコースのデータはありません。")
        return

    # --- コース情報ヘッダー ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("レース数", f"{course_data['n_races']}")
    with col2:
        total_n = sum(s.get("n_samples", 0) for s in sections)
        st.metric("総サンプル数", f"{total_n:,}")
    with col3:
        avg_r2 = sum(s.get("r_squared", 0) for s in sections) / len(sections) if sections else 0
        st.metric("平均R2", f"{avg_r2:.3f}")

    # --- 区間一覧テーブル ---
    st.subheader("区間一覧")

    table_rows = []
    for sec in sections:
        row = {
            "区間": sec.get("section_idx", ""),
            "距離": sec.get("distance_range", ""),
            "構造": STRUCTURE_LABELS.get(sec.get("structure", ""), sec.get("structure", "")),
            "N": sec.get("n_samples", 0),
            "R2": sec.get("r_squared", 0),
        }

        # OOS R2
        oos = sec.get("oos", {})
        row["OOS R2"] = oos.get("r_squared", "-")

        # 各係数
        coefficients = sec.get("coefficients", {})
        for fname in show_features:
            c = coefficients.get(fname, {})
            val = c.get("value", 0)
            p = c.get("p_value", 1)
            if p <= p_threshold:
                row[COEFF_LABELS.get(fname, fname)] = f"{val:.4f}"
            else:
                row[COEFF_LABELS.get(fname, fname)] = "-"

        # フラグ
        flags = sec.get("flags", [])
        flag_labels = []
        for f in flags:
            fs = FLAG_STYLES.get(f, {"label": f})
            flag_labels.append(fs["label"])
        row["フラグ"] = ", ".join(flag_labels) if flag_labels else "-"

        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)

    # スタイリング関数
    def style_course_table(row):
        styles = [""] * len(row)
        cols = list(row.index)

        # R2列ハイライト
        if "R2" in cols:
            idx = cols.index("R2")
            try:
                r2_val = float(row["R2"])
                styles[idx] = _r2_style(r2_val)
            except (ValueError, TypeError):
                pass

        # OOS R2列ハイライト
        if "OOS R2" in cols:
            idx = cols.index("OOS R2")
            try:
                r2_val = float(row["OOS R2"])
                styles[idx] = _r2_style(r2_val)
            except (ValueError, TypeError):
                styles[idx] = f"color: {TEXT_MUTED}"

        # 係数列ハイライト
        for fname in show_features:
            label = COEFF_LABELS.get(fname, fname)
            if label in cols:
                idx = cols.index(label)
                val_str = str(row[label])
                if val_str == "-":
                    styles[idx] = f"color: {TEXT_MUTED}"
                else:
                    try:
                        val = float(val_str)
                        if val > 0:
                            styles[idx] = f"color: {TEXT_RED}"
                        elif val < 0:
                            styles[idx] = f"color: #42a5f5"
                    except ValueError:
                        pass

        # フラグ列ハイライト
        if "フラグ" in cols:
            idx = cols.index("フラグ")
            flag_str = str(row["フラグ"])
            if "距離ロス負" in flag_str or "R2極低" in flag_str:
                styles[idx] = f"background-color: {BG_RED_LIGHT}; color: {TEXT_RED}"
            elif "R2低" in flag_str or "クッション負" in flag_str:
                styles[idx] = f"color: {TEXT_YELLOW}"
            elif "N少" in flag_str:
                styles[idx] = f"color: {TEXT_MUTED}"

        # N列ハイライト
        if "N" in cols:
            idx = cols.index("N")
            try:
                n_val = int(row["N"])
                if n_val < 100:
                    styles[idx] = f"color: {TEXT_MUTED}"
            except (ValueError, TypeError):
                pass

        return styles

    styled_table = df_table.style.apply(style_course_table, axis=1)
    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
        height=min(len(df_table) * 38 + 40, 600),
    )

    # --- 係数の棒グラフ ---
    st.subheader("区間別 係数グラフ")

    # 表示する係数を選択
    chart_feature = st.selectbox(
        "表示する指標",
        show_features,
        format_func=lambda x: COEFF_LABELS.get(x, x),
        key="chart_feature",
    )

    chart_data = []
    for sec in sections:
        coefficients = sec.get("coefficients", {})
        c = coefficients.get(chart_feature, {})
        val = c.get("value", 0)
        p = c.get("p_value", 1)
        structure = sec.get("structure", "unknown")
        chart_data.append({
            "区間": f"{sec['section_idx']}: {sec.get('distance_range', '')}",
            "係数値": val,
            "有意": "有意" if p <= p_threshold else "非有意",
            "構造": STRUCTURE_LABELS.get(structure, structure),
        })

    df_chart = pd.DataFrame(chart_data)

    # 色分け: 有意な正=赤, 有意な負=青, 非有意=灰
    colors = []
    for _, r in df_chart.iterrows():
        if r["有意"] == "非有意":
            colors.append(TEXT_MUTED)
        elif r["係数値"] > 0:
            colors.append(TEXT_RED)
        else:
            colors.append("#42a5f5")

    # Streamlit bar chart (simple)
    chart_df_display = df_chart.set_index("区間")[["係数値"]]
    st.bar_chart(chart_df_display, color="#42a5f5")

    # 有意性の凡例
    st.caption(
        f"**{COEFF_LABELS.get(chart_feature, chart_feature)}** "
        f"({metadata.get('feature_units', {}).get(chart_feature, '秒')}) / "
        f"p < {p_threshold} のみ有意"
    )

    # --- コース構造の表示 ---
    st.subheader("コース構造詳細")

    structure_rows = []
    for sec in sections:
        sd = sec.get("structure_detail", {})
        structure_rows.append({
            "区間": sec.get("section_idx", ""),
            "距離": sec.get("distance_range", ""),
            "構造": STRUCTURE_LABELS.get(sec.get("structure", ""), sec.get("structure", "")),
            "勾配(%)": sd.get("gradient", 0.0),
            "コーナー半径": sd.get("corner_radius", "-") or "-",
            "備考": sd.get("notes", ""),
        })

    df_structure = pd.DataFrame(structure_rows)

    def style_structure_table(row):
        styles = [""] * len(row)
        cols = list(row.index)
        if "構造" in cols:
            idx = cols.index("構造")
            s = str(row["構造"])
            if "上り" in s or "急坂" in s:
                styles[idx] = f"color: {TEXT_RED}"
            elif "下り" in s:
                styles[idx] = f"color: #42a5f5"
            elif "コーナー" in s:
                styles[idx] = f"color: {TEXT_ORANGE}"
        if "勾配(%)" in cols:
            idx = cols.index("勾配(%)")
            try:
                g = float(row["勾配(%)"])
                if g > 0:
                    styles[idx] = f"color: {TEXT_RED}"
                elif g < 0:
                    styles[idx] = f"color: #42a5f5"
            except (ValueError, TypeError):
                pass
        return styles

    st.dataframe(
        df_structure.style.apply(style_structure_table, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # --- OOS係数安定性 ---
    with st.expander("OOS係数安定性 (LOMO-CV)", expanded=False):
        oos_rows = []
        for sec in sections:
            oos = sec.get("oos", {})
            stability = oos.get("coefficient_stability", {})
            if not stability:
                continue
            row = {
                "区間": sec.get("section_idx", ""),
                "OOS R2": oos.get("r_squared", "-"),
                "RMSE": oos.get("rmse", "-"),
                "Folds": oos.get("n_folds", "-"),
            }
            for fname in show_features:
                cs = stability.get(fname, {})
                mean_val = cs.get("mean", 0)
                std_val = cs.get("std", 0)
                cv_val = cs.get("cv")
                if cv_val is not None and abs(mean_val) > 1e-8:
                    row[COEFF_LABELS.get(fname, fname)] = f"{mean_val:.4f} (CV={cv_val:.2f})"
                else:
                    row[COEFF_LABELS.get(fname, fname)] = f"{mean_val:.4f}"
            oos_rows.append(row)

        if oos_rows:
            df_oos = pd.DataFrame(oos_rows)
            st.dataframe(df_oos, use_container_width=True, hide_index=True)
        else:
            st.info("OOSデータがありません。")

    # フッター
    st.markdown("---")
    st.caption(
        "コース分析: 各区間のラップタイムを説明する回帰係数。"
        "正=遅くなる要因、負=速くなる要因。Ridge回帰 + LOMO-CV。"
    )


# ============================================================
# コースマップページ
# ============================================================

# --- 競馬場の形状パラメータ ---
# 各競馬場のコース形状を実際の構造で再現する。
# home_straight: ホームストレッチ(直線)の長さ (m)
# back_straight: バックストレッチ(向正面)の長さ (m)
# corner_radius_12: 1C-2Cの半径 (m)
# corner_radius_34: 3C-4Cの半径 (m)
# direction: "右" or "左" or "直線"
# perimeter: 1周の距離 (m, 芝外回り基準)

VENUE_SHAPES = {
    "東京": {
        "home_straight": 525,
        "back_straight": 450,
        "corner_radius_12": 130,
        "corner_radius_34": 130,
        "direction": "左",
        "perimeter": 2083,
    },
    "中山": {
        "home_straight": 310,
        "back_straight": 240,
        "corner_radius_12": 100,
        "corner_radius_34": 80,
        "direction": "右",
        "perimeter": 1840,
    },
    "阪神": {
        "home_straight": 473,
        "back_straight": 380,
        "corner_radius_12": 130,
        "corner_radius_34": 120,
        "direction": "右",
        "perimeter": 2089,
    },
    "京都": {
        "home_straight": 404,
        "back_straight": 380,
        "corner_radius_12": 120,
        "corner_radius_34": 120,
        "direction": "右",
        "perimeter": 1894,
    },
    "中京": {
        "home_straight": 412,
        "back_straight": 340,
        "corner_radius_12": 110,
        "corner_radius_34": 110,
        "direction": "左",
        "perimeter": 1705,
    },
    "新潟": {
        "home_straight": 659,
        "back_straight": 350,
        "corner_radius_12": 110,
        "corner_radius_34": 110,
        "direction": "左",
        "perimeter": 2223,
    },
    "札幌": {
        "home_straight": 266,
        "back_straight": 260,
        "corner_radius_12": 105,
        "corner_radius_34": 105,
        "direction": "右",
        "perimeter": 1640,
    },
    "函館": {
        "home_straight": 262,
        "back_straight": 260,
        "corner_radius_12": 100,
        "corner_radius_34": 100,
        "direction": "右",
        "perimeter": 1626,
    },
    "福島": {
        "home_straight": 292,
        "back_straight": 270,
        "corner_radius_12": 100,
        "corner_radius_34": 100,
        "direction": "右",
        "perimeter": 1600,
    },
    "小倉": {
        "home_straight": 293,
        "back_straight": 270,
        "corner_radius_12": 90,
        "corner_radius_34": 85,
        "direction": "右",
        "perimeter": 1616,
    },
}


def _generate_track_path(venue_name: str, n_points: int = 1000) -> list[tuple[float, float]]:
    """競馬場のコースを実際の形状に忠実に (x, y) 座標列として生成する。

    コース構成 (上から見た図):
      ホームストレッチ(下側, y=0) ─ 左右のコーナー(半円) ─ バックストレッチ(上側)

    path[0] = ゴール位置 = ホームストレッチ右端 (home/2, 0)
    走行方向に沿って1周分の座標を返す。

    右回り(時計回り):
      ゴール → ホーム(右→左) → 4C-3C(左側半円) → バック(左→右) → 2C-1C(右側半円) → ゴール
    左回り(反時計回り):
      ゴール → 1C-2C(右側半円) → バック(右→左) → 3C-4C(左側半円) → ホーム(左→右) → ゴール

    半径の異なる左右コーナーにより、中山のような洋梨型を再現。

    戻り値: [(x, y), ...] n_points個の点 (1周分、走行方向順)
    """
    shape_info = VENUE_SHAPES.get(venue_name)
    if not shape_info:
        return []

    home = shape_info["home_straight"]
    back = shape_info["back_straight"]
    r12 = shape_info["corner_radius_12"]
    r34 = shape_info["corner_radius_34"]
    direction = shape_info["direction"]

    # 各パートの弧長
    arc_34 = math.pi * r34
    arc_12 = math.pi * r12
    total_len = home + arc_34 + back + arc_12

    # 点数配分
    n_home = max(2, round(n_points * home / total_len))
    n_c34 = max(2, round(n_points * arc_34 / total_len))
    n_back = max(2, round(n_points * back / total_len))
    n_c12 = max(2, n_points - n_home - n_c34 - n_back)

    # 基準座標
    home_left = -home / 2.0
    home_right = home / 2.0

    # 左コーナー(3C-4C側): 中心=(-home/2, r34)
    c34_cx, c34_cy = home_left, r34
    top_left_y = 2.0 * r34

    # 右コーナー(1C-2C側): 中心=(+home/2, r12)
    c12_cx, c12_cy = home_right, r12
    top_right_y = 2.0 * r12

    points = []

    if direction == "右":
        # 右回り(時計回り): path[0]=(home/2, 0) ゴール

        # Part 1: ホームストレッチ (右→左)
        for i in range(n_home):
            t = i / n_home
            points.append((home_right - t * home, 0.0))

        # Part 2: 4C→3C 左側半円 (底→頂, 時計回り)
        # 中心(c34_cx, c34_cy)から: 底θ=3π/2 → 頂θ=π/2 (θ減少)
        for i in range(n_c34):
            t = i / n_c34
            theta = (3 * math.pi / 2) - t * math.pi
            points.append((
                c34_cx + r34 * math.cos(theta),
                c34_cy + r34 * math.sin(theta),
            ))

        # Part 3: バックストレッチ (左→右)
        for i in range(n_back):
            t = i / n_back
            points.append((
                home_left + t * (home_right - home_left),
                top_left_y + t * (top_right_y - top_left_y),
            ))

        # Part 4: 2C→1C 右側半円 (頂→底, 時計回り)
        # 中心(c12_cx, c12_cy)から: 頂θ=π/2 → 底θ=-π/2 (θ減少)
        for i in range(n_c12):
            t = i / n_c12
            theta = (math.pi / 2) - t * math.pi
            points.append((
                c12_cx + r12 * math.cos(theta),
                c12_cy + r12 * math.sin(theta),
            ))

    else:
        # 左回り(反時計回り): path[0]=(home/2, 0) ゴール
        # ゴール → 1C(右コーナー底→頂) → バック(右→左) → 4C(左コーナー頂→底) → ホーム(左→右) → ゴール

        # Part 1: 1C→2C 右側半円 (底→頂, 反時計回り)
        # 中心(c12_cx, c12_cy)から: 底θ=-π/2 → 頂θ=π/2 (θ増加)
        for i in range(n_c12):
            t = i / n_c12
            theta = (-math.pi / 2) + t * math.pi
            points.append((
                c12_cx + r12 * math.cos(theta),
                c12_cy + r12 * math.sin(theta),
            ))

        # Part 2: バックストレッチ (右→左)
        for i in range(n_back):
            t = i / n_back
            points.append((
                home_right + t * (home_left - home_right),
                top_right_y + t * (top_left_y - top_right_y),
            ))

        # Part 3: 3C→4C 左側半円 (頂→底, 反時計回り)
        # 中心(c34_cx, c34_cy)から: 頂θ=π/2 → 底θ=3π/2 (θ増加)
        for i in range(n_c34):
            t = i / n_c34
            theta = (math.pi / 2) + t * math.pi
            points.append((
                c34_cx + r34 * math.cos(theta),
                c34_cy + r34 * math.sin(theta),
            ))

        # Part 4: ホームストレッチ (左→右)
        for i in range(n_home):
            t = i / n_home
            points.append((home_left + t * home, 0.0))

    return points


def _geometric_perimeter(venue_name: str) -> float:
    """VENUE_SHAPES から描画用の幾何学的周長を計算する。
    パスの点数配分もこの長さに基づくため、マッピングにはこちらを使用する。
    """
    shape = VENUE_SHAPES.get(venue_name)
    if not shape:
        return 1600.0  # fallback
    return (shape["home_straight"]
            + math.pi * shape["corner_radius_34"]
            + shape["back_straight"]
            + math.pi * shape["corner_radius_12"])


def _map_sections_to_path(
    sections: list[dict],
    track_path: list[tuple[float, float]],
    total_distance: int,
    perimeter: float,
) -> list[dict]:
    """区間情報をトラック座標にマッピングする。

    スタート位置はゴールから total_distance m 手前。
    各区間 (200m) をパス上の等距離区間に割り当てる。
    複数周回のコースではパス座標を繰り返す。

    戻り値: [{"section": {...}, "coords": [(x,y),...]}, ...]
    """
    n_points = len(track_path)
    result = []

    # ゴールは idx=0。スタートはゴールから距離 total_distance 手前。
    # パスは走行方向順なので、スタートの idx を求める。
    start_offset = (perimeter - (total_distance % perimeter)) % perimeter

    for sec in sections:
        start_m = sec["start_m"]
        end_m = sec["end_m"]

        # パス上の距離 (ゴール=0 基準、走行方向)
        sec_path_start = (start_offset + start_m) % perimeter
        sec_path_end = (start_offset + end_m) % perimeter

        # インデックスに変換
        idx_start = int((sec_path_start / perimeter) * n_points) % n_points
        idx_end = int((sec_path_end / perimeter) * n_points) % n_points

        # 座標を抽出
        if idx_start <= idx_end:
            coords = track_path[idx_start:idx_end + 1]
        else:
            # 周回をまたぐ場合
            coords = track_path[idx_start:] + track_path[:idx_end + 1]

        if len(coords) < 2:
            coords = [track_path[idx_start], track_path[(idx_start + 1) % n_points]]

        result.append({
            "section": sec,
            "coords": coords,
        })

    return result


def _section_difficulty(section: dict) -> float:
    """区間の厳しさスコアを 0.0 (楽) ~ 1.0 (厳しい) で返す。

    要素:
    - 勾配の絶対値 (上りは特に厳しい)
    - コーナーの曲率 (半径が小さいほど厳しい)
    """
    gradient = section.get("gradient", 0.0)
    corner_radius = section.get("corner_radius")
    sec_type = section.get("type", "straight")

    score = 0.0

    # 勾配スコア: 上りは厳しい、下りはやや楽だが急下りは足への負担
    abs_grad = abs(gradient)
    if gradient > 0:
        # 上り: 特に厳しい
        score += min(abs_grad / 2.5, 1.0) * 0.7  # 2.5%で最大0.7
    elif gradient < 0:
        # 下り: 少し負担
        score += min(abs_grad / 3.0, 0.3) * 0.3

    # コーナースコア: 半径が小さいほど厳しい
    if corner_radius is not None and corner_radius > 0:
        # 半径55m(小倉ダ)=厳しい, 170m(阪神外)=楽
        curvature_score = max(0, 1.0 - (corner_radius - 50) / 150)
        score += curvature_score * 0.3

    # uphill タイプは追加ボーナス
    if sec_type == "uphill":
        score = min(score + 0.2, 1.0)

    return min(score, 1.0)


def _difficulty_color(score: float) -> str:
    """厳しさスコアを色に変換する (緑→黄→赤)。

    0.0 = 緑 (#2d8a4e)
    0.5 = 黄 (#d4a017)
    1.0 = 赤 (#c62828)
    """
    if score <= 0.5:
        t = score / 0.5
        r = int(45 + t * (212 - 45))
        g = int(138 + t * (160 - 138))
        b = int(78 + t * (23 - 78))
    else:
        t = (score - 0.5) / 0.5
        r = int(212 + t * (198 - 212))
        g = int(160 - t * (160 - 40))
        b = int(23 + t * (40 - 23))
    return f"rgb({r},{g},{b})"


def _difficulty_color_fill(score: float) -> str:
    """塗りつぶし用の半透明色"""
    if score <= 0.5:
        t = score / 0.5
        r = int(45 + t * (212 - 45))
        g = int(138 + t * (160 - 138))
        b = int(78 + t * (23 - 78))
    else:
        t = (score - 0.5) / 0.5
        r = int(212 + t * (198 - 212))
        g = int(160 - t * (160 - 40))
        b = int(23 + t * (40 - 23))
    return f"rgba({r},{g},{b},0.4)"


def _identify_corners(sections: list[dict], total_distance: int) -> list[dict]:
    """連続するcorner区間をまとめてコーナー番号を割り当てる。

    戻り値: [{"corner_number": "1C", "start_m": 400, "end_m": 800, "mid_m": 600}, ...]
    """
    corners = []
    current_corner_start = None
    corner_count = 0

    for sec in sections:
        if sec["type"] == "corner":
            if current_corner_start is None:
                current_corner_start = sec["start_m"]
                corner_count += 1
            current_corner_end = sec["end_m"]
        else:
            if current_corner_start is not None:
                mid = (current_corner_start + current_corner_end) / 2
                corners.append({
                    "corner_number": f"{corner_count}C",
                    "start_m": current_corner_start,
                    "end_m": current_corner_end,
                    "mid_m": mid,
                })
                current_corner_start = None

    # 最後のコーナー
    if current_corner_start is not None:
        mid = (current_corner_start + current_corner_end) / 2
        corners.append({
            "corner_number": f"{corner_count}C",
            "start_m": current_corner_start,
            "end_m": current_corner_end,
            "mid_m": mid,
        })

    # コーナー番号を走行順でリナンバリング
    # course_structure.jsonの慣例に合わせる
    # turnsが2の場合: 3C, 4C (短距離) のように後ろのコーナーだけ
    # turnsが4の場合: 1C, 2C, 3C, 4C
    n_corners = len(corners)
    if n_corners <= 2:
        # 3C, 4Cとして扱う
        labels = [f"{3 + i}C" for i in range(n_corners)]
    elif n_corners <= 4:
        labels = [f"{i + 1}C" for i in range(n_corners)]
    else:
        # 6コーナー等 (2周以上)
        labels = [f"{i + 1}C" for i in range(n_corners)]

    for i, corner in enumerate(corners):
        corner["label"] = labels[i] if i < len(labels) else f"{i + 1}C"

    return corners


@st.cache_data(ttl=3600)
def load_course_structure():
    """course_structure.json を読み込む"""
    # ローカル開発用パス
    local_path = Path("C:/Users/okusa/Desktop/keiba-ai/config/course_structure.json")
    # デプロイ用: data/ディレクトリにもコピーしておく
    deploy_path = DATA_DIR / "course_structure.json"

    for p in [local_path, deploy_path]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def page_course_map():
    """コースマップページ — 競馬場を上から見た図で区間ごとの厳しさを色付き表示"""

    st.header("コースマップ")
    st.caption("競馬場のコース形状と各区間の厳しさを上空図で表示")

    course_data = load_course_structure()
    if not course_data:
        st.warning("コース構造データがありません。")
        return

    # コースキー一覧 (_metadata除外)
    all_keys = [k for k in course_data if k != "_metadata"]

    # 競馬場一覧
    venues_in_data = sorted(set(course_data[k]["venue"] for k in all_keys))

    # サイドバーでフィルタ
    with st.sidebar:
        st.subheader("コースマップ設定")
        selected_venue = st.selectbox("競馬場", venues_in_data, key="map_venue")

        # 選択した競馬場のコース一覧
        venue_courses = [k for k in all_keys if course_data[k]["venue"] == selected_venue]

        # 芝/ダートフィルタ
        surfaces = sorted(set(course_data[k]["surface"] for k in venue_courses))
        selected_surface = st.radio("馬場", surfaces, horizontal=True, key="map_surface")

        filtered = [k for k in venue_courses if course_data[k]["surface"] == selected_surface]
        filtered.sort(key=lambda k: course_data[k]["distance"])

        if not filtered:
            st.warning("該当コースがありません。")
            return

        selected_course_key = st.selectbox(
            "距離",
            filtered,
            format_func=lambda k: f"{course_data[k]['distance']}m" + (
                f" ({k.split('_')[-1]})" if len(k.split('_')) > 3 else ""
            ),
            key="map_distance",
        )

    if not selected_course_key:
        return

    course = course_data[selected_course_key]
    sections = course.get("sections", [])
    venue = course["venue"]
    distance = course["distance"]
    direction = course["direction"]
    straight_length = course.get("straight_length", 0)
    elevation_diff = course.get("elevation_diff", 0)
    hill_height = course.get("hill_height", 0)

    # コース情報表示
    info_cols = st.columns(5)
    with info_cols[0]:
        st.metric("距離", f"{distance}m")
    with info_cols[1]:
        st.metric("回り", direction)
    with info_cols[2]:
        st.metric("直線", f"{straight_length}m")
    with info_cols[3]:
        st.metric("高低差", f"{elevation_diff}m")
    with info_cols[4]:
        st.metric("坂高さ", f"{hill_height}m" if hill_height > 0 else "-")

    # 直線コース (新潟1000m) は特別扱い
    if direction == "直線":
        _render_straight_course(sections, distance, venue)
        return

    # トラック形状を生成
    shape_info = VENUE_SHAPES.get(venue)
    if not shape_info:
        st.warning(f"{venue} の形状データが未定義です。")
        return

    perimeter = _geometric_perimeter(venue)
    track_path = _generate_track_path(venue, n_points=1000)

    if not track_path:
        st.warning("トラック形状の生成に失敗しました。")
        return

    # 区間をトラック上にマッピング (幾何学的周長を使用)
    mapped = _map_sections_to_path(sections, track_path, distance, perimeter)

    # コーナー識別
    corners = _identify_corners(sections, distance)

    # --- Plotlyで描画 ---
    fig = go.Figure()

    track_width = 15.0

    # コース全体のアウトライン (薄いグレー)
    all_x = [p[0] for p in track_path] + [track_path[0][0]]
    all_y = [p[1] for p in track_path] + [track_path[0][1]]
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        mode="lines",
        line=dict(color="rgba(100,100,100,0.3)", width=track_width * 1.5),
        hoverinfo="skip",
        showlegend=False,
    ))

    # 各区間を色付き太線で描画
    type_labels = {
        "straight": "直線", "corner": "コーナー",
        "uphill": "急坂(上り)", "straight_uphill": "直線(上り)",
        "straight_downhill": "直線(下り)", "corner_uphill": "コーナー(上り)",
        "corner_downhill": "コーナー(下り)",
    }
    for m in mapped:
        sec = m["section"]
        coords = m["coords"]
        difficulty = _section_difficulty(sec)
        color = _difficulty_color(difficulty)

        type_label = type_labels.get(sec.get("type", ""), sec.get("type", ""))
        gradient = sec.get("gradient", 0.0)
        corner_radius = sec.get("corner_radius")
        notes = sec.get("notes", "")

        hover_parts = [
            f"<b>{sec['start_m']}m - {sec['end_m']}m</b>",
            f"区間タイプ: {type_label}",
            f"勾配: {gradient:+.1f}%",
        ]
        if corner_radius:
            hover_parts.append(f"コーナー半径: {corner_radius}m")
        hover_parts.append(f"厳しさ: {difficulty:.0%}")
        if notes:
            hover_parts.append(f"<i>{notes}</i>")

        fig.add_trace(go.Scatter(
            x=[c[0] for c in coords],
            y=[c[1] for c in coords],
            mode="lines",
            line=dict(color=color, width=track_width),
            hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
            showlegend=False,
        ))

    # ゴール位置マーカー
    goal_x, goal_y = track_path[0]
    fig.add_trace(go.Scatter(
        x=[goal_x], y=[goal_y],
        mode="markers+text",
        marker=dict(color="#ffffff", size=12, symbol="x",
                    line=dict(width=2, color="#ffffff")),
        text=["GOAL"],
        textposition="bottom center",
        textfont=dict(color="#ffffff", size=12, family="Arial Black"),
        hovertemplate="ゴール<extra></extra>",
        showlegend=False,
    ))

    # スタート位置マーカー
    start_offset_on_path = (perimeter - (distance % perimeter)) % perimeter
    start_idx = int((start_offset_on_path / perimeter) * len(track_path)) % len(track_path)
    start_x, start_y = track_path[start_idx]
    fig.add_trace(go.Scatter(
        x=[start_x], y=[start_y],
        mode="markers+text",
        marker=dict(color="#64b5f6", size=10,
                    symbol="triangle-right" if direction == "右" else "triangle-left",
                    line=dict(width=1, color="#64b5f6")),
        text=["START"],
        textposition="top center",
        textfont=dict(color="#64b5f6", size=11, family="Arial Black"),
        hovertemplate=f"スタート ({distance}m)<extra></extra>",
        showlegend=False,
    ))

    # コーナー番号ラベル (コース内側にオフセット)
    for corner in corners:
        mid_m = corner["mid_m"]
        mid_offset = (start_offset_on_path + mid_m) % perimeter
        mid_idx = int((mid_offset / perimeter) * len(track_path)) % len(track_path)
        cx, cy = track_path[mid_idx]

        # コース中心方向に向かってオフセット（ラベルをコース内側に配置）
        _r34 = shape_info.get("corner_radius_34", 100)
        _r12 = shape_info.get("corner_radius_12", 100)
        track_center_x = 0.0
        track_center_y = (_r34 + _r12) / 2.0
        dx_lbl = cx - track_center_x
        dy_lbl = cy - track_center_y
        dist_to_center = math.sqrt(dx_lbl * dx_lbl + dy_lbl * dy_lbl)
        if dist_to_center > 0:
            label_x = cx - (dx_lbl / dist_to_center) * 40
            label_y = cy - (dy_lbl / dist_to_center) * 40
        else:
            label_x, label_y = cx, cy

        fig.add_annotation(
            x=label_x, y=label_y,
            text=f"<b>{corner['label']}</b>",
            showarrow=False,
            font=dict(color="#ffa726", size=13, family="Arial Black"),
        )

    # 走行方向の矢印
    for pos in [0.15, 0.4, 0.65, 0.85]:
        idx = int(pos * len(track_path)) % len(track_path)
        next_idx = (idx + 5) % len(track_path)
        fig.add_annotation(
            x=track_path[next_idx][0], y=track_path[next_idx][1],
            ax=track_path[idx][0], ay=track_path[idx][1],
            arrowhead=3, arrowsize=1.2, arrowwidth=2,
            arrowcolor="rgba(255,255,255,0.4)",
            showarrow=True, text="",
        )

    # レイアウト設定 (ダークモード、横長楕円)
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        title=dict(
            text=f"{venue} {selected_surface}{distance}m ({direction}回り)",
            font=dict(size=18, color="#e0e0e0"),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="x", scaleratio=1,
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=500,
        hoverlabel=dict(
            bgcolor="#1e1e2e", font_size=13,
            font_color="white", bordercolor="#555",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- 凡例 ---
    st.markdown("#### 色の凡例")
    legend_cols = st.columns(5)
    legend_items = [
        ("平坦直線", 0.0),
        ("緩いコーナー/微傾斜", 0.2),
        ("きついコーナー/中傾斜", 0.5),
        ("急坂/小半径コーナー", 0.75),
        ("最急坂区間", 1.0),
    ]
    for col, (label, score) in zip(legend_cols, legend_items):
        color = _difficulty_color(score)
        with col:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='width:40px;height:16px;background:{color};margin:0 auto;border-radius:3px;'></div>"
                f"<span style='color:#b0b0b0;font-size:0.8em;'>{label}</span></div>",
                unsafe_allow_html=True,
            )

    # --- 区間詳細テーブル ---
    with st.expander("区間詳細データ", expanded=False):
        detail_rows = []
        for sec in sections:
            difficulty = _section_difficulty(sec)
            type_label = STRUCTURE_LABELS.get(sec.get("type", ""), sec.get("type", ""))
            detail_rows.append({
                "区間": f"{sec['start_m']}m - {sec['end_m']}m",
                "タイプ": type_label,
                "勾配(%)": sec.get("gradient", 0.0),
                "コーナー半径(m)": sec.get("corner_radius") or "-",
                "厳しさ": f"{difficulty:.0%}",
                "備考": sec.get("notes", ""),
            })

        df_detail = pd.DataFrame(detail_rows)

        def style_map_detail(row):
            styles = [""] * len(row)
            cols = list(row.index)
            if "勾配(%)" in cols:
                idx = cols.index("勾配(%)")
                try:
                    g = float(row["勾配(%)"])
                    if g > 0.5:
                        styles[idx] = f"color: {TEXT_RED}; font-weight: bold"
                    elif g > 0:
                        styles[idx] = f"color: {TEXT_ORANGE}"
                    elif g < -0.3:
                        styles[idx] = f"color: #42a5f5"
                    elif g < 0:
                        styles[idx] = f"color: #64b5f6"
                except (ValueError, TypeError):
                    pass
            if "厳しさ" in cols:
                idx = cols.index("厳しさ")
                try:
                    v = float(row["厳しさ"].replace("%", "")) / 100
                    if v >= 0.7:
                        styles[idx] = f"background-color: {BG_RED_STRONG}; color: {TEXT_RED}; font-weight: bold"
                    elif v >= 0.4:
                        styles[idx] = f"background-color: {BG_ORANGE_LIGHT}; color: {TEXT_ORANGE}"
                    elif v > 0:
                        styles[idx] = f"color: {TEXT_YELLOW}"
                    else:
                        styles[idx] = f"color: {TEXT_GREEN_MED}"
                except (ValueError, TypeError):
                    pass
            return styles

        st.dataframe(
            df_detail.style.apply(style_map_detail, axis=1),
            use_container_width=True,
            hide_index=True,
        )


def _render_straight_course(sections: list[dict], distance: int, venue: str):
    """直線コース (新潟芝1000m) を水平バーで描画する"""
    fig = go.Figure()

    total_len = distance
    bar_height = 30

    for sec in sections:
        difficulty = _section_difficulty(sec)
        color = _difficulty_color(difficulty)
        fill_color = _difficulty_color_fill(difficulty)

        x0 = sec["start_m"]
        x1 = sec["end_m"]

        type_label = STRUCTURE_LABELS.get(sec.get("type", ""), sec.get("type", ""))
        gradient = sec.get("gradient", 0.0)
        notes = sec.get("notes", "")

        hover_parts = [
            f"<b>{x0}m - {x1}m</b>",
            f"区間タイプ: {type_label}",
            f"勾配: {gradient:+.1f}%",
            f"厳しさ: {difficulty:.0%}",
        ]
        if notes:
            hover_parts.append(f"<i>{notes}</i>")

        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[-bar_height / 2, -bar_height / 2, bar_height / 2, bar_height / 2, -bar_height / 2],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=color, width=2),
            hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
            showlegend=False,
        ))

        # 距離ラベル
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0,
            text=f"{x0}-{x1}m",
            showarrow=False,
            font=dict(color="white", size=10),
        )

    # スタート・ゴール
    fig.add_annotation(x=0, y=-bar_height, text="START", showarrow=False,
                       font=dict(color="#64b5f6", size=12, family="Arial Black"))
    fig.add_annotation(x=total_len, y=-bar_height, text="GOAL", showarrow=False,
                       font=dict(color="#ffffff", size=12, family="Arial Black"))

    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        title=dict(
            text=f"{venue} 芝{distance}m (直線)",
            font=dict(size=18, color="#e0e0e0"),
        ),
        xaxis=dict(showgrid=False, zeroline=False, title="距離 (m)"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-50, 50]),
        margin=dict(l=20, r=20, t=50, b=40),
        height=250,
        hoverlabel=dict(bgcolor="#1e1e2e", font_size=13, font_color="white", bordercolor="#555"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# メインUI
# ============================================================

def main():
    # session_state 初期化
    if "page" not in st.session_state:
        st.session_state["page"] = "予想"
    if "horse_name" not in st.session_state:
        st.session_state["horse_name"] = ""

    # query parameter による馬詳細ページ遷移
    qp = st.query_params
    if qp.get("page") == "horse" and qp.get("name"):
        st.session_state["page"] = "馬詳細"
        st.session_state["horse_name"] = qp["name"]

    PAGE_OPTIONS = ["予想", "レート一覧", "馬詳細", "コース分析", "コースマップ"]

    # サイドバー - ページ選択
    with st.sidebar:
        st.title("keiba-ai")
        current_idx = PAGE_OPTIONS.index(st.session_state["page"]) if st.session_state["page"] in PAGE_OPTIONS else 0
        page = st.radio(
            "ページ",
            PAGE_OPTIONS,
            index=current_idx,
            key="page_radio",
        )
        # ラジオ操作時に session_state を同期
        if page != st.session_state["page"]:
            st.session_state["page"] = page
            # ラジオで馬詳細以外に切り替えたら馬名リセット
            if page != "馬詳細":
                st.session_state["horse_name"] = ""
        st.markdown("---")

    if st.session_state["page"] == "予想":
        page_predictions()
    elif st.session_state["page"] == "レート一覧":
        page_ratings()
    elif st.session_state["page"] == "馬詳細":
        page_horse_detail()
    elif st.session_state["page"] == "コース分析":
        page_course_analysis()
    elif st.session_state["page"] == "コースマップ":
        page_course_map()


# ============================================================
# レートページ（v2: 芝/ダート分離、クラス補正付き絶対レート）
# ============================================================

def _is_v2_ratings(ratings_data):
    """ratings.json が v2 形式かを判定"""
    return ratings_data.get("version", 1) >= 2


def _get_surface_data(hdata, surface_key):
    """v2形式から指定surfaceのデータを取得。v1互換も保持。"""
    if surface_key in hdata:
        return hdata[surface_key]
    # v1互換: cr, h がトップレベルにある
    if "cr" in hdata and "h" in hdata:
        return {"cr": hdata["cr"], "h": hdata["h"]}
    return None


def _best_rating(hdata):
    """馬のベストレート（芝/ダートの高い方）を返す"""
    best = 0.0
    for key in ("turf", "dirt"):
        sd = hdata.get(key)
        if sd:
            cr = sd.get("cr", 0.0)
            if cr > best:
                best = cr
    # v1 fallback
    if "cr" in hdata:
        return hdata["cr"]
    return best


def _rating_color_style(v):
    """レート値に応じたセルスタイルを返す（1000mあたり秒差スケール: 正=速い=強い）"""
    if v >= 2.5:
        return f"background-color: {BG_GREEN_STRONG}; color: {TEXT_GREEN}; font-weight: bold"
    elif v >= 1.5:
        return f"background-color: {BG_GREEN_MEDIUM}; color: {TEXT_GREEN_MED}; font-weight: bold"
    elif v >= 0.75:
        return f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
    elif v >= 0.0:
        return f"color: {TEXT_LIGHT}"
    elif v >= -1.0:
        return f"color: {TEXT_MUTED}"
    else:
        return f"background-color: {BG_RED_LIGHT}; color: {TEXT_RED}"


def page_ratings():
    """レート一覧ページ"""
    ratings_data = load_ratings_data()
    if not ratings_data:
        st.warning("レートデータがありません。export_ratings.py を実行してください。")
        return

    ratings = ratings_data.get("ratings", {})
    exported_at = ratings_data.get("exported_at", "")
    is_v2 = _is_v2_ratings(ratings_data)

    st.header("レート一覧")
    if exported_at:
        try:
            exp_dt = datetime.fromisoformat(exported_at)
            st.caption(f"更新: {exp_dt.strftime('%Y-%m-%d %H:%M')}")
        except ValueError:
            pass

    with st.sidebar:
        st.markdown("### フィルタ")
        if is_v2:
            surface_filter = st.selectbox("コース", ["芝", "ダート"], key="rating_surface")
        else:
            surface_filter = st.selectbox("コース", ["全て", "芝", "ダート"], key="rating_surface")
        dist_options = ["全て", "短距離(~1400m)", "マイル(1401~1800m)", "中距離(1801~2200m)", "長距離(2201m~)"]
        dist_filter = st.selectbox("距離帯", dist_options)
        search_query = st.text_input("馬名検索", "")

    # v2: surface_key
    if is_v2:
        surface_key = "turf" if surface_filter == "芝" else "dirt"
    else:
        surface_key = None

    # データ構築
    rows = []
    for hid, hdata in ratings.items():
        name = hdata.get("n", "")

        if is_v2:
            sdata = hdata.get(surface_key)
            if not sdata:
                continue
            cr = sdata.get("cr", 0.0)
            history = sdata.get("h", [])
        else:
            cr = hdata.get("cr", 0.0)
            history = hdata.get("h", [])

        if not history:
            continue

        latest = history[-1]
        latest_date = latest.get("d", "")
        latest_fo = latest.get("fo")
        latest_odds = latest.get("o")
        latest_ct = latest.get("ct", "")
        latest_dist = latest.get("dist", 0)
        latest_rn = latest.get("rn", "")
        latest_venue = latest.get("v", "")
        expected_rating = sdata.get("er") if is_v2 and sdata else None

        # v1互換コースフィルタ
        if not is_v2:
            if surface_filter == "芝" and "芝" not in latest_ct:
                continue
            if surface_filter == "ダート" and "ダ" not in latest_ct and "ダート" not in latest_ct:
                continue

        # 距離帯フィルタ
        if dist_filter == "短距離(~1400m)" and latest_dist > 1400:
            continue
        if dist_filter == "マイル(1401~1800m)" and (latest_dist <= 1400 or latest_dist > 1800):
            continue
        if dist_filter == "中距離(1801~2200m)" and (latest_dist <= 1800 or latest_dist > 2200):
            continue
        if dist_filter == "長距離(2201m~)" and latest_dist <= 2200:
            continue

        # 馬名検索
        if search_query and search_query not in name:
            continue

        rows.append({
            "horse_id": hid,
            "馬名": name,
            "レート": cr,
            "期待": expected_rating if expected_rating is not None else cr,
            "直近レース": f"{latest_venue} {latest_rn}" if latest_rn else latest_date,
            "日付": latest_date,
            "着順": latest_fo if latest_fo else "-",
            "オッズ": latest_odds if latest_odds else "-",
            "距離": latest_dist,
        })

    if not rows:
        st.info("該当する馬がありません。")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("期待", ascending=False).reset_index(drop=True)
    df.insert(0, "順位", range(1, len(df) + 1))

    st.markdown(f"**{len(df)}頭** (馬名クリックで詳細を別タブで表示)")

    # ページネーション
    PER_PAGE = 50
    total_pages = max(1, (len(df) + PER_PAGE - 1) // PER_PAGE)
    if "ratings_page" not in st.session_state:
        st.session_state["ratings_page"] = 0
    current_page = st.session_state["ratings_page"]
    if current_page >= total_pages:
        current_page = 0
        st.session_state["ratings_page"] = 0

    if total_pages > 1:
        page_cols = st.columns([1, 3, 1])
        with page_cols[0]:
            if st.button("<< 前", key="ratings_prev", disabled=(current_page == 0)):
                st.session_state["ratings_page"] = current_page - 1
                st.rerun()
        with page_cols[1]:
            st.markdown(
                f"<div style='text-align:center; color:{TEXT_MUTED};'>"
                f"{current_page + 1} / {total_pages} ページ</div>",
                unsafe_allow_html=True,
            )
        with page_cols[2]:
            if st.button("次 >>", key="ratings_next", disabled=(current_page >= total_pages - 1)):
                st.session_state["ratings_page"] = current_page + 1
                st.rerun()

    start_idx = current_page * PER_PAGE
    end_idx = min(start_idx + PER_PAGE, len(df))
    page_df = df.iloc[start_idx:end_idx]

    # HTMLテーブルで表示（コンパクト）
    import urllib.parse

    html_rows = []
    for _, row in page_df.iterrows():
        # 馬名リンク（別タブで馬詳細ページを開く）
        encoded_name = urllib.parse.quote(row["馬名"])
        horse_link = f'<a href="?page=horse&name={encoded_name}" target="_blank" style="color:#64b5f6; text-decoration:none; font-weight:bold;">{row["馬名"]}</a>'

        # 期待レートスタイル
        er = row["期待"]
        er_s = _rating_color_style(er)
        er_cell = f'<span style="{er_s}">{er:+.2f}</span>'

        # 直近レートスタイル
        rate_s = _rating_color_style(row["レート"])
        rate_cell = f'<span style="{rate_s}">{row["レート"]:+.2f}</span>'

        # 着順スタイル
        fo = row["着順"]
        fo_s = f"color:{TEXT_LIGHT}"
        if fo != "-":
            try:
                fov = int(fo)
                if fov == 1:
                    fo_s = f"background-color:{BG_GREEN_STRONG}; color:#ffffff; font-weight:bold; padding:1px 5px; border-radius:3px"
                elif fov <= 3:
                    fo_s = f"color:{TEXT_GREEN_MED}"
            except (ValueError, TypeError):
                pass
        fo_cell = f'<span style="{fo_s}">{fo}</span>'

        odds = row["オッズ"]

        html_rows.append(
            f"<tr>"
            f'<td style="color:{TEXT_MUTED}; text-align:right; padding-right:8px;">{row["順位"]}</td>'
            f'<td>{horse_link}</td>'
            f'<td style="text-align:right;">{er_cell}</td>'
            f'<td style="text-align:right;">{rate_cell}</td>'
            f'<td style="color:{TEXT_LIGHT};">{row["直近レース"]}</td>'
            f'<td style="color:{TEXT_MUTED};">{row["日付"]}</td>'
            f'<td style="text-align:center;">{fo_cell}</td>'
            f'<td style="color:{TEXT_LIGHT}; text-align:right;">{odds}</td>'
            f"</tr>"
        )

    table_html = f"""
    <style>
    .ratings-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9em;
    }}
    .ratings-table th {{
        color: {TEXT_MUTED};
        font-weight: bold;
        text-align: left;
        padding: 4px 6px;
        border-bottom: 1px solid #444;
    }}
    .ratings-table td {{
        padding: 3px 6px;
        border-bottom: 1px solid #333;
        white-space: nowrap;
    }}
    .ratings-table tr:hover {{
        background-color: #2a2a2a;
    }}
    .ratings-table a:hover {{
        color: #90caf9 !important;
        text-decoration: underline !important;
    }}
    </style>
    <table class="ratings-table">
    <thead>
    <tr>
        <th style="text-align:right; padding-right:8px;">#</th>
        <th>馬名</th>
        <th style="text-align:right;">期待</th>
        <th style="text-align:right;">直近</th>
        <th>直近レース</th>
        <th>日付</th>
        <th style="text-align:center;">着順</th>
        <th style="text-align:right;">オッズ</th>
    </tr>
    </thead>
    <tbody>
    {"".join(html_rows)}
    </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# 馬詳細ページ（v2: 芝/ダート分離対応）
# ============================================================

def page_horse_detail():
    """馬詳細ページ"""
    ratings_data = load_ratings_data()
    if not ratings_data:
        st.warning("レートデータがありません。export_ratings.py を実行してください。")
        return

    ratings = ratings_data.get("ratings", {})
    is_v2 = _is_v2_ratings(ratings_data)

    # 馬名リスト（ベストレートが高い順）
    horse_list = []
    for hid, hdata in ratings.items():
        name = hdata.get("n", "")
        cr = _best_rating(hdata) if is_v2 else hdata.get("cr", 0.0)
        if name:
            horse_list.append((hid, name, cr))
    horse_list.sort(key=lambda x: x[2], reverse=True)

    # session_state から馬名が指定されている場合、検索欄に反映
    preset_horse = st.session_state.get("horse_name", "")

    with st.sidebar:
        search = st.text_input(
            "馬名で検索",
            value=preset_horse,
            key="horse_search",
        )
        if search:
            filtered = [(hid, n, cr) for hid, n, cr in horse_list if search in n]
        else:
            filtered = horse_list[:200]

        if not filtered:
            st.info("該当する馬がいません。")
            return

        options = {f"{n} ({cr:+.2f})": hid for hid, n, cr in filtered}

        # session_state から馬名指定がある場合、その馬を初期選択
        default_idx = 0
        if preset_horse:
            for i, label in enumerate(options.keys()):
                if label.startswith(preset_horse + " "):
                    default_idx = i
                    break
            # 使用後にリセット
            st.session_state["horse_name"] = ""

        selected_label = st.selectbox(
            "馬を選択",
            list(options.keys()),
            index=default_idx,
        )
        selected_hid = options[selected_label]

    hdata = ratings[selected_hid]
    horse_name = hdata.get("n", "")

    st.header(f"{horse_name}")

    if is_v2:
        turf_data = hdata.get("turf")
        dirt_data = hdata.get("dirt")

        # 現在レート表示
        cols = st.columns(4)
        with cols[0]:
            if turf_data:
                er = turf_data.get("er", turf_data["cr"])
                st.metric("芝 期待レート", f"{er:+.2f}")
            else:
                st.metric("芝 期待レート", "-")
        with cols[1]:
            if turf_data:
                st.metric("芝 直近レート", f"{turf_data['cr']:+.2f}")
            else:
                st.metric("芝 直近レート", "-")
        with cols[2]:
            if dirt_data:
                er = dirt_data.get("er", dirt_data["cr"])
                st.metric("ダート 期待レート", f"{er:+.2f}")
            else:
                st.metric("ダート 期待レート", "-")
        with cols[3]:
            if dirt_data:
                st.metric("ダート 直近レート", f"{dirt_data['cr']:+.2f}")
            else:
                st.metric("ダート 直近レート", "-")

        st.markdown("---")

        # 芝レート推移
        if turf_data and len(turf_data.get("h", [])) >= 2:
            st.subheader("芝レート推移")
            chart_data = pd.DataFrame([
                {"日付": rh["d"], "レート": rh["r"]}
                for rh in turf_data["h"]
            ])
            chart_data["日付"] = pd.to_datetime(chart_data["日付"])
            chart_data = chart_data.set_index("日付")
            st.line_chart(chart_data, y="レート", color="#4caf50")

        # ダートレート推移
        if dirt_data and len(dirt_data.get("h", [])) >= 2:
            st.subheader("ダートレート推移")
            chart_data = pd.DataFrame([
                {"日付": rh["d"], "レート": rh["r"]}
                for rh in dirt_data["h"]
            ])
            chart_data["日付"] = pd.to_datetime(chart_data["日付"])
            chart_data = chart_data.set_index("日付")
            st.line_chart(chart_data, y="レート", color="#ff9800")

        # 過去レース一覧（芝+ダート統合、日付降順）
        all_history = []
        if turf_data:
            all_history.extend(turf_data.get("h", []))
        if dirt_data:
            all_history.extend(dirt_data.get("h", []))
        all_history.sort(key=lambda x: x.get("d", ""), reverse=True)

    else:
        # v1互換
        current_rating = hdata.get("cr", 0.0)
        history = hdata.get("h", [])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("現在レート", f"{current_rating:.2f}")
        with col2:
            if len(history) >= 2:
                prev = history[-2]["r"]
                diff = current_rating - prev
                st.metric("前走からの変化", f"{diff:+.2f}")
            else:
                st.metric("前走からの変化", "-")
        with col3:
            st.metric("出走数(2024~)", f"{len(history)}")

        st.markdown("---")

        if len(history) >= 2:
            st.subheader("レート推移")
            chart_data = pd.DataFrame([
                {"日付": rh["d"], "レート": rh["r"]}
                for rh in history
            ])
            chart_data["日付"] = pd.to_datetime(chart_data["日付"])
            chart_data = chart_data.set_index("日付")
            st.line_chart(chart_data, y="レート", color="#4caf50")

        all_history = list(reversed(history))

    # 過去レース一覧
    st.subheader("過去レース一覧")
    race_rows = []
    for rh in all_history:
        fo = rh.get("fo")
        odds = rh.get("o")
        race_rows.append({
            "日付": rh.get("d", ""),
            "レース名": rh.get("rn", ""),
            "場所": rh.get("v", ""),
            "コース": f"{rh.get('ct', '')} {rh.get('dist', '')}m",
            "馬場": rh.get("tc", ""),
            "着順": fo if fo else "-",
            "オッズ": f"{odds:.1f}" if odds else "-",
            "レート": f"{rh.get('r', 0):+.2f}",
        })

    if race_rows:
        rdf = pd.DataFrame(race_rows)

        def highlight_horse_race_row(row):
            styles = [""] * len(row)
            cols = list(row.index)
            if "着順" in cols:
                idx = cols.index("着順")
                fo = row["着順"]
                if fo != "-":
                    try:
                        fov = int(fo)
                        if fov == 1:
                            styles[idx] = f"background-color: {BG_GREEN_STRONG}; color: #ffffff; font-weight: bold"
                        elif fov <= 3:
                            styles[idx] = f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
                        elif fov >= 10:
                            styles[idx] = f"color: {TEXT_MUTED}"
                    except (ValueError, TypeError):
                        pass
            if "レート" in cols:
                idx = cols.index("レート")
                try:
                    v = float(row["レート"])
                    styles[idx] = _rating_color_style(v)
                except (ValueError, TypeError):
                    pass
            return styles

        styled = rdf.style.apply(highlight_horse_race_row, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            height=min(len(rdf) * 38 + 40, 600),
        )
    else:
        st.info("レース履歴がありません。")

    # フッター
    st.markdown("---")
    st.caption(
        "keiba-ai 予想ページ | "
        "レートはクラス補正付き絶対評価値です（G1=80-100, OP=55-70, 未勝利=20-35）。"
    )


# ============================================================
# 予想ページ（既存）
# ============================================================

def page_predictions():
    """予想ページ（既存の予測表示）"""
    # サイドバー
    with st.sidebar:
        files = load_prediction_files()
        if not files:
            st.warning("予測データがありません")
            st.markdown(
                "data/ ディレクトリに予測JSONが配置されていません。\n\n"
                "`export_predictions.py` を実行してデータを生成してください。"
            )
            st.stop()

        selected_file = st.selectbox(
            "開催日",
            files,
            format_func=lambda x: x["label"],
        )

        st.markdown("---")

        # 詳細表示切替
        show_detail = st.checkbox("詳細情報を表示", value=False)

    # データ読み込み
    data = load_prediction_data(str(selected_file["path"]))
    race_date = data.get("race_date", "")
    exported_at = data.get("exported_at", "")
    venues_data = data.get("venues", {})

    if not venues_data:
        st.warning(f"{race_date} のレースデータがありません")
        st.stop()

    # ヘッダー
    st.header(f"予想: {race_date}")
    if exported_at:
        try:
            exp_dt = datetime.fromisoformat(exported_at)
            st.markdown(
                f"<div class='update-info'>更新: {exp_dt.strftime('%Y-%m-%d %H:%M')}</div>",
                unsafe_allow_html=True,
            )
        except ValueError:
            pass

    # 凡例（AI評価ランクの説明）
    with st.expander("AI評価ランクの見方", expanded=False):
        st.markdown("""
| ランク | 意味 | 目安 |
|:---:|:---|:---|
| **S** | 最高評価 | 過去データで高い回収率実績 |
| **A** | 高評価 | プラス回収が見込める |
| **B** | やや高評価 | 平均以上の期待値 |
| **C** | 普通 | 市場評価通り |
| **C-** | やや低評価 | 平均をやや下回る |
| **D** | 低評価 | 回収率が低い傾向 |
| **E** | かなり低評価 | 大幅な割引が必要 |
| **F** | 最低評価 | 見送り推奨 |

- **推奨「買」**: AI評価が高く、買い目として推奨
- **推奨「避」**: AI評価が低く、見送りを推奨
- **勝率・複勝率**: AIが算出した予測確率（オッズ・過去成績ベース）
        """)

    # 注目レースランキング
    all_races = []
    ALL_TIER_KEYS = ["130%", "120%", "110%", "100%", "90%", "80%", "70%", "60%", "50%", "40%"]
    for venue_name, venue_data in venues_data.items():
        for race in venue_data.get("races", []):
            horses = race.get("horses", [])

            # 各馬のティアを集計
            tier_counts = {k: 0 for k in ALL_TIER_KEYS}
            for h in horses:
                t = h.get("and_filter_tier")
                if t and t in tier_counts:
                    tier_counts[t] += 1

            # スコア
            highlight_score = (
                tier_counts["130%"] * 10
                + tier_counts["120%"] * 5
                + tier_counts["110%"] * 2
                + tier_counts["100%"] * 1
            )

            # Top3の馬名（高評価順）
            tier_horses = [
                (h, INTERNAL_TIER_SORT.get(h.get("and_filter_tier"), 5))
                for h in horses
            ]
            tier_horses.sort(key=lambda x: (x[1], x[0].get("rank", 999)))
            top3_parts = []
            for h, _ in tier_horses[:3]:
                if not h.get("and_filter_tier"):
                    continue
                wp = h.get("win_prob")
                wp_str = f" {wp * 100:.0f}%" if wp else ""
                label = _tier_to_label(h.get("and_filter_tier"))
                top3_parts.append(
                    f"{h['horse_number']}{h['horse_name']}"
                    f"({label}{wp_str})"
                )
            top3_names = ", ".join(top3_parts)

            # 評価サマリー
            eval_summary_parts = []
            for tk in ALL_TIER_KEYS:
                if tier_counts[tk] > 0:
                    label = _tier_to_label(tk)
                    eval_summary_parts.append(f"{label}:{tier_counts[tk]}")
            eval_summary = " ".join(eval_summary_parts) if eval_summary_parts else "-"

            # 低評価注意マーク
            n_low = tier_counts["40%"] + tier_counts["50%"]
            alert_mark = ""
            if n_low > 0:
                alert_mark = f" [注意{n_low}]"

            all_races.append({
                "venue": venue_name,
                "race_number": race["race_number"],
                "race_name": race["race_name"],
                "course": race["course"],
                "head_count": race.get("head_count", len(horses)),
                "highlight_score": highlight_score,
                "eval_summary": eval_summary,
                "top3_names": top3_names,
                "n_high": tier_counts["130%"] + tier_counts["120%"],
                "n_low": n_low,
                "alert_mark": alert_mark,
                "race_data": race,
            })

    all_races.sort(key=lambda x: x["highlight_score"], reverse=True)

    # 注目レーストップ5
    st.markdown("### 注目レース（AI高評価馬が多い順）")
    ranking_rows = []
    for i, r in enumerate(all_races[:5]):
        ranking_rows.append({
            "順位": f"{i + 1}.",
            "場所": r["venue"],
            "R": f"{r['race_number']}R",
            "レース名": r["race_name"] + r.get("alert_mark", ""),
            "コース": r["course"],
            "頭数": r["head_count"],
            "評価分布": r["eval_summary"],
            "注目馬": r["top3_names"],
        })

    if ranking_rows:
        rk_df = pd.DataFrame(ranking_rows)
        st.dataframe(
            rk_df,
            use_container_width=True,
            hide_index=True,
            height=min(len(rk_df) * 38 + 40, 300),
        )

    st.markdown("---")

    # 会場タブ
    venue_names = list(venues_data.keys())
    venue_tabs = st.tabs(venue_names)

    for venue_tab, venue_name in zip(venue_tabs, venue_names):
        with venue_tab:
            venue_races = venues_data[venue_name].get("races", [])

            for race in venue_races:
                race_num = race["race_number"]
                race_name = race["race_name"]
                course = race["course"]
                track_cond = race.get("track_condition", "")
                horses = race.get("horses", [])

                # 評価別カウント
                tier_counts = {}
                for h in horses:
                    t = h.get("and_filter_tier") or None
                    label = _tier_to_label(t) if t else "-"
                    tier_counts[label] = tier_counts.get(label, 0) + 1

                eval_str_parts = []
                for lbl in ["S", "A", "B", "C", "C-", "D", "D-", "E", "E-", "F"]:
                    if tier_counts.get(lbl, 0) > 0:
                        eval_str_parts.append(f"{lbl}:{tier_counts[lbl]}")
                eval_str = " ".join(eval_str_parts) if eval_str_parts else ""

                has_high = tier_counts.get("S", 0) + tier_counts.get("A", 0) > 0
                has_low = tier_counts.get("E-", 0) + tier_counts.get("F", 0) > 0

                header = (
                    f"{race_num}R **{race_name}** "
                    f"({course} {track_cond}) "
                    f"{eval_str}"
                )

                with st.expander(header, expanded=(has_high or has_low)):
                    # レース情報
                    info_cols = st.columns([2, 3])
                    with info_cols[0]:
                        hc = race.get("head_count", len(horses))
                        st.caption(
                            f"場所: {venue_name} | コース: {course} | "
                            f"馬場: {track_cond or '不明'} | 頭数: {hc}"
                        )

                    with info_cols[1]:
                        pace_data = race.get("pace_prediction")
                        render_pace_prediction(pace_data)

                    # 出走馬テーブル
                    if not horses:
                        st.warning("馬データなし")
                        continue

                    # 芝レース判定（馬場適性表示用）
                    is_turf = course.startswith("芝") if course else False

                    table_rows = []
                    for h in horses:
                        tier_label = _tier_to_label(h.get("and_filter_tier"))

                        # 馬場適性: 芝レースのみ表示
                        if is_turf:
                            cushion_val = h.get("cushion_roi", "-") or "-"
                        else:
                            cushion_val = "-"

                        # 勝率・複勝率
                        wp = h.get("win_prob")
                        pp = h.get("place_prob")
                        win_prob_str = f"{wp * 100:.1f}%" if wp else "-"
                        place_prob_str = f"{pp * 100:.1f}%" if pp else "-"

                        # 含水率×血統
                        moisture_blood_val = h.get("moisture_blood_roi", "-") or "-"

                        # 前走比較（旧: 条件バイアス）
                        context_bias_raw = h.get("context_bias", "-") or "-"
                        context_bias_display = _format_context_bias(context_bias_raw)

                        # シグナル簡略化
                        signal_display = _simplify_signal_tags(h.get("signal_tags", ""))

                        row_data = {
                            "順位": h.get("rank", 0),
                            "推奨": h.get("recommendation", ""),
                            "番": h.get("horse_number", ""),
                            "馬名": h.get("horse_name", ""),
                            "AI評価": tier_label,
                            "勝率": win_prob_str,
                            "複勝率": place_prob_str,
                            "前走比較": context_bias_display,
                            "馬場適性": cushion_val,
                            "馬場×血統": moisture_blood_val,
                            "血統適性": h.get("blood_roi", "-") or "-",
                            "父": h.get("sire_name", ""),
                            "脚質": h.get("running_style", ""),
                            "騎手": h.get("jockey_name", ""),
                            "オッズ": h.get("odds") or "-",
                            "材料": signal_display,
                        }

                        # 詳細モードの場合、追加列を表示
                        if show_detail:
                            plwp = h.get("pl_win_prob")
                            owp = h.get("odds_win_prob")
                            row_data["参考勝率1"] = f"{plwp * 100:.1f}%" if plwp else "-"
                            row_data["参考勝率2"] = f"{owp * 100:.1f}%" if owp else "-"

                        table_rows.append(row_data)

                    tdf = pd.DataFrame(table_rows)

                    # AI評価順にソート
                    tdf["_tier_sort"] = tdf["AI評価"].map(TIER_SORT_ORDER).fillna(5)
                    tdf = (tdf
                           .sort_values(["_tier_sort", "順位"])
                           .drop(columns="_tier_sort")
                           .reset_index(drop=True))

                    # 表示列の定義
                    display_cols = [
                        "順位", "推奨", "番", "馬名",
                        "AI評価", "勝率", "複勝率", "オッズ",
                        "前走比較",
                        "馬場適性", "馬場×血統", "血統適性",
                        "父", "脚質", "騎手", "材料",
                    ]
                    if show_detail:
                        # 詳細モード: 参考勝率も表示
                        display_cols = [
                            "順位", "推奨", "番", "馬名",
                            "AI評価", "勝率", "複勝率", "オッズ",
                            "参考勝率1", "参考勝率2",
                            "前走比較",
                            "馬場適性", "馬場×血統", "血統適性",
                            "父", "脚質", "騎手", "材料",
                        ]
                    display_cols = [c for c in display_cols if c in tdf.columns]

                    styled = (
                        tdf[display_cols]
                        .style.apply(highlight_row, axis=1)
                    )

                    st.dataframe(
                        styled,
                        use_container_width=True,
                        hide_index=True,
                        height=min(len(tdf) * 38 + 40, 600),
                    )

    # フッター
    st.markdown("---")
    st.caption(
        "keiba-ai 予想ページ | "
        "予測はAIモデルによる参考情報です。馬券購入は自己責任でお願いします。"
    )


if __name__ == "__main__":
    main()
