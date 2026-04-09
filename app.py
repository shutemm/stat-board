"""keiba-ai 予想公開ページ

予測結果JSONを読み込んで表示するStreamlitアプリ。
モデルコード・特徴量計算・学習ロジックは一切含まない。

データ更新: export_predictions.py で生成した JSON を data/ に配置する。
レートデータ: export_ratings.py で生成した ratings.json を data/ に配置する。
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
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
# メインUI
# ============================================================

def main():
    # サイドバー - ページ選択
    with st.sidebar:
        st.title("keiba-ai")
        page = st.radio(
            "ページ",
            ["予想", "レート一覧", "馬詳細"],
            index=0,
        )
        st.markdown("---")

    if page == "予想":
        page_predictions()
    elif page == "レート一覧":
        page_ratings()
    elif page == "馬詳細":
        page_horse_detail()


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
    """レート値に応じたセルスタイルを返す（絶対レートスケール対応）"""
    if v >= 75:
        return f"background-color: {BG_GREEN_STRONG}; color: {TEXT_GREEN}; font-weight: bold"
    elif v >= 60:
        return f"background-color: {BG_GREEN_MEDIUM}; color: {TEXT_GREEN_MED}; font-weight: bold"
    elif v >= 48:
        return f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
    elif v >= 35:
        return f"color: {TEXT_LIGHT}"
    elif v >= 25:
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
        latest_cls = latest.get("cls", "")

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
            "クラス": latest_cls,
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
    df = df.sort_values("レート", ascending=False).reset_index(drop=True)
    df.insert(0, "順位", range(1, len(df) + 1))

    st.markdown(f"**{len(df)}頭**")

    display_cols = ["順位", "馬名", "レート", "クラス", "直近レース", "日付", "着順", "オッズ"]

    def highlight_rating_row(row):
        styles = [""] * len(row)
        cols = list(row.index)
        if "レート" in cols:
            idx = cols.index("レート")
            v = row["レート"]
            styles[idx] = _rating_color_style(v)
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
                except (ValueError, TypeError):
                    pass
        return styles

    styled = df[display_cols].style.apply(highlight_rating_row, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(len(df) * 38 + 40, 800),
    )


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

    with st.sidebar:
        search = st.text_input("馬名で検索", "", key="horse_search")
        if search:
            filtered = [(hid, n, cr) for hid, n, cr in horse_list if search in n]
        else:
            filtered = horse_list[:200]

        if not filtered:
            st.info("該当する馬がいません。")
            return

        options = {f"{n} (レート: {cr:.1f})": hid for hid, n, cr in filtered}
        selected_label = st.selectbox(
            "馬を選択",
            list(options.keys()),
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
                st.metric("芝レート", f"{turf_data['cr']:.1f}")
            else:
                st.metric("芝レート", "-")
        with cols[1]:
            if turf_data and len(turf_data.get("h", [])) >= 2:
                prev = turf_data["h"][-2]["r"]
                diff = turf_data["cr"] - prev
                st.metric("芝 前走差", f"{diff:+.1f}")
            else:
                st.metric("芝 前走差", "-")
        with cols[2]:
            if dirt_data:
                st.metric("ダートレート", f"{dirt_data['cr']:.1f}")
            else:
                st.metric("ダートレート", "-")
        with cols[3]:
            if dirt_data and len(dirt_data.get("h", [])) >= 2:
                prev = dirt_data["h"][-2]["r"]
                diff = dirt_data["cr"] - prev
                st.metric("ダート 前走差", f"{diff:+.1f}")
            else:
                st.metric("ダート 前走差", "-")

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
            "クラス": rh.get("cls", ""),
            "場所": rh.get("v", ""),
            "コース": f"{rh.get('ct', '')} {rh.get('dist', '')}m",
            "馬場": rh.get("tc", ""),
            "着順": fo if fo else "-",
            "オッズ": f"{odds:.1f}" if odds else "-",
            "レート": f"{rh.get('r', 0):.1f}",
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
