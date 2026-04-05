"""keiba-ai 予想公開ページ

予測結果JSONを読み込んで表示するStreamlitアプリ。
モデルコード・特徴量計算・学習ロジックは一切含まない。

データ更新: export_predictions.py で生成した JSON を data/ に配置する。
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

# ティア色マッピング（全10段階: 高ROI 130-90% + 低ROI 80-40%）
TIER_STYLES = {
    # 高ROIティア（恵まれた馬）
    "130%": {
        "bg": BG_GREEN_STRONG,
        "color": TEXT_GREEN,
        "font_weight": "bold",
        "font_size": "1.1em",
    },
    "120%": {
        "bg": BG_GREEN_MEDIUM,
        "color": TEXT_GREEN_MED,
        "font_weight": "bold",
        "font_size": "1.05em",
    },
    "110%": {
        "bg": BG_GREEN_LIGHT,
        "color": TEXT_GREEN_MED,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "100%": {
        "bg": "",
        "color": TEXT_LIGHT,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "90%": {
        "bg": BG_YELLOW_LIGHT,
        "color": TEXT_YELLOW,
        "font_weight": "normal",
        "font_size": "1em",
    },
    # 低ROIティア（恵まれない馬）
    "80%": {
        "bg": BG_ORANGE_LIGHT,
        "color": TEXT_ORANGE_LIGHT,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "70%": {
        "bg": BG_ORANGE,
        "color": TEXT_ORANGE,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "60%": {
        "bg": BG_RED_LIGHT,
        "color": TEXT_RED,
        "font_weight": "normal",
        "font_size": "1em",
    },
    "50%": {
        "bg": BG_RED_MEDIUM,
        "color": TEXT_RED,
        "font_weight": "bold",
        "font_size": "1em",
    },
    "40%": {
        "bg": BG_RED_STRONG,
        "color": TEXT_RED_STRONG,
        "font_weight": "bold",
        "font_size": "1.1em",
    },
}

# ティアソート順（高い方が上位、低い方が下位）
TIER_SORT_ORDER = {
    "130%": 0, "120%": 1, "110%": 2, "100%": 3, "90%": 4,
    "80%": 6, "70%": 7, "60%": 8, "50%": 9, "40%": 10,
    "-": 5,
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


# ============================================================
# ヘルパー
# ============================================================

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
    """ティア文字列からセルスタイルを返す"""
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


def parse_roi_value(val_str) -> float | None:
    """ROI文字列から数値を抽出する。"""
    return _extract_roi_number(str(val_str) if val_str else "")


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

    # Rank ハイライト
    if "Rank" in cols:
        idx = cols.index("Rank")
        rank = row["Rank"]
        if rank == 1:
            styles[idx] = f"background-color: {BG_GREEN_STRONG}; color: #ffffff; font-weight: bold"
        elif rank <= 3:
            styles[idx] = f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_LIGHT}"

    # ROIティア列ハイライト
    if "ROIティア" in cols:
        idx = cols.index("ROIティア")
        s = tier_style(str(row["ROIティア"]))
        if s:
            styles[idx] = s

    # 馬場ROI列ハイライト
    if "馬場ROI" in cols:
        idx = cols.index("馬場ROI")
        s = roi_style(str(row["馬場ROI"]))
        if s:
            styles[idx] = s

    # 含水率×血統ROI列ハイライト
    if "含水率×血統ROI" in cols:
        idx = cols.index("含水率×血統ROI")
        s = roi_style(str(row["含水率×血統ROI"]))
        if s:
            styles[idx] = s

    # 血統ROI列ハイライト
    if "血統ROI" in cols:
        idx = cols.index("血統ROI")
        s = roi_style(str(row["血統ROI"]))
        if s:
            styles[idx] = s

    # 勝率ハイライト
    if "勝率" in cols:
        idx = cols.index("勝率")
        v = _extract_roi_number(str(row["勝率"]).replace("%", "% "))
        # _extract_roi_number expects "%" — workaround: parse directly
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

    # 展開列ハイライト
    if "展開" in cols:
        idx = cols.index("展開")
        v = row["展開"]
        if v == "有利":
            styles[idx] = f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
        elif v == "不利":
            styles[idx] = f"background-color: {BG_RED_LIGHT}; color: {TEXT_RED}"

    # 条件バイアス列ハイライト
    if "条件バイアス" in cols:
        idx = cols.index("条件バイアス")
        v = str(row["条件バイアス"])
        if "有利転換" in v:
            styles[idx] = f"background-color: {BG_GREEN_MEDIUM}; color: {TEXT_GREEN}; font-weight: bold"
        elif "不利転換" in v:
            styles[idx] = f"background-color: {BG_RED_MEDIUM}; color: {TEXT_RED}; font-weight: bold"
        elif v != "-":
            try:
                cb_val = float(v.split()[0])
                if cb_val > 1.05:
                    styles[idx] = f"color: {TEXT_GREEN_MED}"
                elif cb_val < 0.95:
                    styles[idx] = f"color: {TEXT_ORANGE}"
            except (ValueError, IndexError):
                pass

    return styles


def render_pace_prediction(pace_data: dict):
    """展開予測を表示"""
    if not pace_data:
        return

    pred = pace_data.get("predicted_pace", "?")
    label = pace_data.get("pace_label", pred)
    probs = pace_data.get("pace_probs", {})
    color = PACE_COLORS.get(pred, TEXT_LIGHT)

    prob_str = " / ".join(f"{k}:{v:.0%}" for k, v in probs.items())
    st.markdown(
        f"**展開予測**: <span style='color:{color}; font-size:1.2em; "
        f"font-weight:bold;'>{label}</span> ({prob_str})",
        unsafe_allow_html=True,
    )

    adv = pace_data.get("advantaged_styles", [])
    if adv:
        st.caption(f"有利脚質: {', '.join(adv)}")


# ============================================================
# メインUI
# ============================================================

def main():
    # サイドバー
    with st.sidebar:
        st.title("keiba-ai 予想")
        st.markdown("---")

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
        st.caption("keiba-ai prediction viewer")

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

    # 注目レースランキング（ティア130%/120%の馬がいるレースを上位に、40%/50%は注意マーク）
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

            # スコア: 130%馬数*10 + 120%馬数*5 + 110%馬数*2 + 100%馬数*1
            highlight_score = (
                tier_counts["130%"] * 10
                + tier_counts["120%"] * 5
                + tier_counts["110%"] * 2
                + tier_counts["100%"] * 1
            )

            # Top3のティア馬名（高ティアと低ティアの注目馬）
            tier_horses = [
                (h, TIER_SORT_ORDER.get(h.get("and_filter_tier") or "-", 5))
                for h in horses
            ]
            tier_horses.sort(key=lambda x: (x[1], x[0].get("rank", 999)))
            top3_parts = []
            for h, _ in tier_horses[:3]:
                if not h.get("and_filter_tier"):
                    continue
                wp = h.get("win_prob")
                wp_str = f" {wp * 100:.0f}%" if wp else ""
                top3_parts.append(
                    f"{h['horse_number']}{h['horse_name']}"
                    f"({h.get('and_filter_tier', '-')}{wp_str})"
                )
            top3_names = ", ".join(top3_parts)

            # ティアサマリー文字列（高ティアと低ティア両方表示）
            tier_summary_parts = []
            for tk in ALL_TIER_KEYS:
                if tier_counts[tk] > 0:
                    tier_summary_parts.append(f"{tk}:{tier_counts[tk]}")
            tier_summary = " ".join(tier_summary_parts) if tier_summary_parts else "-"

            # 低ティア注意マーク
            n_low_tier = tier_counts["40%"] + tier_counts["50%"]
            alert_mark = ""
            if n_low_tier > 0:
                alert_mark = f" [!{n_low_tier}]"

            all_races.append({
                "venue": venue_name,
                "race_number": race["race_number"],
                "race_name": race["race_name"],
                "course": race["course"],
                "head_count": race.get("head_count", len(horses)),
                "highlight_score": highlight_score,
                "tier_summary": tier_summary,
                "top3_names": top3_names,
                "n_high_tier": tier_counts["130%"] + tier_counts["120%"],
                "n_low_tier": n_low_tier,
                "alert_mark": alert_mark,
                "race_data": race,
            })

    all_races.sort(key=lambda x: x["highlight_score"], reverse=True)

    # 注目レーストップ5
    st.markdown("### 注目レース（ROIティア順）")
    ranking_rows = []
    for i, r in enumerate(all_races[:5]):
        ranking_rows.append({
            "順位": f"{i + 1}.",
            "場所": r["venue"],
            "R": f"{r['race_number']}R",
            "レース名": r["race_name"] + r.get("alert_mark", ""),
            "コース": r["course"],
            "頭数": r["head_count"],
            "ティア分布": r["tier_summary"],
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

                # ティア別カウント
                tier_counts = {}
                for h in horses:
                    t = h.get("and_filter_tier") or "-"
                    tier_counts[t] = tier_counts.get(t, 0) + 1

                tier_str_parts = []
                for tk in ALL_TIER_KEYS:
                    if tier_counts.get(tk, 0) > 0:
                        tier_str_parts.append(f"{tk}:{tier_counts[tk]}")
                tier_str = " ".join(tier_str_parts) if tier_str_parts else ""

                has_high_tier = tier_counts.get("130%", 0) + tier_counts.get("120%", 0) > 0
                has_low_tier = tier_counts.get("40%", 0) + tier_counts.get("50%", 0) > 0

                header = (
                    f"{race_num}R **{race_name}** "
                    f"({course} {track_cond}) "
                    f"{tier_str}"
                )

                with st.expander(header, expanded=(has_high_tier or has_low_tier)):
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

                    # 芝レース判定（cushion_roi表示用）
                    is_turf = course.startswith("芝") if course else False

                    table_rows = []
                    for h in horses:
                        tier_val = h.get("and_filter_tier") or "-"

                        # 馬場ROI: 芝レースのみ cushion_roi を表示、ダートは「-」
                        if is_turf:
                            cushion_roi_val = h.get("cushion_roi", "-") or "-"
                        else:
                            cushion_roi_val = "-"

                        # 勝率・複勝率
                        wp = h.get("win_prob")
                        pp = h.get("place_prob")
                        win_prob_str = f"{wp * 100:.1f}%" if wp else "-"
                        place_prob_str = f"{pp * 100:.1f}%" if pp else "-"

                        # 含水率×血統ROI
                        moisture_blood_roi_val = h.get("moisture_blood_roi", "-") or "-"

                        # 条件バイアス（前走→今走の条件変化）
                        context_bias_val = h.get("context_bias", "-") or "-"

                        row_data = {
                            "Rank": h.get("rank", 0),
                            "推奨": h.get("recommendation", ""),
                            "番": h.get("horse_number", ""),
                            "馬名": h.get("horse_name", ""),
                            "ROIティア": tier_val,
                            "勝率": win_prob_str,
                            "複勝率": place_prob_str,
                            "条件バイアス": context_bias_val,
                            "馬場ROI": cushion_roi_val,
                            "含水率×血統ROI": moisture_blood_roi_val,
                            "血統ROI": h.get("blood_roi", "-") or "-",
                            "父": h.get("sire_name", ""),
                            "脚質": h.get("running_style", ""),
                            "騎手": h.get("jockey_name", ""),
                            "オッズ": h.get("odds") or "-",
                            "シグナル": h.get("signal_tags", ""),
                        }
                        table_rows.append(row_data)

                    tdf = pd.DataFrame(table_rows)

                    # ティア順にソート（高い方が上位、同ティアはRank順）
                    tdf["_tier_sort"] = tdf["ROIティア"].map(TIER_SORT_ORDER).fillna(5)
                    tdf = (tdf
                           .sort_values(["_tier_sort", "Rank"])
                           .drop(columns="_tier_sort")
                           .reset_index(drop=True))

                    display_cols = [
                        "Rank", "推奨", "番", "馬名",
                        "ROIティア", "勝率", "複勝率",
                        "条件バイアス",
                        "馬場ROI", "含水率×血統ROI", "血統ROI",
                        "父", "脚質", "騎手", "オッズ", "シグナル",
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
        "keiba-ai 予想公開ページ | "
        "予測はAIモデルによる参考情報です。馬券購入は自己責任でお願いします。"
    )


if __name__ == "__main__":
    main()
