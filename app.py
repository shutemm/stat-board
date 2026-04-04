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
BG_RED_LIGHT = "#3d1a1a"
TEXT_GREEN = "#4caf50"
TEXT_GREEN_MED = "#66bb6a"
TEXT_RED = "#ef5350"
TEXT_LIGHT = "#e0e0e0"
TEXT_MUTED = "#b0b0b0"
PACE_COLORS = {"H": "#ef5350", "M": "#ffa726", "S": "#42a5f5"}

# グループ別ヘッダー色
GROUP_A_HEADER_BG = "#1a3352"   # 青系 — 馬の力
GROUP_B_HEADER_BG = "#3d2e1a"   # 橙系 — レース条件
GROUP_A_HEADER_TEXT = "#64b5f6"  # 青テキスト
GROUP_B_HEADER_TEXT = "#ffb74d"  # 橙テキスト

# Group A (rank_dev系) の列名リスト
GROUP_A_ROI_COLS = ("ROI", "5走ROI", "パターン", "蓄積", "DSGS")
# Group B (レース条件系) の列名リスト
GROUP_B_ROI_COLS = ("速度ROI", "着差ROI", "血統ROI", "加速ROI", "馬場ROI")

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
    .group-legend {
        display: flex; gap: 16px; margin-bottom: 4px; font-size: 0.85em;
    }
    .group-legend-a {
        color: #64b5f6; font-weight: bold;
        border-left: 3px solid #64b5f6; padding-left: 6px;
    }
    .group-legend-b {
        color: #ffb74d; font-weight: bold;
        border-left: 3px solid #ffb74d; padding-left: 6px;
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
        # ファイル名から日付を抽出: predictions_YYYYMMDD.json
        stem = f.stem  # predictions_20260403
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
    """ROI文字列から数値部分を抽出する。

    対応フォーマット: "120%", "~120%(3走)", "(82%芝)", "(120%ダ)"
    """
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


def dsgs_style(val_str: str) -> str:
    """DSGS dims_passed 表示 (X/Y形式) からセルスタイルを返す。

    フォーマット: "3/4", "0/4", "120%(4/4)", "-"
    """
    if not val_str or val_str == "-":
        return ""
    # ROI付き形式 (例: "120%(4/4)")
    if "%" in val_str:
        return roi_style(val_str)
    # X/Y 形式
    try:
        parts = val_str.split("/")
        if len(parts) == 2:
            passed = int(parts[0])
            total = int(parts[1])
            if total == 0:
                return ""
            ratio = passed / total
            if ratio >= 1.0:
                return f"background-color: {BG_GREEN_STRONG}; color: {TEXT_GREEN}; font-weight: bold"
            if ratio >= 0.75:
                return f"background-color: {BG_GREEN_MEDIUM}; color: {TEXT_GREEN_MED}; font-weight: bold"
            if ratio >= 0.5:
                return f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
            if ratio >= 0.25:
                return f"color: {TEXT_LIGHT}"
            return f"color: {TEXT_MUTED}"
    except (ValueError, TypeError):
        pass
    return ""


def parse_roi_value(val_str) -> float | None:
    """ROI文字列から数値を抽出する。パース失敗時はNoneを返す。"""
    return _extract_roi_number(str(val_str) if val_str else "")


def highlight_row(row):
    """テーブル行のハイライト（ROIベース）"""
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

    # ROI列ハイライト（メイン — 太字）
    if "ROI" in cols:
        idx = cols.index("ROI")
        s = roi_style(str(row["ROI"]))
        if s:
            styles[idx] = s + "; font-weight: bold; font-size: 1.05em"

    # 個別ROI列ハイライト — Group A (rank_dev系)
    for roi_col in GROUP_A_ROI_COLS:
        if roi_col in cols:
            idx = cols.index(roi_col)
            if roi_col == "DSGS":
                s = dsgs_style(str(row[roi_col]))
            else:
                s = roi_style(str(row[roi_col]))
            if s:
                styles[idx] = s

    # 個別ROI列ハイライト — Group B (レース条件系)
    for roi_col in GROUP_B_ROI_COLS:
        if roi_col in cols:
            idx = cols.index(roi_col)
            s = roi_style(str(row[roi_col]))
            if s:
                styles[idx] = s

    # 展開列ハイライト
    if "展開" in cols:
        idx = cols.index("展開")
        v = row["展開"]
        if v == "有利":
            styles[idx] = f"background-color: {BG_GREEN_LIGHT}; color: {TEXT_GREEN_MED}"
        elif v == "不利":
            styles[idx] = f"background-color: {BG_RED_LIGHT}; color: {TEXT_RED}"

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

    # 注目レースランキング（全会場横断でベースROI上位）
    all_races = []
    for venue_name, venue_data in venues_data.items():
        for race in venue_data.get("races", []):
            horses = race.get("horses", [])
            buy_horses = [h for h in horses if h.get("recommendation") == "買"]

            # 各馬のベースROIを数値化し、上位3頭の平均ROIで評価
            roi_values = []
            for h in horses:
                rv = parse_roi_value(h.get("base_roi", "-"))
                if rv is not None:
                    roi_values.append(rv)
            roi_values.sort(reverse=True)
            top3_roi = roi_values[:3]
            avg_top3_roi = sum(top3_roi) / len(top3_roi) if top3_roi else 0

            # ベースROI順で上位3頭の馬名
            horses_with_roi = [
                (h, parse_roi_value(h.get("base_roi", "-")) or 0)
                for h in horses
            ]
            horses_with_roi.sort(key=lambda x: x[1], reverse=True)
            top3_names = ", ".join(
                f"{h['horse_number']}{h['horse_name']}"
                for h, _ in horses_with_roi[:3]
            )

            all_races.append({
                "venue": venue_name,
                "race_number": race["race_number"],
                "race_name": race["race_name"],
                "course": race["course"],
                "head_count": race.get("head_count", len(horses)),
                "avg_top3_roi": avg_top3_roi,
                "top3_names": top3_names,
                "n_buy": len(buy_horses),
                "race_data": race,
            })

    all_races.sort(key=lambda x: x["avg_top3_roi"], reverse=True)

    # 注目レーストップ5
    st.markdown("### 注目レース（ROI順）")
    ranking_rows = []
    for i, r in enumerate(all_races[:5]):
        ranking_rows.append({
            "順位": f"{i + 1}.",
            "場所": r["venue"],
            "R": f"{r['race_number']}R",
            "レース名": r["race_name"],
            "コース": r["course"],
            "頭数": r["head_count"],
            "Top3 ROI": f"{r['avg_top3_roi']:.0f}%",
            "注目馬": r["top3_names"],
        })

    if ranking_rows:
        rk_df = pd.DataFrame(ranking_rows)

        def highlight_ranking(row):
            styles = [""] * len(row)
            cols = list(row.index)
            if "Top3 ROI" in cols:
                idx = cols.index("Top3 ROI")
                s = roi_style(str(row["Top3 ROI"]))
                if s:
                    styles[idx] = s
            return styles

        st.dataframe(
            rk_df.style.apply(highlight_ranking, axis=1),
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

                buy_count = sum(
                    1 for h in horses if h.get("recommendation") == "買")
                avoid_count = sum(
                    1 for h in horses if h.get("recommendation") == "避")

                count_str = ""
                if buy_count > 0:
                    count_str += f" 買:{buy_count}"
                if avoid_count > 0:
                    count_str += f" 避:{avoid_count}"

                # 最高ベースROIを取得
                top_roi_vals = [
                    parse_roi_value(h.get("base_roi", "-"))
                    for h in horses
                ]
                top_roi_vals = [v for v in top_roi_vals if v is not None]
                top_roi = max(top_roi_vals) if top_roi_vals else 0

                header = (
                    f"{race_num}R **{race_name}** "
                    f"({course} {track_cond}) "
                    f"Top ROI:{top_roi:.0f}%{count_str}"
                )

                with st.expander(header, expanded=(buy_count > 0)):
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

                    # 芝レースかどうかで馬場ROI列の有無を判定
                    is_turf = race.get("course", "").startswith("芝")

                    table_rows = []
                    for h in horses:
                        row_data = {
                            "Rank": h.get("rank", 0),
                            "推奨": h.get("recommendation", ""),
                            "番": h.get("horse_number", ""),
                            "馬名": h.get("horse_name", ""),
                            # Group A: 馬の力 (rank_dev系) + シグナル
                            "ROI": h.get("base_roi", "-") or "-",
                            "5走ROI": h.get("roll5_roi", "-") or "-",
                            "パターン": h.get("pattern_roi", "-") or "-",
                            "蓄積": h.get("accumulation_roi", "-") or "-",
                            "DSGS": h.get("dsgs_roi", "-") or "-",
                            # Group B: レース条件系
                            "速度ROI": h.get("rel_lap_roi", "-") or "-",
                            "着差ROI": h.get("margin_roi", "-") or "-",
                            "血統ROI": h.get("blood_roi", "-") or "-",
                            "加速ROI": h.get("accel_roi", "-") or "-",
                            "展開": h.get("pace_advantage", ""),
                            # 共通
                            "父": h.get("sire_name", ""),
                            "脚質": h.get("running_style", ""),
                            "騎手": h.get("jockey_name", ""),
                            "オッズ": h.get("odds") or "-",
                            "シグナル": h.get("signal_tags", ""),
                        }
                        if is_turf:
                            row_data["馬場ROI"] = h.get("cushion_roi", "-") or "-"
                        table_rows.append(row_data)

                    tdf = pd.DataFrame(table_rows)

                    # 推奨順にソート: 買 > 避 > その他、同じカテゴリ内はRank順
                    sort_key = tdf["推奨"].map({"買": 0, "避": 2, "": 1})
                    tdf = (tdf.assign(_sort=sort_key)
                           .sort_values(["_sort", "Rank"])
                           .drop(columns="_sort")
                           .reset_index(drop=True))

                    # Group A列 + Group B列を視覚的に分ける
                    group_a_cols = ["ROI", "5走ROI", "パターン", "蓄積", "DSGS"]
                    group_b_cols = [
                        "速度ROI", "着差ROI", "血統ROI", "加速ROI",
                        "展開",
                    ]
                    if is_turf:
                        group_b_cols.insert(4, "馬場ROI")

                    display_cols = (
                        ["Rank", "推奨", "番", "馬名"]
                        + group_a_cols
                        + ["|"]  # セパレータ（後で除去）
                        + group_b_cols
                        + ["父", "脚質", "騎手", "オッズ", "シグナル"]
                    )
                    # セパレータ列を除去（実際にはDataFrameに存在しない）
                    display_cols = [
                        c for c in display_cols
                        if c != "|" and c in tdf.columns
                    ]

                    # グループ凡例を表示
                    st.markdown(
                        '<div class="group-legend">'
                        '<span class="group-legend-a">馬の力: ROI / 5走ROI / パターン / 蓄積 / DSGS</span>'
                        '<span class="group-legend-b">レース条件: 速度ROI / 着差ROI / 血統ROI / 加速ROI'
                        + (' / 馬場ROI' if is_turf else '')
                        + ' / 展開</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    # ヘッダー色分け用スタイル関数
                    def _col_header_style(col_name: str) -> dict:
                        """列名に基づくヘッダー装飾用dict（styler.set_table_styles用）"""
                        if col_name in GROUP_A_ROI_COLS:
                            return {
                                "selector": "th",
                                "props": [
                                    ("background-color", GROUP_A_HEADER_BG),
                                    ("color", GROUP_A_HEADER_TEXT),
                                    ("font-weight", "bold"),
                                    ("border-bottom", f"2px solid {GROUP_A_HEADER_TEXT}"),
                                ],
                            }
                        if col_name in GROUP_B_ROI_COLS or col_name == "展開":
                            return {
                                "selector": "th",
                                "props": [
                                    ("background-color", GROUP_B_HEADER_BG),
                                    ("color", GROUP_B_HEADER_TEXT),
                                    ("font-weight", "bold"),
                                    ("border-bottom", f"2px solid {GROUP_B_HEADER_TEXT}"),
                                ],
                            }
                        return None

                    # table_styles を構築
                    table_styles = []
                    for i, col in enumerate(display_cols):
                        hdr = _col_header_style(col)
                        if hdr:
                            table_styles.append({
                                "selector": f"th.col{i}",
                                "props": hdr["props"],
                            })

                    styled = (
                        tdf[display_cols]
                        .style.apply(highlight_row, axis=1)
                        .set_table_styles(table_styles)
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
