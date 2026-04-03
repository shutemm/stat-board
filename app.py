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

def roi_style(val_str: str) -> str:
    """ROI文字列からセルスタイルを返す"""
    if val_str == "-" or "%" not in val_str:
        return ""
    try:
        v = float(val_str.replace("%", "").split("(")[0])
    except (ValueError, TypeError):
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


def parse_roi_value(val_str) -> float | None:
    """ROI文字列から数値を抽出する。パース失敗時はNoneを返す。"""
    if val_str == "-" or not val_str or "%" not in str(val_str):
        return None
    try:
        return float(str(val_str).replace("%", "").split("(")[0])
    except (ValueError, TypeError):
        return None


def get_estimated_roi(horse: dict) -> str:
    """馬データから推定ROIを取得（フォールバック付き後方互換）。"""
    est = horse.get("estimated_roi", "")
    if est and est != "-":
        return est
    for key in ("pattern_roi", "accumulation_roi", "roll5_roi", "last_race_roi"):
        v = horse.get(key, "-")
        if v and v != "-":
            return v
    return "-"


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

    # 推定ROI列ハイライト（メイン）
    if "推定ROI" in cols:
        idx = cols.index("推定ROI")
        s = roi_style(str(row["推定ROI"]))
        if s:
            styles[idx] = s + "; font-weight: bold; font-size: 1.05em"

    # 個別ROI列ハイライト
    for roi_col in ("パターン", "蓄積", "5走", "前走", "血統ROI"):
        if roi_col in cols:
            idx = cols.index(roi_col)
            s = roi_style(str(row[roi_col]))
            if s:
                styles[idx] = s

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

    # 注目レースランキング（全会場横断で推定ROI上位）
    all_races = []
    for venue_name, venue_data in venues_data.items():
        for race in venue_data.get("races", []):
            horses = race.get("horses", [])
            buy_horses = [h for h in horses if h.get("recommendation") == "買"]

            # 各馬の推定ROIを数値化し、上位3頭の平均ROIで評価
            roi_values = []
            for h in horses:
                rv = parse_roi_value(get_estimated_roi(h))
                if rv is not None:
                    roi_values.append(rv)
            roi_values.sort(reverse=True)
            top3_roi = roi_values[:3]
            avg_top3_roi = sum(top3_roi) / len(top3_roi) if top3_roi else 0

            # 推定ROI順で上位3頭の馬名
            horses_with_roi = [
                (h, parse_roi_value(get_estimated_roi(h)) or 0)
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
    st.markdown("### 注目レース（推定ROI順）")
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

                # 最高推定ROIを取得
                top_roi_vals = [
                    parse_roi_value(get_estimated_roi(h))
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

                    table_rows = []
                    for h in horses:
                        # 推定ROI: JSONに含まれていればそれを使用、
                        # なければ後方互換でフォールバック計算
                        est_roi = h.get("estimated_roi", "")
                        if not est_roi or est_roi == "-":
                            for _key in (
                                "pattern_roi", "accumulation_roi",
                                "roll5_roi", "last_race_roi",
                            ):
                                v = h.get(_key, "-")
                                if v and v != "-":
                                    est_roi = v
                                    break
                            else:
                                est_roi = "-"

                        table_rows.append({
                            "Rank": h.get("rank", 0),
                            "推奨": h.get("recommendation", ""),
                            "番": h.get("horse_number", ""),
                            "馬名": h.get("horse_name", ""),
                            "推定ROI": est_roi,
                            "パターン": h.get("pattern_roi", "-") or "-",
                            "蓄積": h.get("accumulation_roi", "-") or "-",
                            "5走": h.get("roll5_roi", "-") or "-",
                            "前走": h.get("last_race_roi", "-") or "-",
                            "血統ROI": h.get("blood_roi", "-"),
                            "父": h.get("sire_name", ""),
                            "脚質": h.get("running_style", ""),
                            "騎手": h.get("jockey_name", ""),
                            "オッズ": h.get("odds") or "-",
                            "シグナル": h.get("signal_tags", ""),
                        })

                    tdf = pd.DataFrame(table_rows)

                    # 推奨順にソート: 買 > 避 > その他、同じカテゴリ内はRank順
                    sort_key = tdf["推奨"].map({"買": 0, "避": 2, "": 1})
                    tdf = (tdf.assign(_sort=sort_key)
                           .sort_values(["_sort", "Rank"])
                           .drop(columns="_sort")
                           .reset_index(drop=True))

                    display_cols = [
                        "Rank", "推奨", "番", "馬名",
                        "推定ROI", "パターン", "蓄積", "5走", "前走",
                        "血統ROI", "父", "脚質", "騎手", "オッズ", "シグナル",
                    ]
                    display_cols = [c for c in display_cols if c in tdf.columns]

                    st.dataframe(
                        tdf[display_cols].style.apply(highlight_row, axis=1),
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
