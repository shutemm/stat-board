"""予測結果エクスポートスクリプト

keiba-ai のDBから予測結果を抽出し、公開用JSONファイルとして出力する。
モデルロジック・特徴量・学習コードは一切含まない。出力は表示用データのみ。

使い方:
    python public_prediction/export_predictions.py [--date YYYY-MM-DD]

    --date を省略すると今日の日付を使用する。
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

# keiba-ai のパスを追加
KEIBA_AI_DIR = Path(__file__).resolve().parent.parent.parent / "keiba-ai"
sys.path.insert(0, str(KEIBA_AI_DIR))

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def export_predictions(target_date: date) -> Path:
    """指定日のレース予測結果をJSONにエクスポートする。

    出力フォーマット:
    {
        "exported_at": "2026-04-03T12:00:00",
        "race_date": "2026-04-03",
        "venues": {
            "中山": {
                "races": [
                    {
                        "race_number": 1,
                        "race_name": "...",
                        "course": "ダート1200m",
                        "track_condition": "良",
                        "head_count": 16,
                        "horses": [
                            {
                                "rank": 1,
                                "horse_number": 5,
                                "horse_name": "...",
                                "odds": 3.5,
                                "recommendation": "買",
                                "composite_score": 72.3,
                                "integrated_roi": "135%",
                                "sire_name": "...",
                                "running_style": "先行",
                                "signal_tags": "...",
                            }, ...
                        ],
                        "pace_prediction": {
                            "predicted_pace": "M",
                            "pace_label": "ミドル",
                            "pace_probs": {"H": 0.2, "M": 0.6, "S": 0.2},
                            "advantaged_styles": ["差し", "先行"],
                        },
                    }, ...
                ]
            }, ...
        }
    }
    """
    from database.db import init_db, get_session
    from database.models import Race, RaceResult
    from utils.constants import VENUE_CODES

    init_db()

    # レース取得
    with get_session() as s:
        races = (
            s.query(Race)
            .filter(Race.race_date == target_date)
            .order_by(Race.race_id)
            .all()
        )
        race_list = [{
            "race_id": r.race_id,
            "race_name": r.race_name or "",
            "venue": VENUE_CODES.get(r.venue_code, r.venue_code or ""),
            "venue_code": r.venue_code,
            "course_type": r.course_type or "",
            "distance": r.distance,
            "race_number": r.race_number,
            "track_condition": r.track_condition or "",
            "head_count": r.head_count,
            "pace_category": r.pace_category,
        } for r in races]

    if not race_list:
        print(f"WARNING: {target_date} のレースがDBに見つかりません")
        return None

    # 分析エンジンロード
    from analysis.bet_constructor import BetConstructor

    weights = {
        "dsgs": 0.1, "pace_fitness": 0.3,
        "pl_rating": 0.02, "ts_rating": 0.08, "blood_score": 0.3,
    }
    bc = BetConstructor(weights=weights, top_n_horses=18, ev_threshold=0.5,
                        bet_types=["馬連", "ワイド", "三連複"])
    bc.load()

    combo_chains_df = getattr(bc, '_combo_chains_df', None)

    from analysis.dsgs_scorer import DSGSScorer
    dsgs_scorer = DSGSScorer()
    dsgs_scorer.load(combo_chains_df)

    from analysis.raw_feature_scorer import RawFeatureScorer
    raw_scorer = RawFeatureScorer()
    raw_scorer.load()

    from analysis.cushion_analysis import CushionAnalyzer
    cushion_analyzer = CushionAnalyzer()
    cushion_analyzer.load()

    SIGNAL_NAMES = {}
    try:
        from analysis.combo_chain_predictor import COMBOS_DIRT, COMBOS_TURF
        for i, combo in enumerate(COMBOS_DIRT):
            SIGNAL_NAMES[f"ダート_{i}"] = combo["name"]
        for i, combo in enumerate(COMBOS_TURF):
            SIGNAL_NAMES[f"芝_{i}"] = combo["name"]
    except ImportError:
        pass

    PACE_LABELS = {"H": "ハイ", "M": "ミドル", "S": "スロー"}

    # 会場ごとに整理
    venues = {}
    for r in race_list:
        v = r["venue"]
        if v not in venues:
            venues[v] = []
        venues[v].append(r)

    output = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "race_date": str(target_date),
        "venues": {},
    }

    for venue_name, venue_races in venues.items():
        venue_output = {"races": []}

        for race in venue_races:
            race_id = race["race_id"]
            ct = race["course_type"]
            dist = race["distance"]
            track_cond = race["track_condition"]

            scores = bc.compute_composite_scores(race_id)
            if not scores:
                continue

            # 出走馬情報
            with get_session() as s:
                results = (
                    s.query(RaceResult)
                    .filter_by(race_id=race_id)
                    .order_by(RaceResult.horse_number)
                    .all()
                )
                entries_map = {
                    r.horse_number: {
                        "horse_name": r.horse_name,
                        "horse_id": r.horse_id,
                        "odds": r.odds,
                        "running_style": r.running_style or "",
                        "jockey_name": r.jockey_name or "",
                    }
                    for r in results
                }

            # 展開予測
            pace_data = None
            try:
                pace_result = bc._pace_predictor.predict_pace(race_id)
                if "error" not in pace_result:
                    pred_pace = pace_result.get("predicted_pace", "?")
                    pace_data = {
                        "predicted_pace": pred_pace,
                        "pace_label": PACE_LABELS.get(pred_pace, pred_pace),
                        "pace_probs": {
                            k: round(v, 2)
                            for k, v in pace_result.get("pace_probs", {}).items()
                        },
                        "advantaged_styles": pace_result.get("advantaged_styles", []),
                    }
            except Exception:
                pass

            # 展開適性
            pace_fitness_map = {}
            try:
                pf = bc._pace_predictor.calculate_pace_fitness(race_id)
                if "error" not in pf:
                    for h in pf.get("horses", []):
                        pace_fitness_map[h["horse_number"]] = {
                            "advantage": h["style_advantage"],
                            "running_style": h["running_style"],
                        }
            except Exception:
                pass

            # 血統情報
            blood_info_map = {}
            if hasattr(bc, '_blood_analyzer') and bc._blood_analyzer is not None:
                for hn, sc in scores.items():
                    hid = sc.get("horse_id")
                    if hid:
                        bi = bc._blood_analyzer.get_horse_blood_info(hid)
                        if bi:
                            blood_info_map[hn] = bi

            # クッションCV（芝のみ）
            race_cushion_cv = None
            if ct == "芝":
                race_cushion_cv = cushion_analyzer.get_cushion_value_for_date(
                    race.get("venue_code", ""), str(target_date))

            # 馬ごとの情報を構築
            horses_output = []
            # composite_rank でソート
            sorted_scores = sorted(
                scores.items(),
                key=lambda x: x[1].get("composite_rank", 999),
            )

            for rank_i, (hn, sc) in enumerate(sorted_scores, 1):
                horse_id = sc.get("horse_id", "")
                e = entries_map.get(hn, {})
                pf_info = pace_fitness_map.get(hn, {})

                # 推奨判定（DSGS ROI + composite_rank ベース）
                recommendation = ""
                if ct in ("ダート", "芝") and horse_id:
                    dsgs_roi_info = dsgs_scorer.get_horse_dsgs_roi(
                        horse_id, race_id, ct, combo_chains_df)
                    dims_passed = dsgs_roi_info.get("dims_passed", 0)
                    total_dims = len(dsgs_roi_info.get("dim_details", {}))
                    # 買: 多次元パス + 上位ランク
                    if dims_passed >= max(total_dims - 1, 2) and rank_i <= 5:
                        recommendation = "買"
                    # 避: 全次元不通過 + 下位ランク
                    elif dims_passed == 0 and rank_i > len(scores) * 0.7:
                        recommendation = "避"

                # シグナルタグ
                signal_tags = ""
                if combo_chains_df is not None and not combo_chains_df.empty:
                    prefix = "ダート_" if ct == "ダート" else "芝_" if ct == "芝" else ""
                    if prefix:
                        import pandas as pd
                        cols = [c for c in combo_chains_df.columns if c.startswith(prefix)]
                        if cols:
                            mask = ((combo_chains_df["horse_id"] == horse_id) &
                                    (combo_chains_df["race_id"] == race_id))
                            rows = combo_chains_df.loc[mask, cols]
                            if rows.empty:
                                horse_rows = combo_chains_df[
                                    combo_chains_df["horse_id"] == horse_id].copy()
                                if not horse_rows.empty:
                                    horse_rows = horse_rows.sort_values(
                                        "race_id", ascending=False)
                                    prior = horse_rows[horse_rows["race_id"] < race_id]
                                    if not prior.empty:
                                        rows = prior.iloc[[0]][cols]
                            if not rows.empty:
                                row = rows.iloc[0]
                                tags = []
                                for col in cols:
                                    val = row.get(col)
                                    if pd.isna(val):
                                        continue
                                    col_data = combo_chains_df[col].dropna()
                                    if len(col_data) < 10:
                                        continue
                                    std = col_data.std()
                                    threshold = max(std * 1.0, 2.0)
                                    signal_name = SIGNAL_NAMES.get(col, col)
                                    if val < -threshold:
                                        tags.append(f"不利:{signal_name}")
                                    elif val > threshold:
                                        tags.append(f"有利:{signal_name}")
                                signal_tags = ", ".join(tags)

                # 各ROIソースを個別取得
                pattern_roi_str = "-"
                accumulation_roi_str = "-"
                roll5_roi_str = "-"
                last_race_roi_str = "-"
                integrated_roi_str = "-"

                if horse_id:
                    # 1. パターンROI（複合条件パターン）
                    matched_pats = raw_scorer.check_highlight_patterns(horse_id, ct)
                    if matched_pats:
                        best_pat = max(matched_pats, key=lambda p: p["oos_roi"])
                        pattern_roi_str = f"{best_pat['oos_roi']:.0f}%"

                    # 2. 蓄積SIG ROI
                    accum_pats = raw_scorer.check_accumulation_patterns(horse_id, ct)
                    if accum_pats:
                        best_acc = max(accum_pats, key=lambda p: p["oos_roi"])
                        accumulation_roi_str = f"{best_acc['oos_roi']:.0f}%"

                    # 3. 5走平均rank_dev ROI
                    mr_info = raw_scorer.get_horse_multi_race_features(horse_id, ct)
                    if mr_info and mr_info.get("roll5_roi") is not None:
                        roll5_roi_str = f"{mr_info['roll5_roi']:.0f}%"

                    # 4. 前走rank_dev ROI
                    raw_feats = raw_scorer.get_horse_raw_features(horse_id, ct)
                    rd_info = raw_feats.get("rank_dev")
                    if rd_info and rd_info.get("oos_roi") is not None:
                        last_race_roi_str = f"{rd_info['oos_roi']:.0f}%"

                    # 統合ROI（後方互換）
                    ir_info = raw_scorer.get_integrated_roi(horse_id, ct)
                    if ir_info["roi"] is not None:
                        integrated_roi_str = f"{ir_info['roi']:.0f}%"

                # 推定ROI: フォールバック（パターン→蓄積→5走→前走）
                estimated_roi_str = "-"
                roi_source = ""
                for _roi_str, _src in [
                    (pattern_roi_str, "パターン"),
                    (accumulation_roi_str, "蓄積"),
                    (roll5_roi_str, "5走"),
                    (last_race_roi_str, "前走"),
                ]:
                    if _roi_str != "-":
                        estimated_roi_str = _roi_str
                        roi_source = _src
                        break

                # 血統
                bi = blood_info_map.get(hn, {})
                sire_name = bi.get("sire_name", "")

                # 血統ROI
                blood_roi_str = "-"
                if horse_id and hasattr(bc, '_blood_analyzer') and bc._blood_analyzer:
                    blood_roi = bc._blood_analyzer.compute_blood_roi(
                        horse_id, ct, dist,
                        track_cond if track_cond else None,
                        race.get("venue_code"))
                    if blood_roi is not None:
                        blood_roi_str = f"{blood_roi:.0f}%"

                odds_val = e.get("odds")

                # composite_score は 0-1 スケール → 0-100 に変換して表示
                raw_score = sc.get("composite_score", 0)
                display_score = round(raw_score * 100, 1)

                horse_data = {
                    "rank": rank_i,
                    "horse_number": hn,
                    "horse_name": sc.get("horse_name", ""),
                    "odds": round(odds_val, 1) if odds_val else None,
                    "recommendation": recommendation,
                    "composite_score": display_score,
                    "estimated_roi": estimated_roi_str,
                    "roi_source": roi_source,
                    "pattern_roi": pattern_roi_str,
                    "accumulation_roi": accumulation_roi_str,
                    "roll5_roi": roll5_roi_str,
                    "last_race_roi": last_race_roi_str,
                    "integrated_roi": integrated_roi_str,
                    "blood_roi": blood_roi_str,
                    "sire_name": sire_name,
                    "running_style": pf_info.get("running_style",
                                                  e.get("running_style", "")),
                    "jockey_name": e.get("jockey_name", ""),
                    "signal_tags": signal_tags,
                }
                horses_output.append(horse_data)

            race_output = {
                "race_number": race["race_number"],
                "race_name": race["race_name"],
                "course": f"{ct}{dist}m",
                "course_type": ct,
                "track_condition": track_cond,
                "head_count": race["head_count"] or len(scores),
                "horses": horses_output,
            }
            if pace_data:
                race_output["pace_prediction"] = pace_data

            venue_output["races"].append(race_output)

        if venue_output["races"]:
            output["venues"][venue_name] = venue_output

    # 出力
    date_str = target_date.strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"predictions_{date_str}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"OK: {output_path}")
    print(f"   venues: {list(output['venues'].keys())}")
    total_races = sum(
        len(v["races"]) for v in output["venues"].values())
    print(f"   races: {total_races}")

    # latest シンボリックリンク的にコピー
    latest_path = OUTPUT_DIR / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"   latest: {latest_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="予測結果エクスポート")
    parser.add_argument("--date", type=str, default=None,
                        help="対象日 (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target = date.today()

    print(f"Exporting predictions for {target} ...")
    result = export_predictions(target)
    if result is None:
        print("WARN: No data exported.")
        sys.exit(1)


if __name__ == "__main__":
    main()
