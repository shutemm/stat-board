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

# 勝率補正係数
WIN_PROB_ADJ_PATH = KEIBA_AI_DIR / "config" / "win_prob_adjustments.json"


def _load_win_prob_adjustments():
    """win_prob_adjustments.json を読み込む"""
    if not WIN_PROB_ADJ_PATH.exists():
        return None
    with open(WIN_PROB_ADJ_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_race_win_probs(horses_data, win_adj, heatmap_scorer_obj, course_type, distance, track_condition, quintile_edges=None,
                            cushion_roi_map=None, moisture_blood_roi_map=None,
                            cushion_quintile_edges=None, moisture_blood_quintile_edges=None):
    """レース内の全馬について補正済み勝率・複勝率を計算する。

    Bradley-Terry model:
      base_gamma = 1 / odds
      adjusted_gamma = base_gamma * tier_adj * heatmap_adj * cushion_adj * moisture_blood_adj
      win_prob = adjusted_gamma / sum(adjusted_gamma for all horses in race)

    複勝率は上位3頭に入る確率を近似計算する:
      place_prob ≈ 1 - product((1 - p_j / (1 - p_i)) for top competitors)
      ここでは簡易近似として: place_prob = min(1.0, 3 * win_prob * f(head_count))
      ただしより正確にはBT順位モデルの数値計算を使用する。
    """
    if not win_adj or not horses_data:
        return {}

    tier_adj = win_adj.get("tier_adjustments", {})
    heatmap_adj = win_adj.get("heatmap_roi_adjustments", {})
    cushion_adj = win_adj.get("cushion_roi_adjustments", {})
    moisture_blood_adj = win_adj.get("moisture_blood_roi_adjustments", {})

    import numpy as np

    # 各馬のadjusted_gammaを計算
    gamma_list = []
    for h in horses_data:
        odds = h.get("odds")
        if not odds or odds <= 0:
            gamma_list.append({"hn": h["horse_number"], "gamma": 0.0,
                               "base_gamma": 0.0, "tier_m": 1.0, "hm_m": 1.0,
                               "cush_m": 1.0, "mb_m": 1.0})
            continue

        base_gamma = 1.0 / odds

        # ティア調整
        tier = h.get("and_filter_tier")  # e.g. "130%" or None
        if tier:
            tier_key = tier.replace("%", "")  # "130"
        else:
            tier_key = "none"
        t_adj = tier_adj.get(tier_key, tier_adj.get("none", 1.0))

        # ヒートマップROI五分位調整
        horse_id = h.get("_horse_id", "")
        hm_adj = 1.0
        if heatmap_scorer_obj and horse_id and quintile_edges is not None:
            unified = heatmap_scorer_obj.compute_unified_roi(
                horse_id, course_type, distance, track_condition)
            uroi = unified.get("unified_roi")
            if uroi is not None:
                qi = int(np.digitize(uroi, quintile_edges)) + 1
                qi = max(1, min(5, qi))
                q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
                q = q_labels[qi - 1]
                hm_adj = heatmap_adj.get(q, 1.0)

        # クッション値ROI五分位調整
        cush_m = 1.0
        if cushion_adj and cushion_roi_map and cushion_quintile_edges:
            hn_key = h["horse_number"]
            cush_roi = cushion_roi_map.get(hn_key)
            if cush_roi is not None:
                qi = int(np.digitize(cush_roi, cushion_quintile_edges)) + 1
                qi = max(1, min(5, qi))
                q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
                q = q_labels[qi - 1]
                cush_m = cushion_adj.get(q, 1.0)

        # 含水率x血統ROI五分位調整
        mb_m = 1.0
        if moisture_blood_adj and moisture_blood_roi_map and moisture_blood_quintile_edges:
            hn_key = h["horse_number"]
            mb_roi = moisture_blood_roi_map.get(hn_key)
            if mb_roi is not None:
                qi = int(np.digitize(mb_roi, moisture_blood_quintile_edges)) + 1
                qi = max(1, min(5, qi))
                q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
                q = q_labels[qi - 1]
                mb_m = moisture_blood_adj.get(q, 1.0)

        adjusted_gamma = base_gamma * t_adj * hm_adj * cush_m * mb_m
        gamma_list.append({
            "hn": h["horse_number"],
            "gamma": adjusted_gamma,
            "base_gamma": base_gamma,
            "tier_m": t_adj,
            "hm_m": hm_adj,
            "cush_m": cush_m,
            "mb_m": mb_m,
        })

    # レース内正規化 → 勝率
    total_gamma = sum(g["gamma"] for g in gamma_list)
    total_base = sum(g["base_gamma"] for g in gamma_list)

    result = {}
    for g in gamma_list:
        hn = g["hn"]
        if total_gamma > 0 and g["gamma"] > 0:
            win_prob = g["gamma"] / total_gamma
        else:
            win_prob = 0.0

        if total_base > 0 and g["base_gamma"] > 0:
            odds_win_prob = g["base_gamma"] / total_base
        else:
            odds_win_prob = 0.0

        # 複勝率: Harville model で top-3 入り確率
        head_count = len(gamma_list)
        gamma_values = [x["gamma"] for x in gamma_list]
        idx = gamma_list.index(g)
        place_prob = _approx_place_prob_harville(idx, gamma_values, min(3, head_count))

        result[hn] = {
            "win_prob": round(win_prob, 4),
            "place_prob": round(place_prob, 4),
            "odds_win_prob": round(odds_win_prob, 4),
        }

    return result


def _approx_place_prob_harville(target_idx, gammas, top_k=3):
    """Harville model でのtop-k入り確率を計算する。

    Args:
        target_idx: 対象馬のインデックス (gammasリスト内)
        gammas: 全馬のgammaリスト
        top_k: 上位何頭 (通常3 = 複勝)

    Returns:
        float: top-k入り確率
    """
    gi = gammas[target_idx]
    if gi <= 0:
        return 0.0

    n = len(gammas)
    if n <= top_k:
        return 1.0

    total = sum(gammas)
    if total <= 0:
        return 0.0

    # P(i finishes 1st)
    p1 = gi / total

    # P(i finishes 2nd) = sum_{j!=i} P(j 1st) * P(i | remaining)
    p2 = 0.0
    for j in range(n):
        if j == target_idx or gammas[j] <= 0:
            continue
        rem = total - gammas[j]
        if rem > 0:
            p2 += (gammas[j] / total) * (gi / rem)

    if top_k < 3:
        return min(1.0, max(0.0, p1 + p2))

    # P(i finishes 3rd) = sum_{j!=i, k!=i, j!=k} P(j 1st) * P(k 2nd|j) * P(i|j,k)
    p3 = 0.0
    for j in range(n):
        if j == target_idx or gammas[j] <= 0:
            continue
        pj = gammas[j] / total
        rem_j = total - gammas[j]
        if rem_j <= 0:
            continue
        for k in range(n):
            if k == target_idx or k == j or gammas[k] <= 0:
                continue
            pk_given_j = gammas[k] / rem_j
            rem_jk = rem_j - gammas[k]
            if rem_jk > 0:
                p3 += pj * pk_given_j * (gi / rem_jk)

    return min(1.0, max(0.0, p1 + p2 + p3))


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

    # --- 前走条件バイアス計算用 ---
    import sqlite3 as _sqlite3_ctx
    _db_path_ctx = KEIBA_AI_DIR / "keiba.db"

    # 含水率ゾーン定義
    _TURF_MOISTURE_ZONES = {"low": (0, 10.5), "mid": (10.5, 13.7), "high": (13.7, 100)}
    _DIRT_MOISTURE_ZONES = {"low": (0, 3.1), "mid": (3.1, 8.85), "high": (8.85, 100)}

    def _classify_mz(raw_m, ct):
        zones = _TURF_MOISTURE_ZONES if ct == "芝" else _DIRT_MOISTURE_ZONES
        for zn, (lo, hi) in zones.items():
            if lo <= raw_m < hi:
                return zn
        return list(zones.keys())[-1]

    def _get_prev_race_info(horse_id, current_race_date):
        """馬の前走情報（race_id, venue_code, race_date, course_type）を取得"""
        conn = _sqlite3_ctx.connect(str(_db_path_ctx))
        try:
            row = conn.execute("""
                SELECT r.race_id, r.venue_code, r.race_date, r.course_type
                FROM race_results rr
                JOIN races r ON rr.race_id = r.race_id
                WHERE rr.horse_id = ? AND r.race_date < ?
                ORDER BY r.race_date DESC, r.race_id DESC
                LIMIT 1
            """, (horse_id, str(current_race_date))).fetchone()
            if row:
                return {"race_id": row[0], "venue_code": row[1],
                        "race_date": row[2], "course_type": row[3]}
        finally:
            conn.close()
        return None

    def _get_cv_data(venue_code, race_date_str):
        """cushion_valuesテーブルからクッション値・含水率を取得"""
        date_nd = str(race_date_str).replace("-", "")
        conn = _sqlite3_ctx.connect(str(_db_path_ctx))
        try:
            row = conn.execute(
                "SELECT cushion_value, turf_moisture_goal, turf_moisture_corner, "
                "dirt_moisture_goal, dirt_moisture_corner "
                "FROM cushion_values WHERE venue_code=? AND race_date=?",
                (venue_code, date_nd)).fetchone()
            if row:
                return {"cushion": row[0],
                        "turf_moisture_goal": row[1], "turf_moisture_corner": row[2],
                        "dirt_moisture_goal": row[3], "dirt_moisture_corner": row[4]}
        finally:
            conn.close()
        return None

    def _compute_context_bias_ratio(horse_id, sire_id, ct, venue_code, target_date):
        """前走条件バイアス比率を計算する。
        Returns: dict with moisture_ratio, cushion_ratio, combined, prev_info or None
        """
        if not horse_id or not sire_id:
            return None

        prev = _get_prev_race_info(horse_id, target_date)
        if not prev:
            return None

        prev_cv = _get_cv_data(prev["venue_code"], prev["race_date"])
        curr_cv = _get_cv_data(venue_code, str(target_date))
        if not prev_cv or not curr_cv:
            return None

        prev_ct = prev.get("course_type", ct)

        # 含水率比率
        moisture_ratio = 1.0
        prev_mz_label = None
        curr_mz_label = None

        if prev_ct == "芝":
            pg, pc = prev_cv.get("turf_moisture_goal"), prev_cv.get("turf_moisture_corner")
        else:
            pg, pc = prev_cv.get("dirt_moisture_goal"), prev_cv.get("dirt_moisture_corner")

        if ct == "芝":
            cg, cc = curr_cv.get("turf_moisture_goal"), curr_cv.get("turf_moisture_corner")
        else:
            cg, cc = curr_cv.get("dirt_moisture_goal"), curr_cv.get("dirt_moisture_corner")

        if (pg is not None or pc is not None) and (cg is not None or cc is not None):
            prev_raw = ((pg or 0) + (pc or 0)) / (2 if pg is not None and pc is not None else 1)
            curr_raw = ((cg or 0) + (cc or 0)) / (2 if cg is not None and cc is not None else 1)
            prev_mz = _classify_mz(prev_raw, prev_ct)
            curr_mz = _classify_mz(curr_raw, ct)
            prev_mz_label = prev_mz
            curr_mz_label = curr_mz

            if moisture_blood_roi_data:
                from scripts.moisture_blood_roi import lookup_moisture_blood_roi
                prev_mb = lookup_moisture_blood_roi(moisture_blood_roi_data, sire_id, prev_ct, prev_raw)
                curr_mb = lookup_moisture_blood_roi(moisture_blood_roi_data, sire_id, ct, curr_raw)
                prev_roi = (prev_mb["roi"] if prev_mb and prev_mb.get("roi") else None) or 100.0
                curr_roi = (curr_mb["roi"] if curr_mb and curr_mb.get("roi") else None) or 100.0
                if prev_roi > 0:
                    moisture_ratio = curr_roi / prev_roi

        # クッション比率（芝→芝のみ）
        cushion_ratio = 1.0
        if ct == "芝" and prev_ct == "芝":
            prev_cush = prev_cv.get("cushion")
            curr_cush = curr_cv.get("cushion")
            if prev_cush is not None and curr_cush is not None:
                prev_cush_res = cushion_analyzer.compute_cushion_roi(horse_id, prev_cush)
                curr_cush_res = cushion_analyzer.compute_cushion_roi(horse_id, curr_cush)
                prev_c_roi = (prev_cush_res.get("roi") if prev_cush_res else None) or 100.0
                curr_c_roi = (curr_cush_res.get("roi") if curr_cush_res else None) or 100.0
                if prev_c_roi > 0 and curr_c_roi is not None:
                    cushion_ratio = curr_c_roi / prev_c_roi

        combined = moisture_ratio * cushion_ratio
        # クリップ
        combined = max(0.5, min(2.0, combined))

        return {
            "moisture_ratio": round(moisture_ratio, 3),
            "cushion_ratio": round(cushion_ratio, 3),
            "combined": round(combined, 3),
            "prev_race_date": str(prev["race_date"]),
            "prev_moisture_zone": prev_mz_label,
            "curr_moisture_zone": curr_mz_label,
        }

    from analysis.heatmap_scorer import HeatmapScorer
    heatmap_scorer = HeatmapScorer()
    heatmap_scorer.load()

    # --- 含水率×血統ROI計算用 ---
    import numpy as np
    from scripts.moisture_blood_roi import lookup_moisture_blood_roi

    moisture_blood_roi_data = None
    _mb_roi_path = KEIBA_AI_DIR / "config" / "moisture_blood_roi.json"
    try:
        with open(_mb_roi_path, "r", encoding="utf-8") as _f:
            moisture_blood_roi_data = json.load(_f)
        _meta = moisture_blood_roi_data.get("metadata", {})
        print(f"  moisture_blood_roi loaded: cross={_meta.get('n_cross', 0)}, "
              f"sire_fb={_meta.get('n_fallback_sire', 0)}, "
              f"zone_fb={_meta.get('n_fallback_zone', 0)}")
    except Exception as e:
        print(f"  WARNING: moisture_blood_roi.json load failed: {e}")

    def _get_moisture_for_venue(venue_code, race_date_str):
        """CushionValueテーブルから含水率を取得"""
        import sqlite3
        db_path = KEIBA_AI_DIR / "keiba.db"
        date_nodash = str(race_date_str).replace("-", "")
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(
                "SELECT turf_moisture_goal, turf_moisture_corner, "
                "dirt_moisture_goal, dirt_moisture_corner "
                "FROM cushion_values WHERE venue_code = ? AND race_date = ?",
                (venue_code, date_nodash),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "turf_moisture_goal": row[0],
                    "turf_moisture_corner": row[1],
                    "dirt_moisture_goal": row[2],
                    "dirt_moisture_corner": row[3],
                }
        except Exception as e:
            print(f"  WARNING: moisture query failed: {e}")
        return None

    def _compute_moisture_blood_roi(venue_code, race_date_str, course_type, horse_id):
        """含水率×血統ROIを計算する（馬ごと）"""
        if not moisture_blood_roi_data or not horse_id:
            return None

        moist_data = _get_moisture_for_venue(venue_code, race_date_str)
        if not moist_data:
            return None

        # 含水率raw = (goal + corner) / 2
        if course_type == "芝":
            g = moist_data.get("turf_moisture_goal")
            c = moist_data.get("turf_moisture_corner")
        else:
            g = moist_data.get("dirt_moisture_goal")
            c = moist_data.get("dirt_moisture_corner")

        if g is None and c is None:
            return None
        raw_moisture = ((g or 0) + (c or 0)) / (2 if (g is not None and c is not None) else 1)

        # 馬のsire_idを取得
        from database.models import Horse
        sire_id = None
        with get_session() as s:
            horse = s.query(Horse.sire_id).filter_by(horse_id=horse_id).first()
            if horse:
                sire_id = horse.sire_id
        if not sire_id:
            return None

        result = lookup_moisture_blood_roi(
            moisture_blood_roi_data, sire_id, course_type, raw_moisture)
        if result:
            return result
        return None

    # 含水率キャッシュ（会場ごとに1回のみ取得 — moisture_for_venue用）
    _moisture_venue_cache = {}

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

    # 勝率補正係数読み込み
    win_adj = _load_win_prob_adjustments()

    # ヒートマップROI五分位境界を事前計算
    # compute_unified_roi のサンプルから分布を推定
    heatmap_quintile_edges = None
    if win_adj and heatmap_scorer._loaded and heatmap_scorer._df is not None:
        import numpy as np
        try:
            # 各馬の最新レースから unified_roi をサンプル計算
            _df = heatmap_scorer._df
            _latest = _df.sort_values("race_date", ascending=False).drop_duplicates(
                subset="horse_id", keep="first")
            # 最大1000頭をサンプル
            if len(_latest) > 1000:
                _latest = _latest.sample(1000, random_state=42)

            _sample_rois = []
            for _, row in _latest.iterrows():
                hid = row.get("horse_id")
                ct = row.get("course_type", "芝")
                # 簡易計算: 代表的な距離・馬場で
                result = heatmap_scorer.compute_unified_roi(hid, ct, 1800, "良")
                uroi = result.get("unified_roi")
                if uroi is not None:
                    _sample_rois.append(uroi)

            if len(_sample_rois) >= 100:
                heatmap_quintile_edges = [
                    float(np.percentile(_sample_rois, q)) for q in [20, 40, 60, 80]
                ]
                print(f"  heatmap quintile edges: {heatmap_quintile_edges}")
                print(f"  (based on {len(_sample_rois)} horse samples)")
        except Exception as e:
            print(f"  WARNING: heatmap quintile edges computation failed: {e}")
            heatmap_quintile_edges = None

    # クッション値ROI / 含水率×血統ROI の五分位境界を事前計算
    # 過去データのROI分布から境界を取得
    _cushion_quintile_edges = None
    _mb_quintile_edges = None
    if win_adj:
        import numpy as np
        import sqlite3 as _sqlite3
        _db_path = KEIBA_AI_DIR / "keiba.db"
        try:
            _conn = _sqlite3.connect(str(_db_path))
            # クッション値: sire x zone 統計のROI分布からedges
            _cush_q = _conn.execute("""
                SELECT cv.cushion_value, h.sire_id
                FROM cushion_values cv
                JOIN races r ON cv.venue_code = r.venue_code
                            AND cv.race_date = REPLACE(r.race_date, '-', '')
                JOIN race_results rr ON r.race_id = rr.race_id
                JOIN horses h ON rr.horse_id = h.horse_id
                WHERE r.course_type = '芝'
                  AND rr.finish_order IS NOT NULL AND rr.finish_order > 0
                  AND h.sire_id IS NOT NULL
                  AND r.race_date >= '2024-01-01'
            """).fetchall()

            if _cush_q:
                # 三分位境界
                _cv_vals = [r[0] for r in _cush_q if r[0] is not None]
                if len(_cv_vals) > 100:
                    _tercile = [np.percentile(_cv_vals, 33.3), np.percentile(_cv_vals, 66.7)]

                    def _cz(cv):
                        if cv <= _tercile[0]: return "hard"
                        elif cv <= _tercile[1]: return "mid"
                        else: return "soft"

                    # sire x zone ROI 統計
                    _pay_q = _conn.execute(
                        "SELECT race_id, horse_numbers, payout FROM payoffs WHERE bet_type='複勝'"
                    ).fetchall()
                    _pay_map = {}
                    for _pr in _pay_q:
                        try: _pay_map[(_pr[0], int(_pr[1]))] = _pr[2]
                        except: pass

                    _sire_zone_acc = {}
                    for _row in _cush_q:
                        _cv, _sid = _row
                        if _cv is None or _sid is None: continue
                        _z = _cz(_cv)
                        _k = (_sid, _z)
                        if _k not in _sire_zone_acc:
                            _sire_zone_acc[_k] = {"n": 0, "p": 0}
                        _sire_zone_acc[_k]["n"] += 1

                    # sire x zone ROI 分布
                    _cush_rois = []
                    for _k, _v in _sire_zone_acc.items():
                        if _v["n"] >= 30:
                            _cush_rois.append(_v["n"])  # placeholder - use actual from calibration

                    # fallback: 直接calibrationのROI分布を使う
                    # cushion_analysis の統計からROI値を取得
                    _cush_stats = cushion_analyzer._sire_stats
                    if _cush_stats:
                        _cush_rois = [s["roi"] for s in _cush_stats.values()]
                    if len(_cush_rois) >= 20:
                        _cushion_quintile_edges = [
                            float(np.percentile(_cush_rois, q)) for q in [20, 40, 60, 80]
                        ]
                        print(f"  cushion ROI quintile edges: {_cushion_quintile_edges}")

            # 含水率x血統ROI: moisture_blood_roi.json のROI分布
            if moisture_blood_roi_data:
                _mb_rois = []
                for _ct_key in ["芝", "ダート"]:
                    _ct_data = moisture_blood_roi_data.get(_ct_key, {})
                    for _entry in _ct_data.values():
                        if isinstance(_entry, dict) and "roi" in _entry:
                            _mb_rois.append(_entry["roi"])
                if len(_mb_rois) >= 20:
                    _mb_quintile_edges = [
                        float(np.percentile(_mb_rois, q)) for q in [20, 40, 60, 80]
                    ]
                    print(f"  moisture_blood ROI quintile edges: {_mb_quintile_edges}")

            _conn.close()
        except Exception as e:
            print(f"  WARNING: quintile edges computation failed: {e}")

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

            # 含水率×血統ROI（馬ごと: sire_id × moisture_zone）
            venue_code = race.get("venue_code", "")

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

                # ANDフィルタティア判定（推奨判定の前に実施）
                and_filter_tier = None
                and_filter_roi = None
                if horse_id:
                    tier_result = heatmap_scorer.classify_horse(horse_id, ct)
                    and_filter_tier = tier_result.get("tier")
                    and_filter_roi = tier_result.get("oos_roi")

                # 推奨判定（ANDフィルタティア + DSGS ROI + composite_rank ベース）
                recommendation = ""

                # ANDフィルタティアによる推奨
                if and_filter_tier:
                    tier_num = int(and_filter_tier.replace("%", ""))
                    if tier_num >= 110:
                        recommendation = "買"
                    elif tier_num <= 70:
                        recommendation = "避"

                # ティアで判定されなかった場合、DSGSベースで補完
                if not recommendation and ct in ("ダート", "芝") and horse_id:
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
                # === Group A: 馬の力 (rank_dev系) ===
                base_roi_str = "-"
                pattern_roi_str = "-"
                accumulation_roi_str = "-"
                roll5_roi_str = "-"
                integrated_roi_str = "-"

                # === Group B: レース条件系 ===
                rel_lap_roi_str = "-"
                margin_roi_str = "-"
                accel_roi_str = "-"
                cushion_roi_str = "-"
                dsgs_roi_str = "-"
                pace_advantage_str = ""
                _cushion_roi_val = None
                _moisture_blood_roi_val = None

                if horse_id:
                    # 0. ベースROI（前走rank_devデシルOOS ROI — 全馬に値が出る）
                    raw_feats = raw_scorer.get_horse_raw_features(horse_id, ct)
                    rd_info = raw_feats.get("rank_dev")
                    if rd_info and rd_info.get("oos_roi") is not None:
                        base_roi_str = f"{rd_info['oos_roi']:.0f}%"
                    else:
                        # 同一馬場のデータなし → 別馬場からフォールバック
                        other_ct = "芝" if ct == "ダート" else "ダート"
                        raw_other = raw_scorer.get_horse_raw_features(
                            horse_id, other_ct)
                        rd_other = raw_other.get("rank_dev")
                        if rd_other and rd_other.get("oos_roi") is not None:
                            ct_label = "芝" if other_ct == "芝" else "ダ"
                            base_roi_str = (
                                f"({rd_other['oos_roi']:.0f}%{ct_label})")

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
                    else:
                        # 別馬場の5走ROIも取得
                        other_ct = "芝" if ct == "ダート" else "ダート"
                        mr_other = raw_scorer.get_horse_multi_race_features(
                            horse_id, other_ct)
                        if mr_other and mr_other.get("roll5_roi") is not None:
                            ct_label = "芝" if other_ct == "芝" else "ダ"
                            roll5_roi_str = (
                                f"({mr_other['roll5_roi']:.0f}%{ct_label})")

                    # 4. 統合ROI（後方互換）
                    ir_info = raw_scorer.get_integrated_roi(horse_id, ct)
                    if ir_info["roi"] is not None:
                        integrated_roi_str = f"{ir_info['roi']:.0f}%"

                    # === レース条件系ROI ===
                    # raw_feats は上で既に取得済み

                    # 5. 相対速度ROI (rel_lap)
                    rl_info = raw_feats.get("rel_lap")
                    if rl_info and rl_info.get("oos_roi") is not None:
                        rel_lap_roi_str = f"{rl_info['oos_roi']:.0f}%"

                    # 6. 着差率ROI (margin)
                    mg_info = raw_feats.get("margin")
                    if mg_info and mg_info.get("oos_roi") is not None:
                        margin_roi_str = f"{mg_info['oos_roi']:.0f}%"

                    # 7. 加減速ROI (accel_raw)
                    ac_info = raw_feats.get("accel_raw")
                    if ac_info and ac_info.get("oos_roi") is not None:
                        accel_roi_str = f"{ac_info['oos_roi']:.0f}%"

                    # 8. クッション値適性ROI（芝のみ）
                    if ct == "芝" and race_cushion_cv is not None:
                        cush_result = cushion_analyzer.compute_cushion_roi(
                            horse_id, race_cushion_cv)
                        if cush_result and cush_result.get("roi") is not None:
                            cushion_roi_str = f"{cush_result['roi']:.0f}%"
                            _cushion_roi_val = cush_result["roi"]

                    # 9. DSGS ROI (dims_passed ベース表示)
                    if ct in ("ダート", "芝"):
                        dsgs_roi_info = dsgs_scorer.get_horse_dsgs_roi(
                            horse_id, race_id, ct, combo_chains_df)
                        if dsgs_roi_info:
                            dims_passed = dsgs_roi_info.get("dims_passed", 0)
                            total_dims = len([
                                d for d in dsgs_roi_info.get(
                                    "dim_details", {}).values()
                                if d.get("threshold", 100) < 100
                            ])
                            if total_dims == 0:
                                total_dims = 4  # fallback
                            if dsgs_roi_info.get("max_roi") is not None:
                                # 全次元パス: CSV実績ROIを表示
                                dsgs_roi_str = (
                                    f"{dsgs_roi_info['max_roi']:.0f}%"
                                    f"({dims_passed}/{total_dims})"
                                )
                            elif dims_passed > 0:
                                # 部分パス: パス数/全次元数を表示
                                dsgs_roi_str = f"{dims_passed}/{total_dims}"
                            else:
                                dsgs_roi_str = f"0/{total_dims}"
                        else:
                            dsgs_roi_str = "-"

                # 展開適性
                pace_advantage_str = pf_info.get("advantage", "")

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

                # 含水率×血統ROI（馬ごと）
                moisture_blood_roi_str = "-"
                if horse_id and moisture_blood_roi_data:
                    mb_result = _compute_moisture_blood_roi(
                        venue_code, str(target_date), ct, horse_id)
                    if mb_result:
                        moisture_blood_roi_str = f"{mb_result['roi']:.0f}%"
                        _moisture_blood_roi_val = mb_result["roi"]

                # 前走条件バイアス比率
                context_bias_str = "-"
                context_bias_detail = None
                if horse_id:
                    # sire_id取得
                    _sire_id_for_ctx = None
                    try:
                        from database.models import Horse as _HorseCtx
                        with get_session() as _s_ctx:
                            _h_ctx = _s_ctx.query(_HorseCtx.sire_id).filter_by(
                                horse_id=horse_id).first()
                            if _h_ctx:
                                _sire_id_for_ctx = _h_ctx.sire_id
                    except Exception:
                        pass

                    if _sire_id_for_ctx:
                        ctx_result = _compute_context_bias_ratio(
                            horse_id, _sire_id_for_ctx, ct, venue_code, target_date)
                        if ctx_result:
                            cb = ctx_result["combined"]
                            if cb < 0.85:
                                context_bias_str = f"{cb:.2f} (不利転換)"
                            elif cb > 1.15:
                                context_bias_str = f"{cb:.2f} (有利転換)"
                            elif cb != 1.0:
                                context_bias_str = f"{cb:.2f}"
                            context_bias_detail = {
                                "combined": ctx_result["combined"],
                                "moisture_ratio": ctx_result["moisture_ratio"],
                                "cushion_ratio": ctx_result["cushion_ratio"],
                                "prev_race_date": ctx_result["prev_race_date"],
                            }

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
                    # ANDフィルタティア（全特徴量統合）
                    "and_filter_tier": and_filter_tier,
                    "and_filter_roi": and_filter_roi,
                    # Group A: 馬の力 (rank_dev系)
                    "base_roi": base_roi_str,
                    "pattern_roi": pattern_roi_str,
                    "accumulation_roi": accumulation_roi_str,
                    "roll5_roi": roll5_roi_str,
                    "integrated_roi": integrated_roi_str,
                    # Group B: レース条件系
                    "blood_roi": blood_roi_str,
                    "rel_lap_roi": rel_lap_roi_str,
                    "margin_roi": margin_roi_str,
                    "accel_roi": accel_roi_str,
                    "cushion_roi": cushion_roi_str,
                    "moisture_blood_roi": moisture_blood_roi_str,
                    "dsgs_roi": dsgs_roi_str,
                    "context_bias": context_bias_str,
                    "context_bias_detail": context_bias_detail,
                    "pace_advantage": pace_advantage_str,
                    # 共通
                    "sire_name": sire_name,
                    "running_style": pf_info.get("running_style",
                                                  e.get("running_style", "")),
                    "jockey_name": e.get("jockey_name", ""),
                    "signal_tags": signal_tags,
                    # 勝率計算用（内部用、出力時に削除）
                    "_horse_id": horse_id,
                    "_cushion_roi_val": _cushion_roi_val,
                    "_moisture_blood_roi_val": _moisture_blood_roi_val,
                }
                horses_output.append(horse_data)

            # 勝率・複勝率を計算してマージ
            if win_adj and horses_output:
                # cushion_roi / moisture_blood_roi の馬番別マップ
                _cush_map = {}
                _mb_map = {}
                for h in horses_output:
                    _hn = h["horse_number"]
                    if h.get("_cushion_roi_val") is not None:
                        _cush_map[_hn] = h["_cushion_roi_val"]
                    if h.get("_moisture_blood_roi_val") is not None:
                        _mb_map[_hn] = h["_moisture_blood_roi_val"]

                win_probs = _compute_race_win_probs(
                    horses_output, win_adj, heatmap_scorer,
                    ct, dist, track_cond, heatmap_quintile_edges,
                    cushion_roi_map=_cush_map,
                    moisture_blood_roi_map=_mb_map,
                    cushion_quintile_edges=_cushion_quintile_edges,
                    moisture_blood_quintile_edges=_mb_quintile_edges)
                for h in horses_output:
                    hn_key = h["horse_number"]
                    wp = win_probs.get(hn_key, {})
                    h["win_prob"] = wp.get("win_prob", 0.0)
                    h["place_prob"] = wp.get("place_prob", 0.0)
                    h["odds_win_prob"] = wp.get("odds_win_prob", 0.0)
                    # 内部フィールドを削除（公開データに含めない）
                    h.pop("_horse_id", None)
                    h.pop("_cushion_roi_val", None)
                    h.pop("_moisture_blood_roi_val", None)
            else:
                for h in horses_output:
                    h.pop("_horse_id", None)
                    h.pop("_cushion_roi_val", None)
                    h.pop("_moisture_blood_roi_val", None)

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
