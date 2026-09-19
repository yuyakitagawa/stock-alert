import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sklearn バージョン互換パッチ（1.9.0で保存したモデルを1.5.x で読む際に multi_class が欠落する）
from sklearn.linear_model import LogisticRegression as _LR
if not hasattr(_LR, "_multi_class_patched"):
    _orig_lr_pp = _LR.predict_proba
    def _lr_pp(self, X):
        if not hasattr(self, "multi_class"):
            self.multi_class = "auto"
        return _orig_lr_pp(self, X)
    _LR.predict_proba = _lr_pp
    _LR._multi_class_patched = True

import threading
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from lib.utils import get_prices, get_nikkei_returns, extract_features, add_cs_rank_features, get_fundamentals, sell_label, classify_market_regime, get_market_index_df_cached
from config import BASE_DIR, BEAR_MARKET_THRESHOLD, MARKET_TIMING_20D_THRESH
from core.screener import get_tse_stock_list


def main():
    # 全銘柄の終値を舐めるのでローカルミラーを使う。RESTで1銘柄1リクエストを
    # 続けると1回52MBのegressになりSupabase Free枠(5GB/月)を食い潰す。
    from lib import price_store
    price_store.enable()
    print("=" * 55)
    print("スクリーナー × RF ランキング  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"スクリーナー通過銘柄に下落確率スコアをつけてランキング")
    print("=" * 55)

    # モデル読み込み（下落モデルのみ。上昇モデルはLINE Bot等で未使用のため廃止）
    drop_path = os.path.join(BASE_DIR, "rf_drop_model.pkl")
    if not os.path.exists(drop_path):
        print("ERROR: rf_drop_model.pkl が見つかりません。先に rf_train_v3.py を実行してください")
        return
    drop_model = joblib.load(drop_path)
    print(f"\n下落モデル読み込み: {drop_path}")

    # 日経225リターン取得 + 相場レジーム判定
    print("\n日経225リターン取得中...")
    nk5, nk20, nk60 = get_nikkei_returns()
    is_bear = nk20 is not None and nk20 < BEAR_MARKET_THRESHOLD

    # 相場レジーム判定（SMA63/200ベース）
    regime = 'uncertain'
    try:
        _nk_regime_df = get_market_index_df_cached("N225", "%5EN225", 400)
        if _nk_regime_df is not None and len(_nk_regime_df) >= 200:
            _closes = _nk_regime_df["Close"].tolist()
            regime = classify_market_regime(_closes)
    except Exception:
        pass

    # レジーム別 動的銘柄数（Daniel & Moskowitz 2016: 強気時は拡大、弱気時は縮小）
    regime_top_n = {'bull': 10, 'uncertain': 5, 'bear': 3}
    dynamic_top_n = regime_top_n.get(regime, 5)

    # ── VIX恐怖指数・S&P500・USD/JPY（クロスアセット）取得 ─────────────────────
    print("\nマクロデータ取得中（VIX・S&P500・USD/JPY）...")
    _live_macro = {"vix": None, "us5": None, "us20": None, "us60": None}
    _live_jpy   = {"jpy5": None, "usdjpy_closes": None}
    try:
        _vix_df = get_market_index_df_cached("VIX",    "%5EVIX",    days=60)
        if _vix_df is not None and len(_vix_df) > 0:
            _live_macro["vix"] = float(_vix_df["Close"].iloc[-1])
            print(f"  VIX: {_live_macro['vix']:.1f}")
        _sp5_df = get_market_index_df_cached("SP500",  "%5EGSPC",   days=100)
        if _sp5_df is not None and len(_sp5_df) >= 21:
            _p = _sp5_df["Close"].values
            _live_macro["us5"]  = round((_p[-1] - _p[-6])  / _p[-6]  * 100, 2) if len(_p) >= 6  else 0.0
            _live_macro["us20"] = round((_p[-1] - _p[-21]) / _p[-21] * 100, 2) if len(_p) >= 21 else 0.0
            _live_macro["us60"] = round((_p[-1] - _p[-61]) / _p[-61] * 100, 2) if len(_p) >= 61 else None
            print(f"  S&P500: 5日{_live_macro['us5']:+.2f}% / 20日{_live_macro['us20']:+.2f}%")
        _jpy_df = get_market_index_df_cached("USDJPY", "USDJPY%3DX", days=120)
        if _jpy_df is not None and len(_jpy_df) >= 6:
            _jc = _jpy_df["Close"].values
            _live_jpy["jpy5"] = round((_jc[-1] - _jc[-6]) / _jc[-6] * 100, 2) if len(_jc) >= 6 else 0.0
            _live_jpy["usdjpy_closes"] = _jc
            print(f"  USD/JPY: 5日{_live_jpy['jpy5']:+.2f}%")
    except Exception as _e:
        print(f"  マクロデータ取得失敗: {_e}")

    # VIX レジーム調整: VIX > 30 は恐怖相場 → top_n を -1（最小1）
    vix_val = _live_macro.get("vix")
    if vix_val is not None and vix_val > 30:
        dynamic_top_n = max(1, dynamic_top_n - 1)
        print(f"  ⚠️ 高VIX({vix_val:.1f} > 30): 推奨銘柄数を {dynamic_top_n + 1}→{dynamic_top_n}に縮小")

    if nk5 is not None:
        print(f"  日経225: 5日{nk5:+.2f}% / 20日{nk20:+.2f}% / 60日{nk60:+.2f}%")
        regime_label = {'bull': '📈強気', 'bear': '📉弱気', 'uncertain': '🔶中立'}.get(regime, regime)
        print(f"  相場レジーム: {regime_label}  →  推奨銘柄数: {dynamic_top_n}銘柄")
        if is_bear:
            print(f"  ⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）")
    else:
        print("  日経225: 取得失敗（相対リターンはN/A）")
        is_bear = False
        dynamic_top_n = 5

    # ── 市場状況の警告のみ（停止しない — 最終判断はユーザーが行う）
    if nk20 is not None and nk20 < MARKET_TIMING_20D_THRESH:
        print(f"\n⚠️ 下落注意（日経20日: {nk20:+.1f}%）— 相場判断はご自身で。シグナルは継続出力します。")

    # 全TSE銘柄リスト取得（JPX直読み）
    stock_list = get_tse_stock_list()
    if stock_list is None:
        # ここで return するとワークフローが緑のまま成果物ゼロで終わる（実例: 2026-09-03〜04、
        # JPXが data_j.xls を .xlsx に差し替えて404になり、2営業日ぶん気づけなかった）。
        print("ERROR: 銘柄リスト取得失敗")
        sys.exit(1)
    codes = stock_list["code"].tolist()
    names = dict(zip(stock_list["code"], stock_list["name"]))
    print(f"全銘柄スキャン: {len(codes)} 銘柄")
    print(f"\n確率スコア計算中（並列処理）...")

    # フェーズ1: 全銘柄の特徴量を収集（並列）
    nk_rets = (nk5/100, nk20/100, nk60/100) if nk5 is not None else None
    raw_data = []
    lock = threading.Lock()
    done_count = [0]
    fund_map = {}   # code -> {"PER","PBR","ROE"}（全銘柄、pit eps/bps から算出）
    total = len(codes)

    def fetch_one(code, _macro=_live_macro, _jpy=_live_jpy):
        from datetime import date as _date
        prices = get_prices(code, days=400)
        with lock:
            done_count[0] += 1
            if done_count[0] % 500 == 0:
                print(f"  {done_count[0]}/{total} 取得済み... (有効: {len(raw_data)}銘柄)")
        if prices is None or len(prices) < 91:
            time.sleep(0.1)
            return None

        # ファンダメンタル取得（PER/PBR/ROE/決算まで日数/権利落ち後経過日数）
        from lib.fundamentals import get_pit_fundamentals as _get_pit
        fd_raw    = get_fundamentals(code)
        today     = _date.today()
        pit       = _get_pit(code, today) or {}
        per_live = fd_raw.get("PER"); pbr_live = fd_raw.get("PBR")
        # PER/PBR は J-Quants の eps/bps から算出。
        # ライブ取得(yfinance)は日本株でNoneが多いのでフォールバックに留める。
        from lib.fundamentals import get_pit_valuation as _get_val
        _val = _get_val(code, today)
        _close_px = float(prices["Close"].iloc[-1]) if len(prices) > 0 else None
        _eps = _val.get("eps"); _bps = _val.get("bps")
        per_calc = (round(_close_px / _eps, 1) if _eps and _eps > 0 and _close_px else per_live)
        pbr_calc = (round(_close_px / _bps, 2) if _bps and _bps > 0 and _close_px else pbr_live)
        roe_calc = fd_raw.get("ROE") if fd_raw.get("ROE") is not None else pit.get("roe")
        with lock:
            fund_map[code] = {
                "PER": per_calc, "PBR": pbr_calc, "ROE": roe_calc,
                "piotroski":    pit.get("piotroski"),
                "bps_growth":   pit.get("bps_growth"),
                "eps_surprise": pit.get("eps_surprise"),
            }
        # 配当利回り: ライブ株価 × PBR/PER から配当を逆算 or pit.dps使用
        _dps = pit.get("dps"); _close = prices["Close"].iloc[-1] if len(prices) > 0 else None
        div_yield_live = (_dps / _close * 100) if _dps and _dps > 0 and _close else None
        # USD/JPY ベータ（ライブ: 直近60日の株価 vs USD/JPY のベータ）
        _fx_beta_live = None
        _jpy_closes = _jpy.get("usdjpy_closes")
        if _jpy_closes is not None and len(prices) >= 61:
            p_arr_live = prices["Close"].values
            stock_rets_live = np.diff(p_arr_live[-61:]) / p_arr_live[-61:-1]
            fx_rets_live = np.diff(_jpy_closes[-min(61, len(_jpy_closes)):]) / _jpy_closes[-min(61, len(_jpy_closes)):-1]
            _ml = min(len(stock_rets_live), len(fx_rets_live))
            if _ml >= 20:
                _sr = stock_rets_live[:_ml]; _fr = fx_rets_live[:_ml]
                _vfx = np.var(_fr)
                if _vfx > 0:
                    _fx_beta_live = float(np.cov(_sr, _fr)[0, 1] / _vfx)

        fundamentals = {
            "per":                 per_live,
            "pbr":                 pbr_live,
            "roe":                 fd_raw.get("ROE"),
            "days_to_earnings":    None,
            "days_since_div_ex":   pit.get("days_since_div_ex"),
            "month":               today.month,
            "div_yield":           div_yield_live,
            "eps_growth":          pit.get("eps_growth"),
            "dps_growth":          pit.get("dps_growth"),
            # マクロ特徴量
            "vix":                 _macro.get("vix"),
            "us5":                 _macro.get("us5"),
            "us20":                _macro.get("us20"),
            # 新規IB特徴量
            "fx_beta":             _fx_beta_live,
            "jpy5":                _jpy.get("jpy5"),
            "eps_surprise":        pit.get("eps_surprise"),
            "bps_growth":          pit.get("bps_growth"),
            "piotroski":           pit.get("piotroski"),
            "payout":              pit.get("payout"),
            "accruals":            pit.get("accruals"),
        }

        feat = extract_features(
            prices["Close"].values,
            prices["Volume"].tolist() if "Volume" in prices.columns else None,
            nk_rets,
            fundamentals=fundamentals,
        )
        if feat is None:
            time.sleep(0.1)
            return None
        time.sleep(0.2)
        return (code, prices, feat)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_one, c): c for c in codes}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                with lock:
                    raw_data.append(result)

    print(f"有効銘柄: {len(raw_data)} 件")

    # フェーズ2: クロスセクショナルランク特徴量を付加（同日内での相対順位）
    if not raw_data:
        print("ERROR: 有効銘柄なし"); return
    feats_matrix = np.array([d[2] for d in raw_data], dtype=float)
    # 推論時: 全銘柄を同一日として扱い、セクター内相対モメンタムも計算
    from lib.utils import get_sector_cached as _gsc
    _sectors_for_batch = [_gsc(str(d[0])) for d in raw_data]
    feats_aug = add_cs_rank_features(feats_matrix, sectors=_sectors_for_batch)

    # フェーズ3: モデルスコア計算
    results = []
    for idx, (code, prices, feat) in enumerate(raw_data):
        feat_aug = feats_aug[idx]
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1])
        close = float(prices["Close"].iloc[-1])
        drop_pct = round(drop_prob * 100, 1)

        # ボラティリティ（feat[7] = vol20, 年率換算%）
        vol = round(feat[7], 1)
        if vol < 20:
            vol_label = "🟢低"
        elif vol < 40:
            vol_label = "🟡中"
        elif vol < 60:
            vol_label = "🟠高"
        else:
            vol_label = "🔴超高"

        # 下落確率判定（LINE Botの表示区分と統一: <8%安全圏 / 8-15%通常 / >=15%危険）
        if drop_pct < 8:
            judgment = "🟢安全圏  "
        elif drop_pct < 15:
            judgment = "🟡通常    "
        else:
            judgment = "🔴危険    "

        recommend = sell_label(
            drop_pct,
            drawdown60=float(feat[10]),
            down_streak_raw=round(feat[12] * 20),
        )

        # 日経比相対リターン
        p = prices["Close"].values
        s5  = (p[-1] - p[-6])  / p[-6]  * 100 if len(p) >= 6  else 0
        s20 = (p[-1] - p[-21]) / p[-21] * 100 if len(p) >= 21 else 0
        s60 = (p[-1] - p[-61]) / p[-61] * 100 if len(p) >= 61 else 0
        rel5  = round(s5  - nk5,  2) if nk5  is not None else None
        rel20 = round(s20 - nk20, 2) if nk20 is not None else None
        rel60 = round(s60 - nk60, 2) if nk60 is not None else None
        rels = [r for r in [rel5, rel20, rel60] if r is not None]
        rs_score = round(sum(rels) / len(rels), 2) if rels else None

        cs_vol20_rank = round(float(feat_aug[32]) * 100, 0)  # ボラティリティのCS相対ランク(0-100%)

        row = {
            "銘柄コード": code,
            "銘柄名": names.get(code, ""),
            "直近株価(円)": round(close, 1),
            "下落確率(%)": drop_pct,
            "判定": judgment,
            "ボラ(%)": vol,
            "ボラ水準": vol_label,
            "ボラランク(%)": cs_vol20_rank,
            "推奨": recommend,
            "日経比5日(%)": rel5 if rel5 is not None else "-",
            "日経比20日(%)": rel20 if rel20 is not None else "-",
            "日経比60日(%)": rel60 if rel60 is not None else "-",
            "相対強度": rs_score if rs_score is not None else "-",
            "PER": (fund_map.get(code) or {}).get("PER"),
            "PBR": (fund_map.get(code) or {}).get("PBR"),
            "ROE(%)": (fund_map.get(code) or {}).get("ROE"),
            "piotroski":    (fund_map.get(code) or {}).get("piotroski"),
            "bps_growth":   (fund_map.get(code) or {}).get("bps_growth"),
            "eps_surprise": (fund_map.get(code) or {}).get("eps_surprise"),
            "pos52":        round(float(feat[9]), 3),
        }
        results.append(row)

    # ランキング（下落確率が低い順）
    result_df = pd.DataFrame(results).sort_values("下落確率(%)", ascending=True).reset_index(drop=True)
    result_df.index += 1
    result_df.insert(0, "順位", result_df.index)

    # PER/PBR/ROE は全銘柄 fund_map（pit eps/bps 由来）で既に設定済み

    # 表示（動的銘柄数: レジームに応じて 3/5/10）
    print(f"\n{'='*90}")
    regime_label_disp = {'bull': '📈強気', 'bear': '📉弱気', 'uncertain': '🔶中立'}.get(regime, regime)
    print(f"上位{dynamic_top_n}銘柄ランキング [{regime_label_disp}レジーム]（下落確率が低い順）")
    if is_bear:
        print(f"⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）: モデルスコアの信頼性低下。")
    print(f"{'='*90}")
    print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<16}  {'株価':>8}  {'下落確率':>7}  {'判定':<12}  "
          f"{'PER':>6}  {'PBR':>5}  {'Gトレ':>5}  推奨")
    print("-" * 140)
    for _, row in result_df.head(dynamic_top_n).iterrows():
        per_val = row.get("PER"); pbr_val = row.get("PBR")
        per_str = f"{per_val:>5.1f}x" if per_val is not None else "   N/A"
        pbr_str = f"{pbr_val:>4.2f}x" if pbr_val is not None else "  N/A"

        # Googleトレンド
        gtr = row.get("Gトレンド", 0.0) or 0.0
        gtr_str = f"{'↑' if gtr > 0.2 else ('↓' if gtr < -0.1 else '→')}{gtr:+.1f}"

        print(
            f"{int(row['順位']):>4}  {row['銘柄コード']:>6}  "
            f"{str(row['銘柄名']):<16}  "
            f"{row['直近株価(円)']:>8,.0f}円  "
            f"{row['下落確率(%)']:>+6.1f}%  "
            f"{row['判定']:<12}  "
            f"{per_str}  {pbr_str}  "
            f"{gtr_str:>5}  "
            f"{row['推奨']}"
        )

    # フェーズ4b: オルタナティブデータ取得（上位20銘柄）
    ALT_TOP = min(20, len(result_df))
    print(f"\nオルタナティブデータ取得中（上位{ALT_TOP}銘柄）...")
    print("  対象: Googleトレンド")
    alt_results = {}
    alt_errors = 0
    try:
        from lib.alt_data import get_alt_signals

        def _fetch_alt(code, name):
            try:
                return str(code), get_alt_signals(str(code), str(name))
            except Exception:
                return str(code), {}

        with ThreadPoolExecutor(max_workers=5) as _exc:
            _futures = {
                _exc.submit(_fetch_alt, row["銘柄コード"], row["銘柄名"]): row["銘柄コード"]
                for _, row in result_df.head(ALT_TOP).iterrows()
            }
            for _f in as_completed(_futures):
                try:
                    _code, _data = _f.result()
                    alt_results[_code] = _data
                except Exception:
                    alt_errors += 1

        def _safe_get(code, key, default=None):
            return alt_results.get(str(code), {}).get(key, default)

        result_df["Gトレンド"]      = result_df["銘柄コード"].astype(str).map(lambda x: _safe_get(x, "trend_score", 0.0))

        print(f"  取得完了: {len(alt_results)}件 / エラー: {alt_errors}件")
    except Exception as _ae:
        print(f"  オルタナティブデータ取得エラー: {_ae}")
        result_df["Gトレンド"] = 0.0

    # フェーズ8: 相場リスク管制官 — マクロからリスクオン/オフを判定（情報表示のみ）
    from lib.risk_regime import assess as _assess_risk, summary_line as _risk_summary
    risk_verdict = _assess_risk(
        nk20=nk20,
        vix=_live_macro.get("vix"),
        jpy5=_live_jpy.get("jpy5"),
        us20=_live_macro.get("us20"),
        us5=_live_macro.get("us5"),
    )
    print(f"\n🛡️ 相場リスク管制官: {_risk_summary(risk_verdict)}")
    # 当日のリスク判定を保存（メール・Web・活動ログが参照）
    try:
        import json as _json
        from datetime import datetime as _dt
        _risk_out = {"date": _dt.now().strftime("%Y-%m-%d"), **risk_verdict}
        with open(os.path.join(BASE_DIR, "data", "risk_regime.json"), "w", encoding="utf-8") as _f:
            _json.dump(_risk_out, _f, ensure_ascii=False)
    except Exception as _e:
        print(f"  リスク判定の保存失敗（無視）: {_e}")

    # フェーズ8b: 日経 vs S&P500 相対強弱アドバイザー（情報表示のみ・シグナルには影響しない）
    from lib.market_compare import compare as _compare_markets, summary_line as _compare_summary
    market_verdict = _compare_markets(
        nk5=nk5, nk20=nk20, nk60=nk60,
        us5=_live_macro.get("us5"), us20=_live_macro.get("us20"), us60=_live_macro.get("us60"),
    )
    print(f"🌐 日経 vs S&P500: {_compare_summary(market_verdict)}")
    try:
        import json as _json
        _market_out = {"date": datetime.now().strftime("%Y-%m-%d"), **market_verdict}
        with open(os.path.join(BASE_DIR, "data", "market_compare.json"), "w", encoding="utf-8") as _f:
            _json.dump(_market_out, _f, ensure_ascii=False)
    except Exception as _e:
        print(f"  市場比較の保存失敗（無視）: {_e}")

    # CSV保存
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs(os.path.join(BASE_DIR, "data", "rankings"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "data", "rankings", f"ranking_{date_str}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全結果保存: {out_path}")

    # DB保存
    from lib.db import save_daily_ranking
    db_date_str = datetime.now().strftime("%Y-%m-%d")
    db_rows = [
        {
            "code": str(row["銘柄コード"]),
            "name": row["銘柄名"],
            "close": row["直近株価(円)"],
            "drop_prob": row["下落確率(%)"],
            "vol": row["ボラ(%)"],
            "recommend": row["推奨"],
            "rel20": row["日経比20日(%)"] if row["日経比20日(%)"] != "-" else None,
            "per":          row.get("PER"),
            "pbr":          row.get("PBR"),
            "piotroski":    row.get("piotroski"),
            "bps_growth":   row.get("bps_growth"),
            "eps_surprise": row.get("eps_surprise"),
            "pos52":        row.get("pos52"),
        }
        for _, row in result_df.iterrows()
    ]
    save_daily_ranking(db_date_str, db_rows)
    print(f"DB保存: {len(db_rows)}件 → stock_alert.db")

    # QA: 出力データの不変条件チェック（alert-only。違反でも処理は止めない）
    try:
        from lib.data_sanity import run_gate
        run_gate(db_rows, source="rank_stocks")
    except Exception as _e:
        print(f"[rank_stocks] QAチェックでエラー（無視して継続）: {_e}")

    print("完了")


if __name__ == "__main__":
    main()
