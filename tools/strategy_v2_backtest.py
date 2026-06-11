"""
strategy_v2_backtest.py — 「集中アルファ v2」戦略のイベント駆動バックテスト

設計（FM × アナリスト × コンサル）:
  目的: 複利成長(CAGR)の最大化（1億円到達）。集中3〜5銘柄・最大63日保有・標準リスク。
  核心: モデルは下落予測が得意(AUC0.79)。買いは「低drop_prob(落ちない確信)」主軸、
        売りは「drop_prob上昇」をトリガーにする。

買い（全ゲート通過 → net上位を上限まで・1セクター1銘柄・等加重・利益再投資）:
  コンサル相場ゲート: 日経 ≥ 63日SMA かつ 日経20日 ≥ -3%
  アナリスト銘柄ゲート: drop_prob ≤ 4 / rel20 ≥ 0 / vr2060 ≥ 1.0 /
                       piotroski ≥ 0.6 / accruals低 / vol20 ≤ 22
  FMリスクゲート: 株価 ≥ 300 / 売買代金 ≥ 50百万 / drawdown60 ≥ -15% / down_streak ≤ 3日
  → 通過銘柄を net(=rise-drop) 降順で上限まで採用

売り（先に来たもの）:
  1. drop_prob ≥ 8（モデルが危険検知）  2. net < 0（反転）
  3. トレーリングストップ -1.5×ATR近似  4. 63営業日到達（時間切れ）

【qv 戦略】Quality × Value（業績強 × 株価低迷）:
  買いゲート: Piotroski ≥ 0.67(6/9) / pos52 < 0.45(52週安値圏) / drawdown60 > -15%
             bps_growth > 0 または eps_surprise > 0 / dp ≤ 8 / vol20 ≤ 25
             相場ゲートなし（逆張りなので相場弱くても仕込む）
  ランク: net降順（モデルが最も回復を見込む銘柄を優先）
  売り: v2 と同じ（drop_prob 急騰 / net < 0 / トレール / 90日上限）

使い方:
  python3 tools/strategy_v2_backtest.py --start 2025-01-01           # v2（out-of-sample）
  python3 tools/strategy_v2_backtest.py --start 2025-01-01 --strategy baseline  # 現行Top10
  python3 tools/strategy_v2_backtest.py --bear                       # 2024/08暴落耐性
  python3 tools/strategy_v2_backtest.py --start 2025-01-01 --strategy qv        # QV戦略
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import date, timedelta

# backtest.py はモジュール読込時に独自 argparse を sys.argv に対して実行するため、
# 取り込み中だけ argv を退避して当スクリプトの引数と衝突させない。
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
import tools.backtest as bt   # ヘルパー再利用
sys.argv = _saved_argv
from lib.utils import add_cs_rank_features

BASE_DIR = bt.BASE_DIR

# ── 特徴量インデックス（add_cs_rank_features 後の60次元ベクトル） ─────────────
IDX_RET90        = 3
IDX_VOL20        = 7
IDX_POS52        = 9
IDX_DRAWDOWN60   = 10
IDX_DOWNSTREAK   = 12
IDX_VR2060       = 16
IDX_REL20        = 29
IDX_REL60        = 30
IDX_EPS_GROWTH   = 40
IDX_EPS_SURPRISE = 48
IDX_BPS_GROWTH   = 49
IDX_PIOTROSKI    = 50
IDX_PAYOUT       = 51
IDX_ACCRUALS     = 52

# ── 戦略パラメータ ──────────────────────────────────────────────────────────
MAX_POS          = 5      # 集中保有数（3〜5）
DECISION_STEP    = 10     # 意思決定グリッド（営業日）。買い候補スコアリング/売り判定の周期
HOLD_LIMIT_DAYS  = 63     # 最大保有営業日（モデルホライズン）
START_CAPITAL    = 1.0    # 相対（最終的に倍率で評価）
BIG_WIN          = 15.0   # 大勝ち閾値(%)

# 買いゲート閾値
BUY_DROP_MAX     = 4.0
BUY_VOL20_MAX    = 22.0
BUY_DRAWDOWN_MIN = -0.15
BUY_DOWNSTK_MAX  = 0.15   # 3日 = 3/20
BUY_REL20_MIN    = 0.0
BUY_VR2060_MIN   = 1.0
BUY_PIOTROSKI_MIN= 0.6
BUY_ACCRUALS_MAX = 0.10
MIN_PRICE        = 300.0
MIN_TURNOVER_M   = 50.0   # 20日平均売買代金(百万円)

# 売りゲート閾値
SELL_DROP_MIN    = 8.0
SELL_NET_MAX     = 0.0
TRAIL_ATR_MULT   = 1.5

# 相場ゲート
NK_SMA_DAYS      = 63
NK_20D_MIN       = -3.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2025-01-01")
    p.add_argument("--end",   type=str, default=date.today().isoformat())
    p.add_argument("--bear",  action="store_true", help="2024/08暴落期 (2024-07-01〜2024-10-01)")
    p.add_argument("--strategy", choices=["v2", "v3", "v4", "mom", "momnet", "momreg", "baseline", "qv"], default="v2")
    p.add_argument("--max-pos",  type=int, default=MAX_POS)
    p.add_argument("--hold-days", type=int, default=HOLD_LIMIT_DAYS, help="最大保有営業日（v3で勝ち伸ばし用に延長）")
    p.add_argument("--sector-cap", type=int, default=1, help="1セクター当たり最大保有数")
    p.add_argument("--limit",    type=int, default=0, help="母集団を先頭N銘柄に制限（スモーク用）")
    p.add_argument("--cached-only", action="store_true", help="価格キャッシュ済み銘柄のみ使用（高速）")
    return p.parse_args()


def load_models():
    rise = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
    drop_path = os.path.join(BASE_DIR, "rf_drop_model.pkl")
    drop = joblib.load(drop_path) if os.path.exists(drop_path) else None
    return rise, drop


def load_sector_map():
    """code -> 米国ETF（セクター代理）。1セクター1銘柄の重複排除に使用。"""
    path = os.path.join(BASE_DIR, "data", "sector_map.json")
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}


def build_close_series(hist):
    """hist -> (date配列, close配列, volume配列)（日付昇順）。"""
    c = hist["Close"].dropna()
    c.index = pd.to_datetime(c.index).date
    v = hist["Volume"].reindex(hist["Close"].dropna().index) if "Volume" in hist.columns else None
    dates = np.array(list(c.index))
    closes = c.values.astype(float)
    if "Volume" in hist.columns:
        vv = hist["Volume"]
        vv.index = pd.to_datetime(vv.index).date
        vol = pd.to_numeric(vv.reindex(c.index), errors="coerce").fillna(0).values.astype(float)
    else:
        vol = np.zeros(len(closes))
    return dates, closes, vol


def price_on_or_before(dates, closes, target):
    """target以前で直近のclose（無ければNone）。datesは昇順np配列。"""
    idx = np.searchsorted(dates, target, side="right") - 1
    if idx < 0:
        return None
    return float(closes[idx])


def main():
    args = parse_args()
    if args.bear:
        START = date(2024, 7, 1); END = date(2024, 10, 1)
    else:
        START = date.fromisoformat(args.start); END = date.fromisoformat(args.end)
    max_pos = args.max_pos
    # qv は mean-reversion に時間がかかるため保有上限を 90日に延長（未指定時のみ）
    if args.strategy == "qv" and args.hold_days == HOLD_LIMIT_DAYS:
        args.hold_days = 90

    # backtest.py のグローバル期間を合わせる（FETCH_DAYS等に影響）
    bt.BACKTEST_DATE = START
    bt.TODAY = END
    bt.FETCH_DAYS = max(800, (date.today() - START).days + 200)

    print("=" * 64)
    print(f"  集中アルファ v2 バックテスト [{args.strategy}]  {START} → {END}")
    print(f"  集中{max_pos}銘柄 / 最大{HOLD_LIMIT_DAYS}日保有 / 意思決定{DECISION_STEP}営業日ごと")
    print("=" * 64)

    rise_model, drop_model = load_models()
    sector_map = load_sector_map()
    print("モデル読み込み完了")

    # 日経データ
    nk_hist = bt.get_nikkei_prices()
    nk = nk_hist["Close"].dropna()
    nk.index = pd.to_datetime(nk.index).date
    nk_dates = np.array(list(nk.index)); nk_closes = nk.values.astype(float)
    trading_dates = sorted(d for d in nk.index if START <= d <= END)
    if len(trading_dates) < DECISION_STEP * 2:
        print("ERROR: 期間内営業日が不足"); return
    decision_dates = trading_dates[::DECISION_STEP]

    # 全銘柄の履歴を取得
    all_stocks = bt.fetch_tse_codes()
    if args.limit:
        all_stocks = all_stocks[:args.limit]
    # キャッシュ済み銘柄のみ（高速モード）
    cached_codes = None
    if args.cached_only:
        from lib.db import _conn
        with _conn() as con:
            cached_codes = {str(r[0]) for r in con.execute("SELECT DISTINCT code FROM price_cache")}
        all_stocks = [(c, n) for c, n in all_stocks if str(c) in cached_codes]
    print(f"株価データ取得中（{len(all_stocks)}銘柄）...")
    hist_map = {}
    for i, (code, name) in enumerate(all_stocks):
        h = bt.get_hist_for_features(code)
        if h is not None and len(h) >= 91:
            hist_map[code] = (h, name, *build_close_series(h))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(all_stocks)} ...")
        if not args.cached_only:
            time.sleep(0.03)
    print(f"有効銘柄: {len(hist_map)}")

    def nk_rets_at(d):
        i = np.searchsorted(nk_dates, d, side="right") - 1
        if i < 60:
            return (0.0, 0.0, 0.0)
        p = nk_closes[: i + 1]
        return ((p[-1]-p[-6])/p[-6], (p[-1]-p[-21])/p[-21], (p[-1]-p[-61])/p[-61])

    def market_gate_ok(d):
        i = np.searchsorted(nk_dates, d, side="right") - 1
        if i < NK_SMA_DAYS:
            return True
        p = nk_closes[: i + 1]
        sma = p[-NK_SMA_DAYS:].mean()
        ret20 = (p[-1] - p[-21]) / p[-21] * 100 if len(p) >= 21 else 0.0
        return p[-1] >= sma and ret20 >= NK_20D_MIN

    def turnover_m(dates, closes, vol, d):
        """d時点の20日平均売買代金（百万円）。"""
        i = np.searchsorted(dates, d, side="right") - 1
        if i < 20:
            return 0.0
        return float(np.mean(vol[i-19:i+1] * closes[i-19:i+1]) / 1e6)

    def score_universe(d):
        """d時点で買いゲート通過候補を [(code,name,net,drop,price,sector)] で返す（net降順）。"""
        nkr = nk_rets_at(d)
        raw_feats, meta = [], []
        for code, (h, name, dts, cls, vol) in hist_map.items():
            price = price_on_or_before(dts, cls, d)
            if price is None or price < MIN_PRICE:
                continue
            if turnover_m(dts, cls, vol, d) < MIN_TURNOVER_M:
                continue
            feat = bt.extract_features_at(h, d, nkr, code=code)
            if feat is None:
                continue
            raw_feats.append(feat); meta.append((code, name, price))
        if not raw_feats:
            return []
        fa = add_cs_rank_features(np.array(raw_feats, dtype=float))
        out = []
        for idx, (code, name, price) in enumerate(meta):
            f = fa[idx]
            if args.strategy == "v2":
                if f[IDX_VOL20] > BUY_VOL20_MAX: continue
                if f[IDX_DRAWDOWN60] < BUY_DRAWDOWN_MIN: continue
                if f[IDX_DOWNSTREAK] > BUY_DOWNSTK_MAX: continue
                if f[IDX_REL20] < BUY_REL20_MIN: continue
                if f[IDX_VR2060] < BUY_VR2060_MIN: continue
                if f[IDX_PIOTROSKI] < BUY_PIOTROSKI_MIN: continue
                if f[IDX_ACCRUALS] > BUY_ACCRUALS_MAX: continue
            elif args.strategy in ("v3", "v4"):
                # 最小ゲート: 急落中だけ除外（常時フル投資でnet上位を埋める）
                if f[IDX_DRAWDOWN60] < BUY_DRAWDOWN_MIN: continue
                if f[IDX_DOWNSTREAK] > BUY_DOWNSTK_MAX: continue
            elif args.strategy in ("mom", "momnet", "momreg"):
                # モメンタム: 強い銘柄を素直に買う。急落中だけ除外し、高値圏(pos52)を要求
                if f[IDX_DOWNSTREAK] > BUY_DOWNSTK_MAX: continue
                if f[IDX_POS52] < 0.6: continue          # 52週レンジ上位60%以上（高値圏）
                if f[IDX_RET90] <= 0: continue            # 3ヶ月で上昇しているもの
            elif args.strategy == "qv":
                # Quality × Value: 業績強 × 株価低迷
                if f[IDX_VOL20] > 25.0: continue         # 過度なボラ除外
                if f[IDX_DRAWDOWN60] < BUY_DRAWDOWN_MIN: continue   # フリーフォール除外
                if f[IDX_DOWNSTREAK] > BUY_DOWNSTK_MAX: continue    # 連続急落除外
                if f[IDX_PIOTROSKI] < 0.67: continue     # 財務健全（6/9以上）
                if f[IDX_POS52] >= 0.45: continue        # 52週安値圏（株価低迷）を要求
                # 業績改善シグナル（BPS成長 or EPSサプライズのどちらかがプラスでよい）
                if f[IDX_BPS_GROWTH] <= 0 and f[IDX_EPS_SURPRISE] <= 0: continue
            rp = float(rise_model.predict_proba([f])[0][1]) * 100
            dp = float(drop_model.predict_proba([f])[0][1]) * 100 if drop_model else 0.0
            net = rp - dp
            if args.strategy == "v2" and dp > BUY_DROP_MAX:
                continue
            if args.strategy in ("v3", "v4") and dp > 6.0:
                continue
            if args.strategy == "momnet" and dp > 10.0:    # モメンタム＋モデルの危険回避
                continue
            if args.strategy == "qv" and dp > 8.0:         # QV: 下落リスクが高すぎるもの除外
                continue
            if args.strategy == "baseline" and net < 5.0:
                continue
            sector = sector_map.get(code) or f"_{code}"
            # 並べ替えスコア: mom系は3ヶ月リターン、それ以外はnet
            score = (f[IDX_RET90] * 100) if args.strategy in ("mom", "momnet", "momreg") else net
            out.append((code, name, score, dp, price, sector))
        out.sort(key=lambda x: -x[2])
        return out

    def score_position(code, d):
        """保有銘柄のd時点 (net, drop_prob) を返す。算出不能なら None。"""
        h, name, dts, cls, vol = hist_map[code]
        feat = bt.extract_features_at(h, d, nk_rets_at(d), code=code)
        if feat is None:
            return None
        fa = add_cs_rank_features(np.array([feat], dtype=float))[0]
        rp = float(rise_model.predict_proba([fa])[0][1]) * 100
        dp = float(drop_model.predict_proba([fa])[0][1]) * 100 if drop_model else 0.0
        return (rp - dp, dp, fa[IDX_VOL20])

    # ── イベント駆動シミュレーション ────────────────────────────────────────
    cash = START_CAPITAL
    positions = {}   # code -> dict(entry_date, entry_price, value, peak, name, sector, vol20)
    equity_curve = []
    trades = []

    def total_equity(d):
        eq = cash
        for code, pos in positions.items():
            dts, cls = hist_map[code][2], hist_map[code][3]
            px = price_on_or_before(dts, cls, d) or pos["entry_price"]
            eq += pos["shares"] * px
        return eq

    for d in decision_dates:
        # v4: 相場レジームが risk-off（日経<63SMA等）なら全ポジ退避（暴落サーキットブレーカー）
        regime_off = (args.strategy in ("v4", "momreg")) and (not market_gate_ok(d))

        # 1) 売り判定（保有銘柄）
        for code in list(positions.keys()):
            pos = positions[code]
            dts, cls = hist_map[code][2], hist_map[code][3]
            px = price_on_or_before(dts, cls, d)
            if px is None:
                continue
            pos["peak"] = max(pos["peak"], px)
            held_days = sum(1 for x in trading_dates if pos["entry_date"] < x <= d)
            sell = None
            if regime_off:
                sell = "相場退避"
            sc = score_position(code, d) if sell is None else None
            if sc is not None:
                net, dp, vol20 = sc
                if args.strategy == "v2":
                    if dp >= SELL_DROP_MIN: sell = "drop急騰"
                    elif net < SELL_NET_MAX: sell = "net反転"
                elif args.strategy in ("v3", "v4"):
                    # 勝ち伸ばし: 危険検知(drop急騰)のみモデル売り。net低下では売らない
                    if dp >= 10.0: sell = "drop急騰"
                elif args.strategy == "momnet":
                    if dp >= 12.0: sell = "drop急騰"   # momnetのみ軽くdrop売り
                elif args.strategy == "baseline":
                    if net < 5.0: sell = "net<5"
                elif args.strategy == "qv":
                    # 逆張りなので少し許容。drop急騰 or net大幅マイナスで撤退
                    if dp >= 10.0: sell = "drop急騰"
                    elif net < -5.0: sell = "net大幅反転"
                # mom（純モメンタム）はモデル売りしない＝トレーリング/保有上限のみで勝ち伸ばす
            if sell is None and args.strategy in ("v2", "v3", "v4", "mom", "momnet", "momreg"):
                # トレーリングストップ（vol20ベース）。モメンタムは値動きが荒いので広め
                mult = 2.5 if args.strategy in ("mom", "momnet", "momreg") else TRAIL_ATR_MULT
                vol20 = pos["vol20"]
                stop_frac = mult * (vol20 / 100.0) * np.sqrt(20.0 / 252.0)
                if px <= pos["peak"] * (1 - stop_frac):
                    sell = "トレール"
            if sell is None and held_days >= args.hold_days:
                sell = "時間切れ"
            if sell:
                ret = (px - pos["entry_price"]) / pos["entry_price"] * 100
                cash += pos["shares"] * px
                trades.append({"code": code, "name": pos["name"], "entry": str(pos["entry_date"]),
                               "exit": str(d), "held_days": held_days,
                               "entry_px": round(pos["entry_price"], 1), "exit_px": round(px, 1),
                               "return": round(ret, 2), "reason": sell})
                del positions[code]

        # 2) 買い判定（スロット空き）。相場ゲートは v2 のみ適用（v3/baselineは常時フル投資）
        open_slots = max_pos - len(positions)
        # qv は逆張り（相場弱くても仕込む）なので相場ゲートなし
        gate_ok = market_gate_ok(d) if args.strategy in ("v2", "v4", "momreg") else True
        if open_slots > 0 and gate_ok:
            cands = score_universe(d)
            from collections import Counter as _Counter
            sector_count = _Counter(p["sector"] for p in positions.values())
            eq = total_equity(d)
            alloc = eq / max_pos
            for code, name, net, dp, price, sector in cands:
                if open_slots <= 0:
                    break
                if code in positions:
                    continue
                if sector_count[sector] >= args.sector_cap:   # 1セクター上限
                    continue
                if cash < alloc * 0.99:
                    break
                shares = alloc / price
                cash -= alloc
                sc = score_position(code, d)
                vol20 = sc[2] if sc else 25.0
                positions[code] = {"entry_date": d, "entry_price": price, "shares": shares,
                                   "peak": price, "name": name, "sector": sector, "vol20": vol20}
                sector_count[sector] += 1
                open_slots -= 1

        equity_curve.append((d, total_equity(d)))

    # 期末に全クローズ
    last_d = trading_dates[-1]
    for code in list(positions.keys()):
        pos = positions[code]
        dts, cls = hist_map[code][2], hist_map[code][3]
        px = price_on_or_before(dts, cls, last_d) or pos["entry_price"]
        held_days = sum(1 for x in trading_dates if pos["entry_date"] < x <= last_d)
        ret = (px - pos["entry_price"]) / pos["entry_price"] * 100
        cash += pos["shares"] * px
        trades.append({"code": code, "name": pos["name"], "entry": str(pos["entry_date"]),
                       "exit": str(last_d), "held_days": held_days,
                       "entry_px": round(pos["entry_price"], 1), "exit_px": round(px, 1),
                       "return": round(ret, 2), "reason": "期末"})
        del positions[code]
    equity_curve.append((last_d, cash))

    # ── 成績集計 ────────────────────────────────────────────────────────────
    if not trades:
        print("トレードなし"); return
    rets = np.array([t["return"] for t in trades])
    final_eq = cash
    total_ret = (final_eq / START_CAPITAL - 1) * 100
    days = (END - START).days
    cagr = ((final_eq / START_CAPITAL) ** (365.0 / max(days, 1)) - 1) * 100

    eq_vals = np.array([e for _, e in equity_curve])
    peak = np.maximum.accumulate(eq_vals)
    max_dd = float(((eq_vals - peak) / peak).min() * 100)

    win = (rets > 0).mean() * 100
    big = (rets >= BIG_WIN).mean() * 100

    # 日経ベンチ
    nk_s = price_on_or_before(nk_dates, nk_closes, START)
    nk_e = price_on_or_before(nk_dates, nk_closes, END)
    nk_total = (nk_e - nk_s) / nk_s * 100 if nk_s and nk_e else 0.0

    print(f"\n{'='*64}")
    print(f"【{args.strategy} 成績  {START}→{END}（{days}日）】")
    print(f"  トレード数:        {len(trades)}")
    print(f"  期間トータル:      {total_ret:+.1f}%   (資産 {final_eq/START_CAPITAL:.2f}倍)")
    print(f"  CAGR(年率):        {cagr:+.1f}%")
    print(f"  平均/中央 リターン: {rets.mean():+.2f}% / {np.median(rets):+.2f}%")
    print(f"  勝率:              {win:.0f}%")
    print(f"  大勝率(≥{BIG_WIN:.0f}%):     {big:.0f}%")
    print(f"  最大ドローダウン:   {max_dd:.1f}%")
    print(f"  日経225 同期間:     {nk_total:+.1f}%   → アルファ {total_ret - nk_total:+.1f}%")

    # 1億円到達の目安
    if cagr > 0:
        for start_capital in [1_000_000, 3_000_000, 10_000_000]:
            yrs = np.log(100_000_000 / start_capital) / np.log(1 + cagr / 100)
            print(f"    元本¥{start_capital//10000}万 → 1億円: 約{yrs:.1f}年（CAGR {cagr:.1f}%継続前提）")

    # 売却理由の内訳
    from collections import Counter
    reasons = Counter(t["reason"] for t in trades)
    print(f"  売却理由: " + " / ".join(f"{k}{v}" for k, v in reasons.most_common()))

    # CSV保存
    out_dir = os.path.join(BASE_DIR, "simulations", "backtests")
    os.makedirs(out_dir, exist_ok=True)
    tag = "bear" if args.bear else f"{START}_{END}"
    out = os.path.join(out_dir, f"strategy_v2_{args.strategy}_{tag}.csv")
    pd.DataFrame(trades).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nトレード明細: {out}")
    print("完了")


if __name__ == "__main__":
    main()
