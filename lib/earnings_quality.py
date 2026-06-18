"""
lib/earnings_quality.py
カタリスト候補の「利益の質（化粧決算除外）」と「本業の方向性（斜陽除外）」を判定する。

背景: PBR<1 × ROE<8% × 自己資本比率>50% のスクリーンは「安い箱」しか見ておらず、
低ROEの"理由"を問わないため、(1)一過性益で純利益を水増しした化粧決算、(2)本業が
構造的に縮小する斜陽事業、が上位に混入する。本モジュールはそれを機械的に弾く。

データ源: kabutan 年次業績（lib.kabutan_earnings.fetch_kabutan_earnings の rows）。
  各 row: {fy_end, is_forecast, revenue, op_profit, ord_profit, net_income, eps, dps, ...}

返り値の方針:
- ゲート（除外）: 営業赤字 / 化粧決算（純利益>>営業益）/ 本業減益 → exclude に理由を入れる。
- スコア（加減点）: 営業利益率トレンド・売上CAGR・会社予想方向 → bonus（-, 0, +）。
データが無い銘柄は判定不能として keep（exclude=None, data=False）。
"""
from typing import Optional


def _actuals(rows: list) -> list:
    return [r for r in rows if r and not r.get("is_forecast") and r.get("op_profit") is not None]


def _forecast_row(rows: list) -> Optional[dict]:
    fc = [r for r in rows if r and r.get("is_forecast")]
    return fc[-1] if fc else None


def op_margin_declining_3(acts: list) -> bool:
    """直近3実績期で営業利益率が連続低下していれば True。"""
    usable = [r for r in acts if r.get("revenue")]
    if len(usable) < 3:
        return False
    m = [r["op_profit"] / r["revenue"] for r in usable[-3:]]
    return m[0] > m[1] > m[2]


def revenue_cagr3(acts: list) -> Optional[float]:
    """直近3実績期（2区間）の売上CAGR。算出不能なら None。"""
    usable = [r for r in acts if r.get("revenue") and r["revenue"] > 0]
    if len(usable) < 3:
        return None
    first, last = usable[-3]["revenue"], usable[-1]["revenue"]
    return (last / first) ** (1 / 2) - 1


def assess_earnings_quality(rows: list, cosmetic_mult: float = 1.5) -> dict:
    """利益の質・本業方向性を評価する。

    Returns dict:
      data:    bool   判定に足る実績データがあったか
      exclude: str|None  除外理由（'営業赤字' / '化粧決算' / '本業減益'）。Noneなら通過
      bonus:   float  ランキング加減点（負=減点, 正=加点）
      op:      float|None  直近実績の営業利益
      op_yoy:  float|None  営業利益の前年比（%）
      op_margin: float|None 直近営業利益率（%）
      rev_cagr3: float|None 売上3期CAGR（%）
      fc_dir:  str    会社予想方向（'増収増益' / '減益予想' / '-'）
      notes:   list[str]
    """
    res = {"data": False, "exclude": None, "bonus": 0.0, "op": None,
           "op_yoy": None, "op_margin": None, "rev_cagr3": None,
           "fc_dir": "-", "notes": []}
    acts = _actuals(rows)
    if not acts:
        return res
    res["data"] = True
    latest = acts[-1]
    prev = acts[-2] if len(acts) >= 2 else None
    op = latest["op_profit"]
    res["op"] = op
    if latest.get("revenue"):
        res["op_margin"] = round(op / latest["revenue"] * 100, 1)

    # ── フィルターA: 利益の質（化粧決算除外）──────────────────────────
    if op <= 0:
        res["exclude"] = "営業赤字"
        res["notes"].append(f"営業益{op:.0f}≤0")
        return res
    net = latest.get("net_income")
    if net is not None and net > op * cosmetic_mult:
        res["exclude"] = "化粧決算"
        res["notes"].append(f"純益{net:.0f}>営業益{op:.0f}×{cosmetic_mult}（一過性益の疑い）")
        return res

    # ── フィルターB: 本業の方向性（斜陽除外）──────────────────────────
    if prev and prev.get("op_profit") is not None:
        prev_op = prev["op_profit"]
        if prev_op != 0:
            res["op_yoy"] = round((op - prev_op) / abs(prev_op) * 100, 1)
        if op <= prev_op:
            res["exclude"] = "本業減益"
            res["notes"].append(f"営業益 前年比 {res['op_yoy']}%（減益）")
            return res

    # ── 通過した候補への加減点 ────────────────────────────────────────
    cagr = revenue_cagr3(acts)
    if cagr is not None:
        res["rev_cagr3"] = round(cagr * 100, 1)
        res["bonus"] += 0.15 if cagr > 0 else -0.15
    if op_margin_declining_3(acts):
        res["bonus"] -= 0.15
        res["notes"].append("営業利益率3期連続低下")
    fc = _forecast_row(rows)
    if fc and fc.get("op_profit") is not None:
        inc_op = fc["op_profit"] > op
        inc_rev = (fc.get("revenue") is not None and latest.get("revenue") is not None
                   and fc["revenue"] > latest["revenue"])
        if inc_op and inc_rev:
            res["fc_dir"] = "増収増益"
            res["bonus"] += 0.15
        elif fc["op_profit"] <= 0 or not inc_op:
            res["fc_dir"] = "減益予想"
            res["bonus"] -= 0.10
    res["bonus"] = round(res["bonus"], 3)
    return res
