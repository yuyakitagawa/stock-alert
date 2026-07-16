"""
market_compare.py — 日経225 vs S&P500 相対強弱アドバイザー

「日経が弱いから米国株(S&P500)の方がいいのでは？」という問いに答えるため、
日経225とS&P500の直近リターン（5日・20日・60日）を比較し、
どちらが優勢かを判定する。売買シグナルには影響しない純粋な情報表示。

出力（compare の戻り値）:
  {
    "verdict": "jp_favored" | "us_favored" | "neutral",
    "score":   優劣スコア（正=日本株優位 … 負=米国株優位）,
    "label":   "🇯🇵 日本株優位" 等の表示用ラベル,
    "reasons": ["20日: 日経+1.2% vs S&P500-3.4%（日本株優位 +4.6pt）", ...],
  }
"""
from __future__ import annotations

DIFF20_TH = 3.0   # 20日リターン差がこれ以上で優劣判定
DIFF60_TH = 5.0   # 60日リターン差がこれ以上で優劣判定


def compare(nk5=None, nk20=None, nk60=None, us5=None, us20=None, us60=None) -> dict:
    """日経225とS&P500の直近リターンを比較し、どちらが優勢かを判定する。"""
    score = 0
    reasons: list[str] = []

    if nk20 is not None and us20 is not None:
        diff20 = nk20 - us20
        if diff20 <= -DIFF20_TH:
            score -= 1
            reasons.append(f"20日: 日経{nk20:+.1f}% vs S&P500{us20:+.1f}%（米国株優位 {diff20:+.1f}pt）")
        elif diff20 >= DIFF20_TH:
            score += 1
            reasons.append(f"20日: 日経{nk20:+.1f}% vs S&P500{us20:+.1f}%（日本株優位 {diff20:+.1f}pt）")

    if nk60 is not None and us60 is not None:
        diff60 = nk60 - us60
        if diff60 <= -DIFF60_TH:
            score -= 1
            reasons.append(f"60日: 日経{nk60:+.1f}% vs S&P500{us60:+.1f}%（米国株優位 {diff60:+.1f}pt）")
        elif diff60 >= DIFF60_TH:
            score += 1
            reasons.append(f"60日: 日経{nk60:+.1f}% vs S&P500{us60:+.1f}%（日本株優位 {diff60:+.1f}pt）")

    if score <= -2:
        verdict, label = "us_favored", "🇺🇸 米国株(S&P500)優位"
    elif score >= 2:
        verdict, label = "jp_favored", "🇯🇵 日本株優位"
    else:
        verdict, label = "neutral", "🤝 拮抗（明確な優劣なし）"

    if not reasons:
        reasons.append("データ不足 or 明確な優劣差なし")

    return {
        "verdict": verdict,
        "score":   score,
        "label":   label,
        "reasons": reasons,
        "nk5": nk5, "nk20": nk20, "nk60": nk60,
        "us5": us5, "us20": us20, "us60": us60,
    }


def summary_line(verdict: dict) -> str:
    """1行サマリー（ログ・メール見出し用）。"""
    return f"{verdict['label']}（優劣スコア{verdict['score']:+d}）→ " + " / ".join(verdict["reasons"])
