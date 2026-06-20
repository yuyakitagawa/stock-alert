"""
move_history.py — PDCA「棋譜」: 過去のパラメータ変更履歴と採否を抽出する。

目的:
  - Quant が「過去に何を試したか」を見てから提案できるようにする（振り子防止）
  - 直近で逆方向に動かしたパラメータを再反転する提案を機械的に検出・却下する

pdca_log.md の形式を前提:
  - analyst: {"file": "...", "changes": [{"param_name","old_value","new_value","reason"}, ...]}
  - engineer: ✅ 採用 [...]: ... | ...
  - engineer: ❌ 改善なし [...]: ... revert
"""
import re
import json
from pathlib import Path

PDCA_DIR = Path(__file__).resolve().parent
LOG_PATH = PDCA_DIR / "pdca_log.md"


def _to_num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_recent_moves(n_cycles: int = 12, log_path: Path | None = None) -> list[dict]:
    """
    直近 n_cycles サイクル分の (提案 + 採否) を新しい順で返す。

    戻り値: [
      { "param": "max_depth", "old": 5.0, "new": 4.0,
        "file": "rf_train_v3.py", "adopted": False, "reason": "..." },
      ...
    ]
    """
    path = log_path or LOG_PATH
    if not path.exists():
        return []
    text = path.read_text("utf-8")
    lines = text.splitlines()

    moves: list[dict] = []
    pending: list[dict] | None = None  # 直前の analyst 提案の changes

    for line in lines:
        s = line.strip()

        m = re.match(r"^-\s*analyst:\s*(\{.*\})\s*$", s)
        if m:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                pending = None
                continue
            changes = obj.get("changes", [])
            file = obj.get("file", "?")
            pending = []
            for ch in changes:
                p = re.sub(r"[（(].*", "", str(ch.get("param_name", ""))).strip()
                if not p:
                    continue
                pending.append({
                    "param":  p,
                    "old":    _to_num(ch.get("old_value")),
                    "new":    _to_num(ch.get("new_value")),
                    "file":   file,
                    "reason": ch.get("reason", ""),
                    "adopted": None,   # engineer 行で確定
                })
            continue

        m = re.match(r"^-\s*engineer:\s*(.*)$", s)
        if m and pending is not None:
            outcome = m.group(1)
            adopted = outcome.startswith("✅") or "採用" in outcome[:6]
            for mv in pending:
                mv["adopted"] = adopted
                moves.append(mv)
            pending = None
            continue

    moves.reverse()  # 新しい順
    return moves[: n_cycles * 4]  # 1サイクル最大数パラメータ想定


def detect_oscillation(proposal: dict, moves: list[dict], lookback: int = 30) -> list[dict]:
    """
    提案 proposal の中で「直近の変更を逆方向に戻すだけ」のパラメータを返す。

    振り子の定義:
      直近で param を A→B に動かしたのに、今回 B→A（A方向）に戻そうとしている。
    lookback は param-move 単位の窓（1サイクル≒10変更なので 30 ≒ 直近3サイクル）。
    """
    recent = moves[:lookback]
    # param ごとに直近の動きの向きを集める
    last_dir: dict[str, float] = {}   # param -> 直近の (new - old) の符号
    for mv in recent:
        if mv["param"] in last_dir:
            continue  # 直近のものだけ採用
        if mv["old"] is None or mv["new"] is None:
            continue
        delta = mv["new"] - mv["old"]
        if delta != 0:
            last_dir[mv["param"]] = delta

    oscillating = []
    for ch in proposal.get("changes", []):
        p = re.sub(r"[（(].*", "", str(ch.get("param_name", ""))).strip()
        old = _to_num(ch.get("old_value"))
        new = _to_num(ch.get("new_value"))
        if p not in last_dir or old is None or new is None:
            continue
        now_delta = new - old
        if now_delta == 0:
            continue
        # 直近の動きと符号が逆 = 振り子
        if now_delta * last_dir[p] < 0:
            oscillating.append({
                "param": p,
                "now": f"{old}→{new}",
                "recent": f"前回は逆方向（Δ={last_dir[p]:+g}）",
                "reason": ch.get("reason", ""),
            })
    return oscillating


def format_kifu(moves: list[dict], limit: int = 12) -> str:
    """棋譜（過去の試行履歴）を Quant プロンプト用テキストに整形。"""
    if not moves:
        return "（過去の変更履歴なし）"

    # param ごとに「何回試され、採用されたか」を集計
    from collections import defaultdict
    stats = defaultdict(lambda: {"tried": 0, "adopted": 0, "last": ""})
    for mv in moves:
        st = stats[mv["param"]]
        st["tried"] += 1
        if mv["adopted"]:
            st["adopted"] += 1
        if not st["last"] and mv["old"] is not None and mv["new"] is not None:
            st["last"] = f"{mv['old']}→{mv['new']}"

    lines = ["【過去の試行棋譜（新しい順）】"]
    for mv in moves[:limit]:
        mark = "✅採用" if mv["adopted"] else "❌却下"
        o = mv["old"] if mv["old"] is not None else "?"
        nw = mv["new"] if mv["new"] is not None else "?"
        lines.append(f"  {mark} {mv['param']}: {o}→{nw}")

    lines.append("\n【パラメータ別 試行回数 / 採用回数】")
    for p, st in sorted(stats.items(), key=lambda x: -x[1]["tried"]):
        warn = " ⚠️何度も試行され効果なし" if st["tried"] >= 3 and st["adopted"] == 0 else ""
        lines.append(f"  {p}: {st['tried']}回試行 / {st['adopted']}回採用{warn}")

    return "\n".join(lines)


if __name__ == "__main__":
    mv = load_recent_moves(12)
    print(format_kifu(mv))
    print()
    # 振り子テスト: max_depth を直近と逆に動かす提案
    test_prop = {"file": "rf_train_v3.py", "changes": [
        {"param_name": "max_depth", "old_value": 4, "new_value": 5, "reason": "test"},
    ]}
    osc = detect_oscillation(test_prop, mv)
    print("振り子検出:", osc)
