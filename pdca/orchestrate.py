#!/usr/bin/env python3
"""PDCA Orchestrator: 1日1サイクルの自律改善ループ
   Fund Manager (haiku) → Analyst (sonnet) → Engineer (python) → commit/push
"""
import sys, os, re, json, subprocess
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PDCA_DIR = Path(__file__).resolve().parent
PYTHON   = sys.executable
FEEDBACK = PDCA_DIR / "feedback.md"
LOG      = PDCA_DIR / "pdca_log.md"
BASELINE = PDCA_DIR / "baseline_metrics.json"
TODAY    = date.today().isoformat()

load_dotenv(BASE_DIR / ".env")

try:
    import anthropic
    client = anthropic.Anthropic()
except ImportError:
    print("anthropic パッケージが必要です: pip install anthropic"); sys.exit(1)


# ── ユーティリティ ─────────────────────────────────────────────────────────

def run_backtest():
    r = subprocess.run(
        [PYTHON, "tools/backtest.py", "bear"],
        capture_output=True, text=True, cwd=str(BASE_DIR), timeout=1800,
        env={**os.environ, "STOCK_ALERT_HOME": str(BASE_DIR)},
    )
    out = r.stdout + r.stderr
    m = {}
    for pat, key in [
        (r'平均リターン: ([+-]?\d+\.?\d*)%',                  'avg_return'),
        (r'勝率（\+0%以上）: \d+/\d+ = (\d+\.?\d*)%',         'win_rate'),
        (r'大勝率（\+15%以上）: \d+/\d+ = (\d+\.?\d*)%',      'big_win_rate'),
    ]:
        found = re.search(pat, out)
        if found:
            m[key] = float(found.group(1))
    return m, out


def load_baseline():
    return json.loads(BASELINE.read_text()) if BASELINE.exists() else {}


def save_baseline(m):
    BASELINE.write_text(json.dumps(m, indent=2))


def read_feedback():
    return FEEDBACK.read_text(encoding='utf-8') if FEEDBACK.exists() else "なし"


def log(text):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def call_claude(model, prompt, max_tokens=256):
    r = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text


def parse_json(text):
    m = re.search(r'\{.*\}', text, re.DOTALL)
    try:
        return json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None


def git(*args):
    return subprocess.run(['git'] + list(args), cwd=str(BASE_DIR), capture_output=True)


def git_commit_push(files, msg):
    git('config', 'user.email', 'pdca-bot@github-actions')
    git('config', 'user.name',  'PDCA Bot')
    git('add', *files)
    git('commit', '-m', msg)
    git('pull', '--rebase', 'origin', 'main')
    git('push')


# ── 各エージェント ──────────────────────────────────────────────────────────

def fund_manager(metrics, baseline, feedback):
    bsl = baseline or metrics
    prompt = f"""あなたは株式予測モデルのファンドマネージャーです。

【今日のbacktest(bear mode: 2024年8月下落相場)】
avg_return={metrics.get('avg_return','N/A')}%  win_rate={metrics.get('win_rate','N/A')}%  big_win_rate={metrics.get('big_win_rate','N/A')}%

【ベースライン（これまでの最良値）】
avg_return={bsl.get('avg_return','N/A')}%  win_rate={bsl.get('win_rate','N/A')}%  big_win_rate={bsl.get('big_win_rate','N/A')}%

【人間フィードバック】
{feedback}

今日config.pyの1パラメータを調整して改善を試みるべきか？
JSON1行のみで回答: {{"action":"improve"|"skip","reason":"..."}}"""
    result = parse_json(call_claude("claude-haiku-4-5-20251001", prompt, 200))
    return result or {"action": "skip", "reason": "parse error"}


def _research_papers():
    """最新の量的金融・機械学習論文を web 検索してサマリーを返す（失敗しても継続）"""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            messages=[{"role": "user", "content": (
                "XGBoostによる株価63日先予測モデル（日本株）の改善に役立つ"
                "2024〜2025年の最新研究・論文を検索してください。"
                "スクリーニングフィルター・モメンタム・ボラティリティ・日経アルファ生成に関する知見を"
                "実装に直結する形で3点、日本語で箇条書きにしてください。"
            )}],
        )
        texts = [b.text for b in response.content if hasattr(b, 'text') and b.text]
        return '\n'.join(texts) if texts else "（検索結果なし）"
    except Exception as e:
        return f"（論文検索スキップ: {type(e).__name__}）"


def analyst(metrics, baseline):
    bsl    = baseline or metrics
    config = (BASE_DIR / "config.py").read_text('utf-8')
    fi     = (BASE_DIR / "feature_importance.json").read_text('utf-8')

    print("  → 最新論文を検索中...")
    research = _research_papers()
    print(f"  → 検索完了: {research[:80].strip()}...")

    prompt = f"""あなたは株式予測モデルのアナリストです。
最新研究の知見とcurrent config.pyを踏まえ、bear modeバックテストを改善する変更を1つ提案してください。

【現在】avg={metrics.get('avg_return')}% win={metrics.get('win_rate')}% big={metrics.get('big_win_rate')}%
【ベースライン】avg={bsl.get('avg_return')}% win={bsl.get('win_rate')}% big={bsl.get('big_win_rate')}%

【config.py】
{config}

【特徴量重要度（参考）】
{fi[:600]}

【最新研究からの知見】
{research}

制約:
- config.pyの数値定数を1つだけ変更（コメント行は除く）
- 変更幅は現在値の±25%以内
- 変更禁止: BEAR_MARKET_THRESHOLD, HOT_MARKET_THRESHOLD

JSON1行のみで回答: {{"param_name":"...","old_value":...,"new_value":...,"reason":"..."}}"""
    return parse_json(call_claude("claude-sonnet-4-6", prompt, 400))


def engineer(proposal, before_metrics, baseline):
    cfg  = BASE_DIR / "config.py"
    orig = cfg.read_text('utf-8')
    p    = proposal['param_name']
    nv   = proposal['new_value']

    # 行頭のパラメータ代入を置換（どんな数値でも対応）
    pat = rf'^({re.escape(p)}\s*=\s*)[\d.+\-]+'
    new = re.sub(pat, rf'\g<1>{nv}', orig, flags=re.MULTILINE)

    if new == orig:
        return False, f"{p} の値が見つからず変更失敗"

    cfg.write_text(new, 'utf-8')
    after, _ = run_backtest()

    if not after:
        cfg.write_text(orig, 'utf-8')
        return False, "backtest失敗、revert"

    bsl = baseline if baseline else before_metrics
    improved = any([
        after.get('avg_return',    -9999) > bsl.get('avg_return',    -9999),
        after.get('win_rate',          0) > bsl.get('win_rate',          0),
        after.get('big_win_rate',      0) > bsl.get('big_win_rate',      0),
    ])

    if improved:
        save_baseline(after)
        ov = proposal.get('old_value', '?')
        msg = (
            f"pdca: {p} {ov}→{nv} | "
            f"avg {bsl.get('avg_return','?')}→{after.get('avg_return','?')}% "
            f"win {bsl.get('win_rate','?')}→{after.get('win_rate','?')}% [skip ci]"
        )
        git_commit_push(['config.py', 'pdca/baseline_metrics.json'], msg)
        return True, (
            f"✅ 採用: avg {bsl.get('avg_return')}%→{after.get('avg_return')}%  "
            f"win {bsl.get('win_rate')}%→{after.get('win_rate')}%  "
            f"big {bsl.get('big_win_rate')}%→{after.get('big_win_rate')}%"
        )
    else:
        cfg.write_text(orig, 'utf-8')
        return False, (
            f"❌ 改善なし: avg {after.get('avg_return')}% "
            f"(bsl {bsl.get('avg_return')}%)、revert"
        )


# ── メイン ─────────────────────────────────────────────────────────────────

def push_log():
    git('config', 'user.email', 'pdca-bot@github-actions')
    git('config', 'user.name',  'PDCA Bot')
    git('add', 'pdca/pdca_log.md')
    r = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=str(BASE_DIR))
    if r.returncode != 0:
        git('commit', '-m', f'pdca: log {TODAY} [skip ci]')
        git('pull', '--rebase', 'origin', 'main')
        git('push')


def main():
    print(f"=== PDCA {TODAY} ===")
    log(f"\n## {TODAY}")

    # Step1: backtest実行
    print("Step1: backtest.py bear 実行中（数分かかります）...")
    metrics, raw = run_backtest()
    if not metrics:
        msg = "ERROR: backtest失敗（メトリクス取得不可）"
        print(msg); log(f"- {msg}"); push_log(); sys.exit(1)
    print(f"  avg={metrics.get('avg_return')}%  win={metrics.get('win_rate')}%  big={metrics.get('big_win_rate')}%")
    log(f"- metrics: {json.dumps(metrics)}")

    baseline = load_baseline()
    if not baseline:
        save_baseline(metrics)
        baseline = metrics
        log("- 初回実行: baseline設定")
        print("  初回: baselineを現在値で設定")

    # Step2: Fund Manager
    print("Step2: Fund Manager 評価中...")
    fb = read_feedback()
    fm = fund_manager(metrics, baseline, fb)
    print(f"  → {fm['action']}: {fm['reason']}")
    log(f"- FM: {fm['action']} | {fm['reason']}")

    if fm['action'] == 'skip':
        print("今日は改善スキップ")
        log("- skip")
        push_log()
        return

    # Step3: Analyst
    print("Step3: Analyst 分析中...")
    prop = analyst(metrics, baseline)
    if not prop:
        log("- analyst: parse error"); push_log(); return
    print(f"  → {prop['param_name']}: {prop.get('old_value')}→{prop['new_value']}  ({prop['reason']})")
    log(f"- analyst: {json.dumps(prop, ensure_ascii=False)}")

    # Step4: Engineer（変更→backtest→commit or revert）
    print("Step4: Engineer 実装・検証中...")
    ok, detail = engineer(prop, metrics, baseline)
    print(f"  → {detail}")
    log(f"- engineer: {detail}")

    push_log()
    print("=== 完了 ===")


if __name__ == '__main__':
    main()
