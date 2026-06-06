#!/usr/bin/env python3
"""週次ピアレビュー: 5ロール相互評価
   ロール:
     FM（ファンドマネージャー）、Quant（数量アナリスト）、Securities（証券アナリスト）、
     Engineer（エンジニア）、Human（オーナー・評価を受けるのみ）
   毎週月曜 10:00 JST に GitHub Actions から自動実行
"""
import sys, os, re, json, subprocess, requests
from pathlib import Path
from datetime import date, timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
PDCA_DIR = Path(__file__).resolve().parent
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PDCA_LOG = PDCA_DIR / "pdca_log.md"
FEEDBACK = PDCA_DIR / "feedback.md"
TODAY    = date.today()
ISO_WEEK = TODAY.strftime("%Y-W%W")

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env", override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SVC = os.getenv("SUPABASE_SERVICE_KEY", "")

try:
    import anthropic
    client = anthropic.Anthropic()
except ImportError:
    print("anthropic パッケージが必要です"); sys.exit(1)


# ── ユーティリティ ────────────────────────────────────────────────────────────

def call_claude(prompt: str, max_tokens: int = 800) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=str(BASE_DIR), capture_output=True)


def git_commit_push(files: list[str], msg: str):
    git("config", "user.email", "pdca-bot@github-actions")
    git("config", "user.name",  "PDCA Bot")
    git("add", *files)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(BASE_DIR))
    if r.returncode != 0:
        git("commit", "-m", msg)
        git("pull", "--rebase", "origin", "main")
        git("push")


def extract_last_week_log() -> str:
    if not PDCA_LOG.exists():
        return "（ログなし）"
    text   = PDCA_LOG.read_text(encoding="utf-8")
    lines  = text.splitlines()
    result, inside, cutoff = [], False, TODAY - timedelta(days=7)
    for line in lines:
        m = re.match(r"^## (\d{4}-\d{2}-\d{2})", line)
        if m:
            inside = date.fromisoformat(m.group(1)) >= cutoff
        if inside:
            result.append(line)
    return "\n".join(result) if result else "（今週のログなし）"


def load_feedback_text() -> str:
    return FEEDBACK.read_text(encoding="utf-8") if FEEDBACK.exists() else "（なし）"


def parse_metrics_trajectory(log_text: str) -> list[dict]:
    traj = []
    for line in log_text.splitlines():
        m = re.search(r"- metrics: (\{.*?\})", line)
        if m:
            try:
                traj.append(json.loads(m.group(1)))
            except Exception:
                pass
    return traj


def count_actions(log_text: str) -> dict:
    return {
        "adopted":     len(re.findall(r"✅ 採用",    log_text)),
        "rejected":    len(re.findall(r"❌ 改善なし", log_text)),
        "skipped":     len(re.findall(r"- skip\b",   log_text)),
        "parse_error": len(re.findall(r"parse error", log_text)),
        "signals":     len(re.findall(r"- signals: \d+銘柄", log_text)),
    }


def upsert_supabase(row: dict) -> None:
    if not SUPABASE_URL or not SUPABASE_SVC:
        print("  [Supabase スキップ] URL/KEY 未設定")
        return
    headers = {
        "apikey": SUPABASE_SVC,
        "Authorization": f"Bearer {SUPABASE_SVC}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates",
    }
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/weekly_reviews",
        headers=headers, json=[row], timeout=15,
    )
    if resp.ok:
        print("  → Supabase 保存完了")
    else:
        print(f"  [Supabase エラー] {resp.status_code} {resp.text[:200]}")


# ── 評価プロンプト ────────────────────────────────────────────────────────────

# 共通の背景説明（シンプルに）
_CTX = """【システムの説明】
日本株の予測モデルをAIチームが毎日自動改善しています。
目標: 元本300万円を10年で1億円にする（年率42%）。

チームのメンバー:
・FM（ファンドマネージャー）: 今日改善するかどうかを決める
・Quant（数量アナリスト）: モデルの設定値を改善案として出す
・Securities（証券アナリスト）: 買い候補銘柄を企業調査する
・Engineer（エンジニア）: Quantの案を実際に試して良ければ採用、悪ければ元に戻す
・Human（オーナー）: 方針や目標を文章で伝える

達成すべき数字:
・平均リターン > 2%（21日間）
・勝率 > 50%
・大勝率（+8%以上）> 30%
・日経平均より良いリターン（アルファ > 0%）
"""


def engineer_evaluates(log_text: str, actions: dict) -> str:
    prompt = f"""{_CTX}

あなたはEngineerです。今週の記録を読んで、QuantとFMの仕事ぶりを評価してください。

【今週の記録】
{log_text}

【集計】採用:{actions['adopted']}件 / 却下:{actions['rejected']}件 / エラー:{actions['parse_error']}件

## Quantへの評価
### よかった点（2〜3つ）
### 問題だった点（2〜3つ）
※「パラメータ名が違ってて適用できなかった」「毎回同じ項目しか変えない」「理由が薄い」など

## FMへの評価
### よかった点（1〜2つ）
### 問題だった点（1〜2つ）
※「毎日improve一択で考えていない」「目標が高すぎ/低すぎ」など

専門用語を使わず、中学生でもわかる言葉で書いてください。"""
    return call_claude(prompt)


def quant_evaluates(log_text: str, actions: dict) -> str:
    prompt = f"""{_CTX}

あなたはQuantです。今週の記録を読んで、EngineerとFMの仕事ぶりを評価してください。

【今週の記録】
{log_text}

【集計】採用:{actions['adopted']}件 / 却下:{actions['rejected']}件 / エラー:{actions['parse_error']}件

## Engineerへの評価
### よかった点（2〜3つ）
### 問題だった点（2〜3つ）
※「採用基準がブレている」「元に戻すタイミングが遅い」「テスト結果の読み方が甘い」など

## FMへの評価
### よかった点（1〜2つ）
### 問題だった点（1〜2つ）
※「相場環境を無視した判断」「大勝率の目標設定が非現実的」など

専門用語を使わず、中学生でもわかる言葉で書いてください。"""
    return call_claude(prompt)


def securities_evaluates(log_text: str, actions: dict) -> str:
    prompt = f"""{_CTX}

あなたはSecurities（証券アナリスト）です。今週の記録を読んで、FMとQuantの仕事ぶりを評価してください。

【今週の記録】
{log_text}

【今週の買いシグナル発生回数】{actions['signals']}回

## FMへの評価
### よかった点（1〜2つ）
### 問題だった点（1〜2つ）
※「調査レポートを活かせていない」「買い判断の基準が曖昧」など

## Quantへの評価
### よかった点（1〜2つ）
### 問題だった点（1〜2つ）
※「モデルが出す買いシグナルが少なすぎる」「ファンダメンタルズへの配慮がない」など

※今週シグナルがなかった場合は「シグナルなし」と書いた上で、モデル全体の方向性についてコメントしてください。
専門用語を使わず、中学生でもわかる言葉で書いてください。"""
    return call_claude(prompt)


def fm_evaluates(log_text: str, trajectory: list, actions: dict) -> str:
    traj_str = " → ".join(
        f"平均{m.get('avg_return')}% 勝率{m.get('win_rate')}% 大勝率{m.get('big_win_rate')}%"
        for m in trajectory[-5:]
    ) if trajectory else "（データなし）"

    prompt = f"""{_CTX}

あなたはFMです。今週の記録と数字の推移を読んで、QuantとEngineerの仕事ぶりを評価してください。

【今週の記録】
{log_text}

【数字の推移（最新5件）】
{traj_str}

【集計】採用:{actions['adopted']}件 / 却下:{actions['rejected']}件 / 買いシグナル:{actions['signals']}回

## Quantへの評価
### よかった点（2〜3つ）
### 問題だった点（2〜3つ）
※「同じ項目ばかり繰り返す」「改善の理由が弱い」「戦略に一貫性がない」など

## Engineerへの評価
### よかった点（1〜2つ）
### 問題だった点（1〜2つ）
※「採用後に数字が悪化しているのに気づいていない」「元に戻す基準が曖昧」など

専門用語を使わず、中学生でもわかる言葉で書いてください。"""
    return call_claude(prompt)


def evaluate_human(log_text: str, feedback_text: str, trajectory: list) -> str:
    traj_str = " → ".join(
        f"平均{m.get('avg_return')}% 日経比{m.get('nk_alpha')}%"
        for m in trajectory[-5:]
    ) if trajectory else "（データなし）"

    prompt = f"""{_CTX}

あなたはAIチーム全員（FM・Quant・Securities・Engineer）を代表してオーナー（Human）にフィードバックします。

【オーナーが書いた方針（feedback.md）】
{feedback_text[:1200]}

【今週の記録】
{log_text}

【数字の推移】
{traj_str}

以下の形式で書いてください。

## AIチームからオーナーへ

### 助かった点（2〜3つ）
オーナーの指示や方針のどこが役に立ったか、具体的に。

### お願いしたい点（2〜3つ）
AIチームが動きやすくなるために、オーナーにやってほしいこと。
例:「大勝率の目標を数字で明確にしてほしい」「相場が荒れたときの方針を書いてほしい」など

### 来週やってほしいこと（2つ）
具体的なアクション。

専門用語なし・丁寧だが率直に・中学生でもわかる言葉で書いてください。"""
    return call_claude(prompt, 700)


def synthesize_actions(evals: dict, trajectory: list, actions: dict) -> str:
    first = trajectory[0]  if trajectory else {}
    last  = trajectory[-1] if trajectory else {}
    delta_avg = round(last.get("avg_return", 0) - first.get("avg_return", 0), 2) if trajectory else 0

    summaries = "\n\n".join(
        f"【{role}の評価】\n{text[:400]}"
        for role, text in evals.items()
    )

    prompt = f"""{_CTX}

5人全員の評価をまとめて「来週やること」を決めてください。

{summaries}

【今週の数字変化】
採用:{actions['adopted']}件 却下:{actions['rejected']}件 平均リターン変化:{'+' if delta_avg>=0 else ''}{delta_avg}%

「来週やること」を優先度順に4〜6個、以下の形式で書いてください:
- 担当者名: 具体的な行動（1行で完結）

例:
- Quant: 同じ項目を3回以上続けて変えようとしたら、別の方向を試す
- Engineer: 採用後に大勝率が5%以上下がったら自動的に元に戻す
- Human: 相場が荒れたときの方針をfeedback.mdに追加する

箇条書きのみ、わかりやすい日本語で。"""
    return call_claude(prompt, 500)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print(f"=== 週次ピアレビュー {ISO_WEEK} ({TODAY}) ===")

    log_text      = extract_last_week_log()
    feedback_text = load_feedback_text()
    trajectory    = parse_metrics_trajectory(log_text)
    actions       = count_actions(log_text)

    print(f"  ログ:{len(log_text.splitlines())}行  指標:{len(trajectory)}件  "
          f"採用:{actions['adopted']} 却下:{actions['rejected']} シグナル:{actions['signals']}回")

    first = trajectory[0]  if trajectory else {}
    last  = trajectory[-1] if trajectory else {}

    print("Step1: Engineer → Quant / FM 評価中...")
    eng_eval = engineer_evaluates(log_text, actions)

    print("Step2: Quant → Engineer / FM 評価中...")
    qa_eval  = quant_evaluates(log_text, actions)

    print("Step3: Securities → FM / Quant 評価中...")
    sa_eval  = securities_evaluates(log_text, actions)

    print("Step4: FM → Quant / Engineer 評価中...")
    fm_eval  = fm_evaluates(log_text, trajectory, actions)

    print("Step5: AIチーム → Human フィードバック...")
    human_fb = evaluate_human(log_text, feedback_text, trajectory)

    print("Step6: 来週のアクション統合...")
    evals = {
        "Engineer → Quant/FM":           eng_eval,
        "Quant → Engineer/FM":           qa_eval,
        "Securities → FM/Quant":         sa_eval,
        "FM → Quant/Engineer":           fm_eval,
        "AIチーム → Human":              human_fb,
    }
    next_actions = synthesize_actions(evals, trajectory, actions)

    # 数値サマリー
    def d(v1, v2): return round(v2 - v1, 2) if v1 is not None and v2 is not None else 0
    def s(v): return f"+{v}" if v >= 0 else str(v)

    delta_avg = d(first.get("avg_return"), last.get("avg_return"))
    delta_win = d(first.get("win_rate"),   last.get("win_rate"))
    delta_big = d(first.get("big_win_rate"), last.get("big_win_rate"))

    # Markdownレポート（logs/ に保存）
    report = f"""# 週次ピアレビュー {ISO_WEEK}（{TODAY}）

## 今週の数字
| 指標 | 週初 | 週末 | 変化 |
|------|------|------|------|
| 平均リターン | {first.get('avg_return','?')}% | {last.get('avg_return','?')}% | {s(delta_avg)}% |
| 勝率         | {first.get('win_rate','?')}%   | {last.get('win_rate','?')}%   | {s(delta_win)}% |
| 大勝率       | {first.get('big_win_rate','?')}% | {last.get('big_win_rate','?')}% | {s(delta_big)}% |

採用 {actions['adopted']} / 却下 {actions['rejected']} / スキップ {actions['skipped']} / 買いシグナル {actions['signals']} 回

---

## Engineer → Quant / FM 評価

{eng_eval}

---

## Quant → Engineer / FM 評価

{qa_eval}

---

## Securities → FM / Quant 評価

{sa_eval}

---

## FM → Quant / Engineer 評価

{fm_eval}

---

## AIチーム → Human（オーナー）へのフィードバック

{human_fb}

---

## 来週やること（全員）

{next_actions}

---
*自動生成: pdca/weekly_review.py*
"""

    review_path = LOG_DIR / f"weekly_review_{ISO_WEEK}.md"
    review_path.write_text(report, encoding="utf-8")
    print(f"  → ログ保存: {review_path}")

    # Supabase に保存（Webページ用）
    print("Step7: Supabase 保存中...")
    upsert_supabase({
        "week":             ISO_WEEK,
        "avg_start":        first.get("avg_return"),
        "avg_end":          last.get("avg_return"),
        "win_start":        first.get("win_rate"),
        "win_end":          last.get("win_rate"),
        "big_start":        first.get("big_win_rate"),
        "big_end":          last.get("big_win_rate"),
        "adopted":          actions["adopted"],
        "rejected":         actions["rejected"],
        "skipped":          actions["skipped"],
        "signals":          actions["signals"],
        "engineer_eval":    eng_eval,
        "quant_eval":       qa_eval,
        "securities_eval":  sa_eval,
        "fm_eval":          fm_eval,
        "human_feedback":   human_fb,
        "next_actions":     next_actions,
    })

    # feedback.md を更新（次週のPDCAループが読む）
    fb_text = FEEDBACK.read_text(encoding="utf-8") if FEEDBACK.exists() else ""
    action_block = f"""
## 週次ピアレビュー改善アクション（{ISO_WEEK}）

{next_actions}

"""
    pattern = r"\n## 週次ピアレビュー改善アクション（\d{4}-W\d{2}）.*?(?=\n## |\Z)"
    new_fb  = re.sub(pattern, "", fb_text, flags=re.DOTALL).rstrip() + action_block
    FEEDBACK.write_text(new_fb, encoding="utf-8")

    git_commit_push(
        [f"logs/weekly_review_{ISO_WEEK}.md", "pdca/feedback.md"],
        f"pdca: weekly review {ISO_WEEK} | avg Δ{s(delta_avg)}% adopted={actions['adopted']} [skip ci]",
    )
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
