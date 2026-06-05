#!/usr/bin/env python3
"""週次ピアレビュー: 5ロール相互評価 + Human への提言
   ロール構成:
     • Fund Manager (FM)     — improve/skip判断・投資推奨
     • Quant Analyst         — モデルパラメータ改善提案
     • Securities Analyst    — S買い銘柄の企業調査
     • Engineer              — 実装・バックテスト・採用/revert
     • Human                 — feedback.md 記入・目標設定・実際の投資判断（評価を受けるのみ）
   毎週月曜 GitHub Actions から実行（weekly_review.yml）
"""
import sys, os, re, json, subprocess, smtplib
from pathlib import Path
from datetime import date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

BASE_DIR  = Path(__file__).resolve().parent.parent
PDCA_DIR  = Path(__file__).resolve().parent
LOG_DIR   = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PDCA_LOG  = PDCA_DIR / "pdca_log.md"
FEEDBACK  = PDCA_DIR / "feedback.md"
TODAY     = date.today()
ISO_WEEK  = TODAY.strftime("%Y-W%W")

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env", override=True)

try:
    import anthropic
    client = anthropic.Anthropic()
except ImportError:
    print("anthropic パッケージが必要です"); sys.exit(1)


# ── ユーティリティ ────────────────────────────────────────────────────────────

def call_claude(prompt: str, max_tokens: int = 900) -> str:
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
    """pdca_log.md から過去7日分のエントリを抽出"""
    if not PDCA_LOG.exists():
        return "（ログなし）"
    text  = PDCA_LOG.read_text(encoding="utf-8")
    lines = text.splitlines()
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
    trajectory = []
    for line in log_text.splitlines():
        m = re.search(r"- metrics: (\{.*?\})", line)
        if m:
            try:
                trajectory.append(json.loads(m.group(1)))
            except Exception:
                pass
    return trajectory


def count_actions(log_text: str) -> dict:
    signals_count = len(re.findall(r"- signals: \d+銘柄", log_text))
    return {
        "adopted":     len(re.findall(r"✅ 採用",    log_text)),
        "rejected":    len(re.findall(r"❌ 改善なし", log_text)),
        "skipped":     len(re.findall(r"- skip\b",   log_text)),
        "parse_error": len(re.findall(r"parse error", log_text)),
        "signals":     signals_count,
    }


# ── システム共通コンテキスト ──────────────────────────────────────────────────

SYSTEM_CONTEXT = """このシステムは日本株XGBoost予測モデルの自律PDCAループです。
最終目標: 元本300万円を10年で1億円（年率42%）。

【5つのロール】
  • Fund Manager (FM)     : backtest指標を見てimprove/skip判断、S買い銘柄の投資推奨決定
  • Quant Analyst         : モデルハイパーパラメータ・フィルター定数の改善案を提案
  • Securities Analyst    : S買いシグナル銘柄の企業調査レポートをFMに提出（web検索）
  • Engineer              : 提案を実装→backtest→採用またはrevert→commit
  • Human                 : feedback.md に目標・制約を記述、実際の投資判断を行う（評価は受けるのみ）

【投資フェーズ目標指標】
  avg_return > 2%/21日, win_rate > 50%, big_win_rate > 20%, nk_alpha > 0%（日経超過）
"""


# ── 各ロールの評価関数 ────────────────────────────────────────────────────────

def engineer_evaluates(log_text: str, actions: dict) -> str:
    prompt = f"""{SYSTEM_CONTEXT}

あなたはエンジニアです。今週の活動ログを読み、以下2ロールをGOOD/BADで評価してください。

【今週のPDCAログ】
{log_text}

【集計】採用:{actions['adopted']} / 却下:{actions['rejected']} / parseエラー:{actions['parse_error']}

## Quant Analyst への評価
### GOOD（実装できた・正確だった点 2〜3つ）
### BAD（パラメータ名ミス・同変数の繰り返し・根拠薄弱など 2〜3つ）

## Fund Manager への評価
### GOOD（適切な指示・目標設定 1〜2つ）
### BAD（常にimproveで思考停止・非現実的な目標など 1〜2つ）

ログの具体的な行を引用しながら日本語で書いてください。"""
    return call_claude(prompt)


def quant_analyst_evaluates(log_text: str, actions: dict) -> str:
    prompt = f"""{SYSTEM_CONTEXT}

あなたは数量アナリストです。今週の活動ログを読み、以下2ロールをGOOD/BADで評価してください。

【今週のPDCAログ】
{log_text}

【集計】採用:{actions['adopted']} / 却下:{actions['rejected']} / parseエラー:{actions['parse_error']}

## Engineer への評価
### GOOD（実装精度・採用/revert判断の適切さ 2〜3つ）
### BAD（採用基準の一貫性・backtest解釈の誤りなど 2〜3つ）

## Fund Manager への評価
### GOOD（改善方向の指示・目標設定の妥当性 1〜2つ）
### BAD（市場フェーズを無視・big_win_rate目標が非現実的など 1〜2つ）

ログの具体的な行を引用しながら日本語で書いてください。"""
    return call_claude(prompt)


def securities_analyst_evaluates(log_text: str, actions: dict) -> str:
    prompt = f"""{SYSTEM_CONTEXT}

あなたは証券アナリストです。S買いシグナル銘柄の企業調査を担当しています。
今週の活動ログを読み、以下2ロールをGOOD/BADで評価してください。

【今週のPDCAログ】
{log_text}

【集計】S買いシグナル発生:{actions['signals']}回

## Fund Manager への評価
### GOOD（購入推奨基準の明確さ・リスク管理の適切さ 1〜2つ）
### BAD（調査結果の活用不足・投資判断基準が曖昧など 1〜2つ）

## Quant Analyst への評価
### GOOD（モデルが出したシグナルの質・スクリーニング精度 1〜2つ）
### BAD（シグナルが少なすぎる・ファンダ面の考慮が薄いなど 1〜2つ）

※今週S買いシグナルがなかった場合は「シグナルなし週のため評価材料不足」と明記し、
  モデル全体の方向性についてコメントしてください。
ログの具体的な行を引用しながら日本語で書いてください。"""
    return call_claude(prompt)


def fm_evaluates(log_text: str, trajectory: list, actions: dict) -> str:
    traj_str = " → ".join(
        f"avg={m.get('avg_return')}% win={m.get('win_rate')}% big={m.get('big_win_rate')}%"
        for m in trajectory[-5:]
    ) if trajectory else "（データなし）"

    prompt = f"""{SYSTEM_CONTEXT}

あなたはファンドマネージャーです。今週のログと指標推移を読み、以下2ロールをGOOD/BADで評価してください。

【今週のPDCAログ】
{log_text}

【指標推移（最新5件）】
{traj_str}

【集計】採用:{actions['adopted']} / 却下:{actions['rejected']} / S買いシグナル:{actions['signals']}回

## Quant Analyst への評価
### GOOD（提案の多様性・論拠の質・戦略的思考 2〜3つ）
### BAD（同パラメータの繰り返し・big_win_rate対策の遅れなど 2〜3つ）

## Engineer への評価
### GOOD（実装の信頼性・採用判断の適切さ 1〜2つ）
### BAD（採用後の退行見落とし・revert基準の曖昧さなど 1〜2つ）

ログの具体的な行を引用しながら日本語で書いてください。"""
    return call_claude(prompt)


def evaluate_human(log_text: str, feedback_text: str, trajectory: list) -> str:
    first = trajectory[0]  if trajectory else {}
    last  = trajectory[-1] if trajectory else {}
    traj_str = " → ".join(
        f"avg={m.get('avg_return')}% nk_alpha={m.get('nk_alpha')}%"
        for m in trajectory[-5:]
    ) if trajectory else "（データなし）"

    prompt = f"""{SYSTEM_CONTEXT}

あなたはAIチーム全体（FM・Quant Analyst・Securities Analyst・Engineer）を代表して、
Humanへのフィードバックをまとめてください。

Humanの役割:
  - feedback.md に目標・制約・優先順位を記述してAIに方向を与える
  - 実際の投資判断（買う/買わない）を下す
  - モデル開発の最終承認者

【今週のfeedback.md（Human が書いたもの）】
{feedback_text[:1500]}

【今週のPDCAログ】
{log_text}

【指標推移】
{traj_str}

以下の観点でHumanへのフィードバックを作成してください。

## AIチームからHumanへ

### GOOD（Human の指示・フィードバックで助かった点 2〜3つ）
具体的にfeedback.mdのどの記述がどう役立ったかを示す。

### BAD（Human の指示で困った点・改善してほしい点 2〜3つ）
例:「目標数値の根拠がない」「フィードバック頻度が低い」「big_win_rateとwin_rateの優先度が不明確」など

### Human への来週のお願い（具体的なアクション 2〜3つ）
AIチームが来週より良く動くために、Humanにやってほしいことを明確に。
例:「bear期の損失許容ラインを数値で設定してほしい」「週1回はシグナルの質を主観評価してほしい」

日本語で丁寧かつ率直に書いてください。"""
    return call_claude(prompt, 800)


def synthesize_actions(evals: dict, trajectory: list, actions: dict) -> str:
    first = trajectory[0]  if trajectory else {}
    last  = trajectory[-1] if trajectory else {}
    delta_avg = round(last.get("avg_return", 0) - first.get("avg_return", 0), 2) if trajectory else 0

    summaries = "\n\n".join(
        f"【{role}の評価】\n{text[:500]}"
        for role, text in evals.items()
    )

    prompt = f"""{SYSTEM_CONTEXT}

5ロール全員の相互評価を統合して「来週の改善アクション」を決定してください。

{summaries}

【週次サマリー】
- 採用:{actions['adopted']} 却下:{actions['rejected']} parseエラー:{actions['parse_error']} S買い:{actions['signals']}回
- 指標変動（週初→週末）: avg_return Δ{delta_avg}%

「来週の改善アクション」を優先度順に4〜6件、以下の形式で出力してください:
- ロール名: 具体的な行動（1行で完結）

例:
- Quant Analyst: 同一パラメータを3回以上連続提案した場合は別アプローチへ切り替える
- Engineer: big_win_rateが採用前比-5%超になった場合は即revert
- Human: big_win_rateとnk_alphaの週次トレンドを確認し、方向性が悪化したらfeedback.mdにコメントを追加する

箇条書きのみ、日本語で出力してください。"""
    return call_claude(prompt, 600)


# ── メール送信 ────────────────────────────────────────────────────────────────

NOTIFY_TO = "dosankoure@gmail.com"


def send_review_email(subject: str, eng_eval: str, qa_eval: str, sa_eval: str,
                      fm_eval: str, human_fb: str, next_actions: str, summary: dict):
    gmail_addr = os.getenv("GMAIL_ADDRESS")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail_addr or not gmail_pass:
        print("  [メールスキップ] GMAIL_ADDRESS / GMAIL_APP_PASSWORD 未設定")
        return

    s = summary
    first, last, actions = s["first"], s["last"], s["actions"]

    def signed(v): return f"+{v}" if v >= 0 else str(v)

    body = f"""週次ピアレビュー {ISO_WEEK}（{TODAY}）

━━━━━━━━━━━━━━━━━━━━
📈 今週のサマリー
━━━━━━━━━━━━━━━━━━━━
  avg_return   : {first.get('avg_return','?')}% → {last.get('avg_return','?')}%  ({signed(s['delta_avg'])}%)
  win_rate     : {first.get('win_rate','?')}% → {last.get('win_rate','?')}%  ({signed(s['delta_win'])}%)
  big_win_rate : {first.get('big_win_rate','?')}% → {last.get('big_win_rate','?')}%  ({signed(s['delta_big'])}%)

  採用: {actions['adopted']} / 却下: {actions['rejected']} / スキップ: {actions['skipped']} / parseエラー: {actions['parse_error']} / S買いシグナル: {actions['signals']} 回


━━━━━━━━━━━━━━━━━━━━
🔧 Engineer → Quant Analyst / FM 評価
━━━━━━━━━━━━━━━━━━━━
{eng_eval}


━━━━━━━━━━━━━━━━━━━━
📐 Quant Analyst → Engineer / FM 評価
━━━━━━━━━━━━━━━━━━━━
{qa_eval}


━━━━━━━━━━━━━━━━━━━━
🔍 Securities Analyst → FM / Quant Analyst 評価
━━━━━━━━━━━━━━━━━━━━
{sa_eval}


━━━━━━━━━━━━━━━━━━━━
💼 Fund Manager → Quant Analyst / Engineer 評価
━━━━━━━━━━━━━━━━━━━━
{fm_eval}


━━━━━━━━━━━━━━━━━━━━
💬 AIチーム → あなた（Human）へのフィードバック
━━━━━━━━━━━━━━━━━━━━
{human_fb}


━━━━━━━━━━━━━━━━━━━━
✅ 来週の改善アクション（全ロール）
━━━━━━━━━━━━━━━━━━━━
{next_actions}


━━━━━━━━━━━━━━━━━━━━
詳細ログ: logs/weekly_review_{ISO_WEEK}.md
"""

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = gmail_addr
    msg["To"]      = NOTIFY_TO

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_addr, gmail_pass)
            server.sendmail(gmail_addr, NOTIFY_TO, msg.as_string())
        print(f"  → メール送信完了: {NOTIFY_TO}")
    except Exception as e:
        print(f"  [メール送信エラー] {e}")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print(f"=== 週次ピアレビュー {ISO_WEEK} ({TODAY}) ===")

    log_text      = extract_last_week_log()
    feedback_text = load_feedback_text()
    trajectory    = parse_metrics_trajectory(log_text)
    actions       = count_actions(log_text)

    print(f"  ログ行数: {len(log_text.splitlines())}  指標:{len(trajectory)}件  "
          f"採用:{actions['adopted']} 却下:{actions['rejected']} S買い:{actions['signals']}回")

    first = trajectory[0]  if trajectory else {}
    last  = trajectory[-1] if trajectory else {}

    print("Step1: Engineer → Quant Analyst / FM 評価中...")
    eng_eval = engineer_evaluates(log_text, actions)

    print("Step2: Quant Analyst → Engineer / FM 評価中...")
    qa_eval  = quant_analyst_evaluates(log_text, actions)

    print("Step3: Securities Analyst → FM / Quant Analyst 評価中...")
    sa_eval  = securities_analyst_evaluates(log_text, actions)

    print("Step4: Fund Manager → Quant Analyst / Engineer 評価中...")
    fm_eval  = fm_evaluates(log_text, trajectory, actions)

    print("Step5: AIチーム → Human フィードバック生成中...")
    human_fb = evaluate_human(log_text, feedback_text, trajectory)

    print("Step6: 来週の改善アクション統合中...")
    evals = {
        "Engineer → Quant Analyst/FM":           eng_eval,
        "Quant Analyst → Engineer/FM":           qa_eval,
        "Securities Analyst → FM/Quant Analyst": sa_eval,
        "FM → Quant Analyst/Engineer":           fm_eval,
        "AIチーム → Human":                      human_fb,
    }
    next_actions = synthesize_actions(evals, trajectory, actions)

    # ── レポート生成 ──
    delta_avg = round(last.get("avg_return",    0) - first.get("avg_return",    0), 2) if trajectory else 0
    delta_win = round(last.get("win_rate",      0) - first.get("win_rate",      0), 2) if trajectory else 0
    delta_big = round(last.get("big_win_rate",  0) - first.get("big_win_rate",  0), 2) if trajectory else 0

    def signed(v): return f"+{v}" if v >= 0 else str(v)

    report = f"""# 週次ピアレビュー {ISO_WEEK}（{TODAY}）

## 今週のサマリー
| 指標 | 週初 | 週末 | 変化 |
|------|------|------|------|
| avg_return   | {first.get('avg_return','?')}%   | {last.get('avg_return','?')}%   | {signed(delta_avg)}% |
| win_rate     | {first.get('win_rate','?')}%     | {last.get('win_rate','?')}%     | {signed(delta_win)}% |
| big_win_rate | {first.get('big_win_rate','?')}% | {last.get('big_win_rate','?')}% | {signed(delta_big)}% |

**PDCAサイクル**: 採用 {actions['adopted']} / 却下 {actions['rejected']} / スキップ {actions['skipped']} / parseエラー {actions['parse_error']} / S買いシグナル {actions['signals']} 回

---

## Engineer → Quant Analyst / Fund Manager 評価

{eng_eval}

---

## Quant Analyst → Engineer / Fund Manager 評価

{qa_eval}

---

## Securities Analyst → Fund Manager / Quant Analyst 評価

{sa_eval}

---

## Fund Manager → Quant Analyst / Engineer 評価

{fm_eval}

---

## AIチーム → Human へのフィードバック

{human_fb}

---

## 来週の改善アクション（全ロール）

{next_actions}

---
*自動生成: pdca/weekly_review.py*
"""

    review_path = LOG_DIR / f"weekly_review_{ISO_WEEK}.md"
    review_path.write_text(report, encoding="utf-8")
    print(f"  → レポート保存: {review_path}")

    # メール送信
    print("Step7: メール送信中...")
    send_review_email(
        subject=f"📊 週次ピアレビュー {ISO_WEEK} | avg Δ{signed(delta_avg)}% 採用:{actions['adopted']} 却下:{actions['rejected']}",
        eng_eval=eng_eval,
        qa_eval=qa_eval,
        sa_eval=sa_eval,
        fm_eval=fm_eval,
        human_fb=human_fb,
        next_actions=next_actions,
        summary={
            "delta_avg": delta_avg, "delta_win": delta_win, "delta_big": delta_big,
            "first": first, "last": last, "actions": actions,
        },
    )

    # feedback.md に来週アクションを反映
    fb_text = FEEDBACK.read_text(encoding="utf-8") if FEEDBACK.exists() else ""
    action_block = f"""
## 週次ピアレビュー改善アクション（{ISO_WEEK}）

{next_actions}

"""
    pattern = r"\n## 週次ピアレビュー改善アクション（\d{4}-W\d{2}）.*?(?=\n## |\Z)"
    new_fb  = re.sub(pattern, "", fb_text, flags=re.DOTALL).rstrip() + action_block
    FEEDBACK.write_text(new_fb, encoding="utf-8")
    print("  → feedback.md 更新完了")

    commit_msg = (
        f"pdca: weekly review {ISO_WEEK} | "
        f"avg Δ{delta_avg}% adopted={actions['adopted']} rejected={actions['rejected']} [skip ci]"
    )
    git_commit_push(
        [f"logs/weekly_review_{ISO_WEEK}.md", "pdca/feedback.md"],
        commit_msg,
    )
    print(f"  → コミット&プッシュ完了")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
