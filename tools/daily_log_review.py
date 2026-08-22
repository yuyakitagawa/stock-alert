"""
tools/daily_log_review.py

毎日のGitHub Actions実行ログを集め、Claudeに「プロフェッショナルの観点」（SRE／クオンツ運用／
プロダクト）でレビューさせ、気づきと改善提案を GitHub Issue（全文）と LINE（要約）へ届ける。

流れ:
  1. `gh run list` で直近 --since-hours 以内に完了した対象ワークフローの run を列挙
  2. `gh run view --json jobs` でステップごとの成否・所要時間、`gh run view --log` で本文ログを取得
  3. ログはそのままでは数万行あるので、エラー/警告/Traceback/HTTPエラー/スキップ通知の行と
     各ステップの先頭・末尾だけに圧縮する（condense_log）
  4. Claude に日本語Markdownのレビュー（健全性 / 異常・注意点 / 改善提案(優先度付き) / LINE要約）を書かせる
  5. 全文: $GITHUB_STEP_SUMMARY と `daily-review` ラベルのIssue（1本を使い回してコメント追記）
     要約: LINE push（LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID）

必要な環境変数: GH_TOKEN（actions:read, issues:write）, ANTHROPIC_API_KEY,
              LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID（未設定ならLINEはスキップ）

使い方:
  python tools/daily_log_review.py                 # 本番（Issue追記 + LINE）
  python tools/daily_log_review.py --dry-run       # レビューを標準出力に出すだけ（Issue/LINEなし）
  python tools/daily_log_review.py --since-hours 48
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-opus-5"
JST = timezone(timedelta(hours=9))

# レビュー対象のワークフロー（ファイル名）。ci.yml はPR単位なので対象外。
WORKFLOWS = [
    "daily_alert.yml",
    "edinet_blog.yml",
    "x_post.yml",
    "video_post.yml",
    "ops.yml",
]

# 圧縮時に必ず残す行（エラー・警告・重要イベント）
_KEEP_PATTERN = re.compile(
    r"(Traceback|Error|ERROR|Exception|Failed|FAILED|failed|WARN|Warning|"
    r"HTTP [45]\d\d|status[_ ]code|Timeout|timed out|rate limit|Rate limit|429|"
    r"⚠|❌|🚨|失敗|エラー|例外|スキップ|未設定|取得できません|件数が0|0件|"
    r"exit code|##\[error\]|##\[warning\]|Process completed with exit code|"
    # SEO/GEOレビュー用: 記事タイトル・投稿・見送りの行は必ず残す
    r"タイトル|title|投稿しました|公開しました|記事化|見送り|重複|dealType|tweet|ツイート|X投稿|YouTube)"
)
_MAX_CHARS_PER_RUN = 40_000
_HEAD_LINES = 15
_TAIL_LINES = 40
_ISSUE_LABEL = "daily-review"
_ISSUE_TITLE = "日次ログレビュー（AIフィードバック）"
_LINE_MAX_CHARS = 1000

SYSTEM_PROMPT = """あなたは個人開発の株式アラートシステム「stock-alert」（日本株の下落確率ランキングをXGBoostで算出し、
LINE配信・ブログ自動生成（microCMS → kujira-watch.com「大口投資家の監視ブログ」）・X投稿・YouTube動画投稿を
GitHub Actionsで毎日回している）の運用レビュアーです。
次の4人のプロフェッショナルの視点を併せ持って、渡された当日のワークフロー実行ログを診断してください。

1. SRE: 失敗・タイムアウト・リトライ・所要時間の異常・continue-on-errorで握り潰されている失敗・
   外部API（Yahoo Finance / EDINET / J-Quants / Supabase / microCMS / X / YouTube / Anthropic）の劣化兆候。
2. クオンツ運用: ランキング生成・モデル学習・データ取得件数の異常（0件、急減、古い日付のまま）、
   データ鮮度、PIT（point-in-time）規律が崩れそうな箇所。
3. プロダクト/コンテンツ: ブログ・X・動画投稿の品質・件数・重複・スキップ理由、ユーザーに届く成果物の状態。
4. SEO/GEO（Generative Engine Optimization＝ChatGPT/Gemini/Perplexity等のAI検索に引用されるための最適化）:
   ログに現れる記事タイトル・1文目・スキップ/見送り理由・投稿件数・X投稿文から、
   - 検索意図との一致（銘柄名+コード+「大量保有報告書」等の検索語がタイトルに入っているか、同型タイトルの量産で
     カニバリゼーションが起きていないか、薄い記事（事実の並置だけ）になっていないか）
   - AI引用されやすさ（冒頭に直答文があるか、数値・日付・出所が明記されているか、推測が「※推測:」で分離されているか、
     FAQ/定義文など引用単位として切り出しやすい構造があるか）
   - 投稿の鮮度と量（開示当日中に記事化できた件数／翌日以降に遅れた件数／見送り件数。見送り理由が妥当か）
   - X投稿がブログ記事への導線として機能しているか（投稿文の訴求・絵文字過多・同文連投）
   を評価し、記事生成プロンプト（web/publish_blog_articles.py）やタイトルテンプレート、
   X投稿フォーマット（web/x_post_format.py）への具体的な改善提案を出す。
   ※インデックス反映には数日〜数週間かかるため「すぐ検索順位を確認せよ」といった提案はしない。

ルール:
- ログに根拠がある事だけを書く。推測は「※推測:」と明記する。無い問題を作らない。
- 「問題なし」の項目は1行で済ませ、異常があった項目に紙幅を使う。
- 改善提案は「何を・どのファイル/ステップで・なぜ」まで具体的に。優先度を 高/中/低 で付け、
  大規模リファクタリングより小さなバグ修正・パラメータ調整・監視追加を優先する。
- 略語（AUC、PIT など）は初出時にカッコで日本語説明を添える。
- 出力は日本語Markdown。必ず以下の見出し構成にする（見出し文字列は変えない）:

## 健全性サマリー
（ワークフローごとに ✅/⚠️/❌ と1行コメント）

## 異常・注意点
（根拠となるログ行を短く引用しつつ箇条書き。無ければ「特になし」）

## SEO・GEO所見
（当日生成されたコンテンツについて、上記4の観点での評価と具体的な改善提案。記事が1件も生成されていない日は
 「本日は生成なし」とし、見送り理由の妥当性だけコメントする）

## 改善提案
（運用面＋SEO/GEO面をまとめた優先度付き箇条書き。「[高] 提案: … / 対象: … / 理由: …」の形式）

## LINE要約
（スマホで読む前提。500字以内、絵文字OK。最重要の異常1〜2件、SEO/GEOの気づき1件、最優先の改善提案1件だけ。
 この節の本文だけがLINEに送られる）
"""


# ---------------------------------------------------------------------------
# GitHub Actions からの取得
# ---------------------------------------------------------------------------
def _gh(args: list[str], timeout: int = 120) -> str:
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def list_recent_runs(workflow: str, since: datetime, repo: str | None = None) -> list[dict]:
    """直近 since 以降に作成された完了済みrunを新しい順で返す。"""
    args = [
        "run", "list", "--workflow", workflow, "--limit", "20",
        "--json", "databaseId,conclusion,status,createdAt,updatedAt,displayTitle,url,event",
    ]
    if repo:
        args += ["--repo", repo]
    runs = json.loads(_gh(args) or "[]")
    return filter_runs(runs, since)


def filter_runs(runs: list[dict], since: datetime) -> list[dict]:
    out = []
    for r in runs:
        created = datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
        if created >= since and r.get("status") == "completed":
            out.append(r)
    return out


def fetch_run_detail(run_id: int, repo: str | None = None) -> tuple[list[dict], str]:
    """(jobs情報, 生ログ) を返す。ログ取得失敗時は空文字。"""
    args = ["run", "view", str(run_id), "--json", "jobs"]
    if repo:
        args += ["--repo", repo]
    jobs = json.loads(_gh(args) or "{}").get("jobs", [])
    try:
        log_args = ["run", "view", str(run_id), "--log"]
        if repo:
            log_args += ["--repo", repo]
        log = _gh(log_args, timeout=300)
    except Exception as e:  # ログが大きすぎる/期限切れ等
        log = f"(ログ取得失敗: {e})"
    return jobs, log


# ---------------------------------------------------------------------------
# 圧縮・整形
# ---------------------------------------------------------------------------
def _strip_prefix(line: str) -> str:
    """`gh run view --log` の `job\\tstep\\ttimestamp message` 形式から message だけ取り出す。"""
    parts = line.split("\t", 2)
    if len(parts) == 3:
        msg = parts[2]
        # 先頭のISOタイムスタンプを落とす
        msg = re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?", "", msg)
        return parts[1], msg
    return "", line


def condense_log(raw: str, max_chars: int = _MAX_CHARS_PER_RUN) -> str:
    """ステップごとに 先頭N行 + 重要行 + 末尾M行 だけ残す。"""
    by_step: dict[str, list[str]] = {}
    order: list[str] = []
    for line in raw.splitlines():
        step, msg = _strip_prefix(line)
        if not msg.strip():
            continue
        if step not in by_step:
            by_step[step] = []
            order.append(step)
        by_step[step].append(msg)

    chunks = []
    for step in order:
        lines = by_step[step]
        keep_idx = set(range(min(_HEAD_LINES, len(lines))))
        keep_idx |= set(range(max(0, len(lines) - _TAIL_LINES), len(lines)))
        keep_idx |= {i for i, l in enumerate(lines) if _KEEP_PATTERN.search(l)}
        kept = []
        last = -1
        for i in sorted(keep_idx):
            if i != last + 1 and last >= 0:
                kept.append(f"… ({i - last - 1}行省略) …")
            kept.append(lines[i][:400])
            last = i
        chunks.append(f"### step: {step or '(unknown)'} ({len(lines)}行)\n" + "\n".join(kept))

    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        # 前後を残して中央を落とす（末尾の失敗ステップを切らないため）
        half = max_chars // 2
        text = text[:half] + f"\n\n… (中央 {len(text) - max_chars} 文字省略) …\n\n" + text[-half:]
    return text


def format_jobs(jobs: list[dict]) -> str:
    lines = []
    for job in jobs:
        lines.append(f"job: {job.get('name')} → {job.get('conclusion')}")
        for st in job.get("steps", []):
            started, completed = st.get("startedAt"), st.get("completedAt")
            dur = ""
            if started and completed:
                s = datetime.fromisoformat(started.replace("Z", "+00:00"))
                c = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                dur = f" {int((c - s).total_seconds())}s"
            mark = {"success": "✅", "failure": "❌", "skipped": "⏭", "cancelled": "🚫"}.get(
                st.get("conclusion"), "•"
            )
            lines.append(f"  {mark} {st.get('name')}{dur}")
    return "\n".join(lines)


def build_review_input(runs_by_workflow: dict[str, list[dict]], date_label: str) -> str:
    parts = [f"# レビュー対象日: {date_label}（JST）\n"]
    for wf, runs in runs_by_workflow.items():
        if not runs:
            parts.append(f"## {wf}\n（直近の実行なし）\n")
            continue
        for r in runs:
            parts.append(
                f"## {wf} — run {r['databaseId']} ({r.get('event')}, {r.get('conclusion')}, "
                f"開始 {r['createdAt']}, 終了 {r.get('updatedAt')})\n{r.get('url', '')}\n\n"
                f"```\n{r['_jobs_text']}\n```\n\n{r['_log_text']}\n"
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Claude レビュー
# ---------------------------------------------------------------------------
def review_with_claude(review_input: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": review_input}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claudeがレビューを拒否しました")
    text = "".join(b.text for b in msg.content if b.type == "text")
    print(
        f"[review] tokens in={msg.usage.input_tokens} out={msg.usage.output_tokens} "
        f"stop={msg.stop_reason}"
    )
    return text


def extract_line_summary(review_md: str, max_chars: int = _LINE_MAX_CHARS) -> str:
    m = re.search(r"^## LINE要約\s*\n(.*?)(?=^## |\Z)", review_md, re.S | re.M)
    body = (m.group(1) if m else review_md).strip()
    body = re.sub(r"^\s*[（(].*?[）)]\s*$", "", body, flags=re.M).strip()  # 見出し直下の注釈行
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    return body


def build_line_message(date_label: str, summary: str, issue_url: str | None) -> str:
    lines = [f"🧑‍💻 {date_label} 日次ログレビュー", "", summary]
    if issue_url:
        lines += ["", f"全文: {issue_url}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 配信
# ---------------------------------------------------------------------------
def send_line(message: str) -> bool:
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = os.getenv("LINE_USER_ID")
    if not token or not user_id:
        print("[line] LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID 未設定のためスキップ")
        return False
    resp = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        json={"to": user_id, "messages": [{"type": "text", "text": message}]},
        timeout=15,
    )
    if resp.ok:
        print("[line] 📱 送信しました")
        return True
    print(f"[line] ⚠ 送信失敗 HTTP {resp.status_code}: {resp.text[:200]}")
    return False


def post_issue_comment(body: str, repo: str | None) -> str | None:
    """daily-review ラベルのオープンIssueに追記（無ければ作成）。Issue URLを返す。"""
    repo_args = ["--repo", repo] if repo else []
    existing = _gh(
        ["issue", "list", *repo_args, "--state", "open", "--label", _ISSUE_LABEL,
         "--limit", "1", "--json", "number,url"]
    )
    issues = json.loads(existing or "[]")
    body_path = REPO_ROOT / "logs" / "_daily_review_body.md"
    body_path.parent.mkdir(exist_ok=True)
    body_path.write_text(body, encoding="utf-8")
    try:
        if issues:
            num, url = issues[0]["number"], issues[0]["url"]
            _gh(["issue", "comment", str(num), *repo_args, "--body-file", str(body_path)])
            return url
        subprocess.run(
            ["gh", "label", "create", _ISSUE_LABEL, *repo_args,
             "--description", "日次ログレビュー", "--color", "5319e7"],
            capture_output=True, cwd=REPO_ROOT,
        )
        out = _gh(["issue", "create", *repo_args, "--title", _ISSUE_TITLE,
                   "--label", _ISSUE_LABEL, "--body-file", str(body_path)])
        return out.strip().splitlines()[-1] if out.strip() else None
    finally:
        body_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=int, default=24)
    ap.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY"))
    ap.add_argument("--dry-run", action="store_true", help="Issue/LINEに送らず標準出力のみ")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=args.since_hours)
    date_label = now.astimezone(JST).strftime("%Y-%m-%d")

    runs_by_workflow: dict[str, list[dict]] = {}
    total = 0
    for wf in WORKFLOWS:
        try:
            runs = list_recent_runs(wf, since, args.repo)
        except Exception as e:
            print(f"[gh] {wf} の一覧取得失敗: {e}")
            runs = []
        for r in runs:
            jobs, log = fetch_run_detail(r["databaseId"], args.repo)
            r["_jobs_text"] = format_jobs(jobs)
            r["_log_text"] = condense_log(log)
        runs_by_workflow[wf] = runs
        total += len(runs)
        print(f"[gh] {wf}: {len(runs)} run(s)")

    if total == 0:
        print("対象runが無いためレビューをスキップします")
        return 0

    review_input = build_review_input(runs_by_workflow, date_label)
    print(f"[review] 入力 {len(review_input):,} 文字")
    review_md = review_with_claude(review_input)

    run_url = ""
    if os.getenv("GITHUB_SERVER_URL") and os.getenv("GITHUB_RUN_ID") and args.repo:
        run_url = f"{os.environ['GITHUB_SERVER_URL']}/{args.repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    full_body = f"# {date_label} 日次ログレビュー\n\n{review_md}\n\n---\n[レビュー実行ログ]({run_url})" if run_url else review_md

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(full_body, encoding="utf-8")

    if args.dry_run:
        print(full_body)
        print("\n--- LINE ---\n" + build_line_message(date_label, extract_line_summary(review_md), None))
        return 0

    issue_url = None
    try:
        issue_url = post_issue_comment(full_body, args.repo)
        print(f"[issue] {issue_url}")
    except Exception as e:
        print(f"[issue] ⚠ Issue更新失敗: {e}")
    send_line(build_line_message(date_label, extract_line_summary(review_md), issue_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
