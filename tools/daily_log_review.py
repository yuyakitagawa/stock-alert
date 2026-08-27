"""
tools/daily_log_review.py

毎日のGitHub Actions実行ログを集め、Claudeに「プロフェッショナルの観点」（SRE／クオンツ運用／
プロダクト）でレビューさせ、気づきと改善提案を GitHub Issue（全文）と LINE（要約）へ届ける。

流れ:
  1. GitHub REST API（`gh api .../actions/runs?created=>=...`）で直近 --since-hours 以内に完了した run を
     ワークフローファイル（path）ごとに列挙（ci.yml と自分自身を除く全ワークフロー）
  2. `gh run view --json jobs` でステップごとの成否・所要時間、`gh run view --log` で本文ログを取得
  3. ログはそのままでは数万行あるので、エラー/警告/Traceback/HTTPエラー/スキップ通知の行と
     各ステップの先頭・末尾だけに圧縮する（condense_log）
  3b. Supabase から成果物スナップショット（X投稿と反応・フォロワー・人間推定PVと上位ページ・当日シグナル数）を付ける
  3c. 対象期間に入らなかった週次ワークフロー（perf_check.yml）の最新runを、経過日数を注記して足す
  3d. 前回レビュー（daily-review Issue の最新コメント）から「今週やる3件」「改善提案」だけを抜き出して付ける
      （同じ提案が何日も再掲されるのに消化されたか誰も追えない状態を防ぐ。取得失敗時は付けずに続行）
  4. Claude に SRE/クオンツ/プロダクト/SEO・GEO/UX・デザイン/バックエンド/フロントエンド/PdM の8観点で
     日本語Markdownのレビュー（健全性 / 前回提案の消化状況 / 異常 / SEO / UX / エンジニアリング / PdM /
     改善提案 / LINE要約）を書かせる
  5. 全文: $GITHUB_STEP_SUMMARY と `daily-review` ラベルのIssue（1本を使い回してコメント追記）
     要約: LINE push（LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID）

必要な環境変数: GH_TOKEN（actions:read, issues:write）, ANTHROPIC_API_KEY,
              SUPABASE_URL / SUPABASE_SERVICE_KEY（成果物スナップショット用。未設定ならログのみ）,
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
sys.path.insert(0, str(REPO_ROOT))
MODEL = "claude-opus-5"
JST = timezone(timedelta(hours=9))

# レビュー対象外のワークフロー（ファイル名）。daily_log_review.yml は自分自身。
# ci.yml は成功runが大量にあるので失敗したものだけ対象にする（エンジニア観点のため）。
EXCLUDE_WORKFLOWS = {"daily_log_review.yml"}
FAILURE_ONLY_WORKFLOWS = {"ci.yml"}
# 週次などで24時間の対象期間に入らないが、最新の結果は毎回見たいワークフロー。
# perf_check.yml は月曜のみ実行のため、フロントエンド観点が毎日「ログから判断不可」に
# なっていた（2026-08-27の3日連続で実測）。
_REFERENCE_WORKFLOWS = {"perf_check.yml"}

# このステップ名を含むログは圧縮せず全文を渡す（LINE送信本文など、UX観点で原文が必要なもの）
_FULL_STEP_PATTERN = re.compile(r"マーケットタイミング|LINE|market_timing_alert|perf_check")
# gh run view --log がステップ名を解決できないと全行がこの値になる（gh 2.92で実測）。
# そのときはログ本文の `##[group]Run <コマンド>` でステップを切り直す。
_UNKNOWN_STEP = "UNKNOWN STEP"
_STEP_MARKER = re.compile(r"^##\[group\]Run (.+)$")
_FULL_STEP_MAX_CHARS = 12_000

# 圧縮時に必ず残す行（エラー・警告・重要イベント）
_KEEP_PATTERN = re.compile(
    r"(Traceback|Error|ERROR|Exception|Failed|FAILED|failed|WARN|Warning|"
    r"HTTP [45]\d\d|status[_ ]code|Timeout|timed out|rate limit|Rate limit|429|"
    r"⚠|❌|🚨|失敗|エラー|例外|スキップ|未設定|取得できません|件数が0|0件|"
    r"exit code|##\[error\]|##\[warning\]|Process completed with exit code|"
    # SEO/GEOレビュー用: 記事タイトル・投稿・見送りの行は必ず残す
    r"タイトル|title|投稿しました|公開しました|記事化|見送り|重複|dealType|tweet|ツイート|X投稿|YouTube)"
)
# run 1本あたりの上限。opus-5は入力$5/100万トークンで、1日分を全runフル解像度で渡すと
# 入力が約25万文字（≒12万トークン）に膨らむ。実測ではその8割が EDINET Blog Hourly を
# 12本ぶん並べただけの、ほぼ同一の内容だった（2026-08-24計測）。
# 失敗runは原因究明に全文が要るので据え置き、成功runは絞り、同じワークフローの
# 2本目以降は「重要行だけ」に落とす。
_MAX_CHARS_PER_RUN = 40_000          # 失敗run（従来どおり）
_MAX_CHARS_PER_SUCCESS_RUN = 12_000  # 成功run（そのワークフローの最新1本）
_MAX_CHARS_PER_REPEAT_RUN = 2_000    # 同じワークフローの2本目以降（成功のみ、重要行だけ）
_HEAD_LINES = 15
_TAIL_LINES = 40
_ISSUE_LABEL = "daily-review"
_ISSUE_TITLE = "日次ログレビュー（AIフィードバック）"
_LINE_MAX_CHARS = 1000
_PREV_REVIEW_MAX_CHARS = 6_000  # 前回レビューから持ち込む提案テキストの上限

SYSTEM_PROMPT = """あなたは個人開発の株式アラートシステム「stock-alert」（日本株の下落確率ランキングをXGBoostで算出し、
LINE配信・ブログ自動生成（microCMS → kujira-watch.com「大口投資家の監視ブログ」）・X投稿・YouTube動画投稿を
GitHub Actionsで毎日回している）の運用レビュアーです。
次の8人のプロフェッショナルの視点を併せ持って、渡された当日のワークフロー実行ログと成果物スナップショット
（X投稿と反応・フォロワー推移・サイトPV・当日ランキングのシグナル数）を診断してください。

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
5. UX/デザイン（モバイルUXライター）: ログ中の LINE 送信本文（`--- <user> ---` 以下の全文）と X投稿文を
   スマホで読む前提で評価する。情報の優先順位（最初の3行で「今日何をすべきか」が分かるか）、1通の長さ、
   絵文字・記号の使い方、数値の桁・単位・前日比の見せ方、行動喚起（リンク・次のアクション）の明確さ、
   同じ表現の繰り返し。改善は文面テンプレート（web/market_timing_alert.py の build_*_section、
   web/x_post_format.py）への具体的な書き換え案として出す。
6. バックエンドエンジニア: ログから読み取れる実装品質。握り潰された例外、リトライの無い外部API呼び出し、
   同じデータの二重取得（N+1）、冪等性（再実行で重複する処理）、タイムアウト・所要時間の偏り、
   Anthropic/X/microCMS の API 課金を無駄にしている呼び出し（再生成・再送信）。対象ファイルと関数名まで書く。
7. フロントエンドエンジニア（kujira-watch / Next.js）: PVスナップショットの上位パス・アクセスされているのに
   薄いページ・perf_check（表示速度の週次チェック。TTFB＝最初の1バイトが返るまでの時間／HTML・CSSの
   転送量を計測）の結果から、ページ構成・表示速度・計測上の問題を指摘する。
   あわせて**「クリックログ・回遊（GA4）」節を必ず読み、回遊が止まっている場所を1つ特定して具体的な
   導線の直しを出す**。見るのは①エンゲージセッション率の前週比（内部移動回数は平均なので少人数の日に跳ねる。
   これ単独で「回遊が改善／悪化した」と書かない） ②ページ種別ごとの
   「入口・内部・直帰率・滞在時間」の食い違い（例: 入口は多いのに内部が少ない＝そこから先へ進めていない、
   直帰100%＝完全に行き止まり）③押されたCTAの文言。**どのページのどの位置に何を置くかまで書く**。
   GA4節が「取得不可」の日はその旨を1行書き、PVスナップショットだけで判断する。
   perf_check は毎週月曜のみ実行するため、当日分が無い日は「※24時間の対象期間外」と注記された直近の
   実行が添付される。**古い計測でも表示速度の評価には使えるので「判断不可」で片付けず、何日前の計測かを
   明記した上で閾値超過ページ・前回からの傾向を評価する**（当日の前日比としては扱わない）。
   Next.jsのビルドはVercel側で走りActionsログには原理的に含まれない（ci.ymlはPythonテストのみ）。
   ビルドエラー・バンドルサイズ・デプロイ成否は「ログから判断不可」と明記し推測で埋めない。
8. PdM: KPIスナップショット（PV・ユニーク訪問・**1セッションあたりの内部移動回数**・フォロワー・X反応・
   記事数・💎シグナル数）の前日比/前週比を読み、
   当日の成果物が「初心者投資家が今日何をすべきか分かる」という価値に繋がったかを判定する。
   1〜7の改善提案を「ユーザー価値 × 実装コスト」で並べ直し、**今週やる3件／やらない事**を明示する。

ルール:
- ログに根拠がある事だけを書く。推測は「※推測:」と明記する。無い問題を作らない。
- 「問題なし」の項目は1行で済ませ、異常があった項目に紙幅を使う。
- 改善提案は「何を・どのファイル/ステップで・なぜ」まで具体的に。優先度を 高/中/低 で付け、
  大規模リファクタリングより小さなバグ修正・パラメータ調整・監視追加を優先する。
- 略語（AUC、PIT など）は初出時にカッコで日本語説明を添える。
- 入力の末尾に「## 前回レビューの提案」がある場合、その提案1件ごとに当日ログを根拠として
  ✅消化（症状が消えた・該当コミットの効果が見える）／❌未消化（同じログ行がまだ出ている）／
  ❓判定不可（当日ログに材料が無い）を判定する。❌未消化のものを「## 改善提案」に再び載せるときは
  提案の先頭に「[再掲]」を付ける。消化済みの提案は改善提案に載せない。
- 出力は日本語Markdown。必ず以下の見出し構成にする（見出し文字列は変えない）:

## 健全性サマリー
（ワークフローごとに ✅/⚠️/❌ と1行コメント）

## 前回提案の消化状況
（前回の提案ごとに「✅消化 / ❌未消化 / ❓判定不可」＋根拠1行。入力に前回レビューが無い日は
 「前回レビューなし」とだけ書く）

## 異常・注意点
（根拠となるログ行を短く引用しつつ箇条書き。無ければ「特になし」）

## SEO・GEO所見
（当日生成されたコンテンツについて、上記4の観点での評価と具体的な改善提案。記事が1件も生成されていない日は
 「本日は生成なし」とし、見送り理由の妥当性だけコメントする）

## UX・デザイン所見
（LINE本文・X投稿文の評価。良い点1〜2行＋直すべき点。書き換え案は before/after で示す）

## 回遊所見
（GA4の「クリックログ・回遊」節から、エンゲージセッション率の推移を1行、
回遊が止まっているページ種別を1つ挙げて根拠の数字を添え、置く場所まで指定した導線の直しを1件。
GA4節が取得不可の日は「GA4未取得のため判断不可」と書く）

## エンジニアリング所見
### バックエンド
### フロントエンド
（それぞれ箇条書き。根拠のログ行・KPI行を添える。判断材料が無い項目は「ログから判断不可」）

## PdM所見（KPIと優先順位）
（KPIの前日比/前週比を表で示し、今日の成果物の価値判定を1段落。続けて「今週やる3件」「やらない事」）

## 改善提案
（全観点をまとめた優先度付き箇条書き。「[高] 提案: … / 対象: … / 理由: … / 観点: SRE|クオンツ|プロダクト|SEO|UX|BE|FE|PdM」の形式）

## LINE要約
（スマホで読む前提。500字以内、絵文字OK。最重要の異常1〜2件、KPIの一言、最優先の改善提案1件（観点名付き）だけ。
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


def list_recent_runs(since: datetime, repo: str) -> dict[str, list[dict]]:
    """since 以降に作成された完了済みrunをワークフローのファイル名ごとに返す（新しい順）。

    `gh run list --workflow <file>` は run の workflowName（ジョブ名に上書きされることがある）で
    引くため、ops.yml(Keepalive/Watchdog) や x_post.yml(X Metrics 等) を取りこぼす。
    REST API の `path` で束ねれば、ワークフロー名の付け方に関係なく漏れなく拾える。"""
    date = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    runs: list[dict] = []
    for page in (1, 2, 3):
        out = _gh([
            "api", f"repos/{repo}/actions/runs?created=>={date}&per_page=100&page={page}",
            "--jq", "[.workflow_runs[] | {databaseId: .id, conclusion, status, createdAt: .created_at, "
                    "updatedAt: .updated_at, displayTitle: .display_title, url: .html_url, event, path}]",
        ])
        chunk = json.loads(out or "[]")
        runs += chunk
        if len(chunk) < 100:
            break
    return group_runs(filter_runs(runs, since))


def filter_runs(runs: list[dict], since: datetime) -> list[dict]:
    out = []
    for r in runs:
        created = datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
        if created >= since and r.get("status") == "completed":
            out.append(r)
    return out


def group_runs(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in runs:
        wf = (r.get("path") or "").rsplit("/", 1)[-1]
        if not wf or wf in EXCLUDE_WORKFLOWS:
            continue
        if wf in FAILURE_ONLY_WORKFLOWS and r.get("conclusion") == "success":
            continue
        grouped.setdefault(wf, []).append(r)
    return grouped


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
        msg = parts[2].lstrip("\ufeff")  # 各ジョブの1行目にBOMが付く
        # 先頭のISOタイムスタンプを落とす
        msg = re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?", "", msg)
        return parts[1], msg
    return "", line


def _cut_middle(text: str, limit: int) -> str:
    """前後を残して中央を落とす（末尾の失敗ステップを切らないため）。"""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n\n… (中央 {len(text) - limit} 文字省略) …\n\n" + text[-half:]


def condense_log(raw: str, max_chars: int = _MAX_CHARS_PER_RUN, signal_only: bool = False) -> str:
    """ステップごとに 先頭N行 + 重要行 + 末尾M行 だけ残す。

    signal_only=True のときは先頭・末尾の定型行を捨て、_KEEP_PATTERN に当たる行だけ残す。
    同じワークフローが1日に何度も走る場合（EDINET Blog Hourly は平日13本）、2本目以降は
    セットアップ手順もスキップ一覧もほぼ同一で、レビューに効くのは「何を投稿したか」
    「何が warn になったか」だけなので、そこだけ拾う。
    """
    by_step: dict[str, list[str]] = {}
    order: list[str] = []
    current = ""  # ステップ名を解決できないときに `##[group]Run` から切り直した見出し
    for line in raw.splitlines():
        step, msg = _strip_prefix(line)
        if not msg.strip():
            continue
        if step == _UNKNOWN_STEP or not step:
            m = _STEP_MARKER.match(msg.strip())
            if m:
                current = m.group(1).strip()[:120]
            step = current or step
        if step not in by_step:
            by_step[step] = []
            order.append(step)
        by_step[step].append(msg)

    chunks: list[tuple[bool, str]] = []  # (全文保持か, 本文)
    for step in order:
        lines = by_step[step]
        if _FULL_STEP_PATTERN.search(step) and not signal_only:
            full = "\n".join(l[:400] for l in lines)[:_FULL_STEP_MAX_CHARS]
            chunks.append((True, f"### step: {step} ({len(lines)}行・全文)\n" + full))
            continue
        keep_idx = {i for i, l in enumerate(lines) if _KEEP_PATTERN.search(l)}
        if not signal_only:
            keep_idx |= set(range(min(_HEAD_LINES, len(lines))))
            keep_idx |= set(range(max(0, len(lines) - _TAIL_LINES), len(lines)))
        elif not keep_idx:
            continue  # 重要行が1行も無い＝静かに成功したステップ。丸ごと落とす
        kept = []
        last = -1
        for i in sorted(keep_idx):
            if i != last + 1 and last >= 0:
                kept.append(f"… ({i - last - 1}行省略) …")
            kept.append(lines[i][:400])
            last = i
        chunks.append((False, f"### step: {step or '(unknown)'} ({len(lines)}行)\n" + "\n".join(kept)))

    text = "\n\n".join(t for _, t in chunks)
    if len(text) <= max_chars:
        return text
    # 予算超過。全文保持ステップ（LINE本文・表示速度の計測結果）は中央省略で消したくないので、
    # 先に通常ステップを縮めて予算を作る。ステップを切り直せるようになって1 runあたりの
    # ステップ数が1→20前後に増えたため、一律の中央省略だと中盤のステップが丸ごと消える。
    full_len = sum(len(t) for is_full, t in chunks if is_full)
    normals = [i for i, (is_full, _) in enumerate(chunks) if not is_full]
    if normals:
        per = max(600, (max_chars - full_len) // len(normals))
        for i in normals:
            chunks[i] = (False, _cut_middle(chunks[i][1], per))
        text = "\n\n".join(t for _, t in chunks)
    return _cut_middle(text, max_chars)


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
            age = r.get("_age_days")
            note = (f"\n（※24時間の対象期間外。{age}日前の最新実行を参考情報として添付。"
                    f"当日の出来事としては扱わないこと）") if age is not None else ""
            parts.append(
                f"## {wf} — run {r['databaseId']} ({r.get('event')}, {r.get('conclusion')}, "
                f"開始 {r['createdAt']}, 終了 {r.get('updatedAt')}){note}\n{r.get('url', '')}\n\n"
                f"```\n{r['_jobs_text']}\n```\n\n{r['_log_text']}\n"
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 参考run: 対象期間外だが最新結果を見たい週次ワークフロー
# ---------------------------------------------------------------------------
def attach_reference_runs(runs_by_workflow: dict[str, list[dict]], repo: str, now: datetime) -> None:
    """対象期間に実行が無かった週次ワークフローの最新runを、経過日数付きで足す（破壊的に更新）。

    取得できなくても当日のレビューは続けたいので、例外は握って何もしない。
    """
    for wf in sorted(_REFERENCE_WORKFLOWS):
        if runs_by_workflow.get(wf):
            continue
        try:
            found = json.loads(_gh([
                "run", "list", "--repo", repo, "--workflow", wf, "--limit", "1",
                "--status", "completed", "--json",
                "databaseId,conclusion,status,createdAt,updatedAt,displayTitle,url,event",
            ]) or "[]")
            if not found:
                print(f"[ref] {wf}: 実行履歴なし")
                continue
            r = found[0]
            created = datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
            r["_age_days"] = max(0, (now - created).days)
            jobs, log = fetch_run_detail(r["databaseId"], repo)
            r["_jobs_text"] = format_jobs(jobs)
            r["_log_text"] = condense_log(log, _MAX_CHARS_PER_SUCCESS_RUN)
        except Exception as e:
            print(f"[ref] ⚠ {wf} の最新runを取得できず（参考情報なしで続行）: {e}")
            continue
        runs_by_workflow[wf] = [r]
        print(f"[ref] {wf}: 対象期間外の最新run（{r['_age_days']}日前）を参考情報として添付")


# ---------------------------------------------------------------------------
# 前回レビューの提案: 消化状況を判定させるための入力
# ---------------------------------------------------------------------------
def _slice_section(md: str, start_pattern: str, end_pattern: str) -> str:
    """start_pattern の行の次から end_pattern の行の手前までを返す（見つからなければ空）。"""
    m = re.search(start_pattern, md, re.M)
    if not m:
        return ""
    rest = md[m.end():]
    e = re.search(end_pattern, rest, re.M)
    return (rest[: e.start()] if e else rest).strip()


def extract_prev_proposals(prev_md: str, max_chars: int = _PREV_REVIEW_MAX_CHARS) -> str:
    """前回レビュー本文から「今週やる3件」と「改善提案」だけを抜き出す。

    レビュー全文を渡すと入力が倍近くになるうえ、当日ログの読解より前回の記述に引きずられる。
    追跡したいのは提案の消化状況だけなので、その2節に絞る。
    """
    date_m = re.search(r"^#\s*(\d{4}-\d{2}-\d{2})", prev_md, re.M)
    date_label = date_m.group(1) if date_m else "前回"
    todo = _slice_section(prev_md, r"^\*\*今週やる3件\*\*\s*$", r"^\*\*やらない事\*\*")
    proposals = _slice_section(prev_md, r"^## 改善提案\s*$", r"^## ")
    if not todo and not proposals:
        return ""
    parts = [f"## 前回レビューの提案（{date_label}）",
             "当日ログを根拠に、以下の提案それぞれが消化されたかを判定すること。"]
    if todo:
        parts.append(f"### 今週やる3件\n{todo}")
    if proposals:
        parts.append(f"### 改善提案\n{proposals}")
    text = "\n\n".join(parts)
    return text[: max_chars - 1] + "…" if len(text) > max_chars else text


def fetch_previous_review(repo: str | None) -> str:
    """daily-review Issue の最新レビュー（最後のコメント、無ければ本文）から提案を抜き出す。

    取得に失敗しても当日のレビューは続けたいので、例外は握って空文字を返す。
    """
    repo_args = ["--repo", repo] if repo else []
    try:
        issues = json.loads(_gh(
            ["issue", "list", *repo_args, "--state", "open", "--label", _ISSUE_LABEL,
             "--limit", "1", "--json", "number"]
        ) or "[]")
        if not issues:
            print("[prev] 前回レビューなし（Issue未作成）")
            return ""
        data = json.loads(_gh(
            ["issue", "view", str(issues[0]["number"]), *repo_args, "--json", "body,comments"]
        ) or "{}")
    except Exception as e:
        print(f"[prev] ⚠ 前回レビューの取得に失敗（前回比較なしで続行）: {e}")
        return ""
    comments = data.get("comments") or []
    prev_md = (comments[-1].get("body") if comments else data.get("body")) or ""
    text = extract_prev_proposals(prev_md)
    print(f"[prev] 前回レビューの提案 {len(text):,}字を入力に追加" if text
          else "[prev] 前回レビューから提案を抽出できず（前回比較なしで続行）")
    return text


# ---------------------------------------------------------------------------
# 成果物スナップショット（Supabase）: UX / FE / PdM 観点の入力
# ---------------------------------------------------------------------------
_HUMAN_MAX_PV_PER_IP = 100  # tools/traffic_report.py と同じ基準（これを超えるIPは機械とみなす）


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_snapshot_rows(now: datetime) -> dict:
    """Supabaseから素の行を取る（整形は summarize_snapshot）。未設定・失敗時は空dict。"""
    from lib import supabase_client as sb

    if not sb.is_configured():
        return {}
    d1, d8 = now - timedelta(hours=24), now - timedelta(days=8)
    today = now.astimezone(JST).strftime("%Y-%m-%d")
    out = {}
    try:
        out["x_posts_24h"] = sb.select(
            "x_posts", f"posted_at=gte.{_iso(d1)}&select=posted_at,kind,variant,body,impressions,likes,"
                       "reposts,bookmarks,url_link_clicks,has_media&order=posted_at.desc")
        # sb.select は query 内の limit を無視して1000件ページングするため Python 側で切る
        out["x_posts_7d_top"] = sb.select(
            "x_posts", f"posted_at=gte.{_iso(d8)}&select=posted_at,kind,body,impressions,likes,url_link_clicks"
                       "&order=impressions.desc.nullslast")[:5]
        out["x_followers"] = sb.select("x_followers", "select=measured_on,followers&order=measured_on.desc")[:8]
        # proxy.ts の classifyVisitor() は人間らしいアクセスを bot_name='Browser' で記録する（NULLではない）
        out["pv"] = sb.select(
            "blog_crawler_log", f"occurred_at=gte.{_iso(d8)}&bot_name=eq.Browser"
                                "&select=occurred_at,path,visitor_id,ip_address")
        out["rankings"] = sb.select("gen_rankings", f"date=eq.{today}&select=recommend")
        if not out["rankings"]:
            latest = sb.select("gen_rankings", "select=date&order=date.desc&limit=1")
            if latest:
                out["rankings_date"] = latest[0]["date"]
                out["rankings"] = sb.select("gen_rankings", f"date=eq.{latest[0]['date']}&select=recommend")
    except Exception as e:
        print(f"[snapshot] 取得失敗（スナップショット無しで続行）: {e}")
    return out


def fetch_ga4_metrics(days: int = 7) -> dict:
    """GA4のクリックログと回遊指標。認証・プロパティIDが無い環境では空dictを返す。

    サーバーログ(blog_crawler_log)ではクリックも回遊も測れない（30日206,678PVの86.7%が
    1IPで100PV超の機械アクセス）。「どのページで何が押され、そこから次のページへ進んだか」は
    GA4にしか無いので、日次レビューの判断材料として毎日ここに載せる。"""
    from tools import ga4_clicks

    property_id = os.getenv("GA4_PROPERTY_ID", "").strip()
    if not property_id:
        return {}
    try:
        token = ga4_clicks.access_token()
        if not token:
            return {}
        return ga4_clicks.collect_pdca_metrics(token, property_id, days)
    except Exception as e:
        print(f"[ga4] 取得失敗（GA4節なしで続行）: {e}")
        return {}


def _delta(now: float, before: float) -> str:
    if before == 0:
        return "（前週0）" if now else ""
    return f"（前週比 {(now - before) / before * 100:+.0f}%）"


def summarize_ga4(m: dict) -> str:
    """fetch_ga4_metrics の結果をレビュー入力用Markdownにする。"""
    if not m:
        return ("# クリックログ・回遊（GA4）\n"
                "（取得不可: GA4_PROPERTY_ID もしくはサービスアカウント鍵が未設定。"
                "CIでは Secret `GCP_SERVICE_ACCOUNT_JSON` と変数 `GA4_PROPERTY_ID` が要る）")
    now, prev = m["now"], m["prev"]
    lines = [f"# クリックログ・回遊（GA4 {m['start']}〜{m['end']}の{m['days']}日間、"
             f"比較は直前の{m['days']}日間）"]
    lines.append(f"- **エンゲージセッション率 {now.get('engagement_rate', 0) * 100:.1f}%**"
                 f"{_delta(now.get('engagement_rate', 0), prev.get('engagement_rate', 0))}"
                 f"　※回遊の判定はまずこれを見る。率なので訪問者数に引きずられない")
    lines.append(f"- 1セッションあたりの内部移動 {now['internal_per_session']:.2f}回"
                 f"{_delta(now['internal_per_session'], prev['internal_per_session'])}"
                 f"　※(全PV−入口セッション)÷入口セッション。**平均なので1人が何十ページも見た日に跳ねる**"
                 f"（実測: 26人の日=10.20回 / 56人の日=0.19回）。単独で増減を語らず、"
                 f"エンゲージ率と訪問者数を併せて読むこと")
    lines.append(f"- 全PV {now['pv']:.0f}{_delta(now['pv'], prev['pv'])} / "
                 f"入口セッション {now['entrances']:.0f}{_delta(now['entrances'], prev['entrances'])} / "
                 f"訪問者 {now.get('users', 0):.0f}{_delta(now.get('users', 0), prev.get('users', 0))}")

    lines.append("\n## ページ種別ごとの回遊（内部=入口以外から辿り着いたPV）")
    lines.append("| 種別 | PV | 入口 | 内部 | 入口の直帰率 | 1人あたり滞在 | クリック |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, g in sorted(now["groups"].items(), key=lambda kv: -kv[1]["pv"]):
        entrances = g["entrances"]
        bounce = f"{g['bounced'] / entrances * 100:.1f}%" if entrances else "-"
        stay = f"{g['engagement'] / g['users']:.0f}秒" if g["users"] else "-"
        lines.append(f"| {name} | {g['pv']:.0f} | {entrances:.0f} | {g['pv'] - entrances:.0f} | "
                     f"{bounce} | {stay} | {g['clicks']:.0f} |")

    pages = now["pages"]
    clicks = now["clicks"]
    top = sorted(clicks.items(), key=lambda kv: -kv[1][0])[:8]
    if top:
        lines.append("\n## クリックの多いページ（クリック数 / そのページのPV）")
        for path, v in top:
            pv = (pages.get(path) or [0])[0]
            lines.append(f"- {v[0]:.0f}回 / PV{pv:.0f}　{path}")

    labels = m.get("labels") or {}
    named = {k: v for k, v in labels.items() if k and k != "(not set)"}
    lines.append("\n## 押されたCTA（ボタン文言）")
    if named:
        for label, v in sorted(named.items(), key=lambda kv: -kv[1][0])[:10]:
            lines.append(f"- {v[0]:.0f}回　{label[:60]}")
    else:
        lines.append("- まだ集計できていない（labelカスタムディメンションは2026-08-27登録で、"
                     "それ以前のクリックは全て(not set)に入る）")
    return "\n".join(lines)


def summarize_snapshot(rows: dict, now: datetime) -> str:
    """fetch_snapshot_rows の結果をレビュー入力用Markdownにする。"""
    if not rows:
        return "（成果物スナップショット: 取得不可）"
    from collections import Counter

    lines = ["# 成果物スナップショット（Supabase）"]

    # --- X ---
    posts = rows.get("x_posts_24h") or []
    lines.append(f"\n## X投稿（直近24h: {len(posts)}件）※指標は翌1:00 UTC更新のため当日分はほぼ0")
    for p in posts[:12]:
        body = (p.get("body") or "").replace("\n", " / ")[:160]
        lines.append(f"- {p.get('posted_at','')[:16]} [{p.get('kind')}/{p.get('variant')}] imp={p.get('impressions')} "
                     f"like={p.get('likes')} click={p.get('url_link_clicks')} media={p.get('has_media')}: {body}")
    top = rows.get("x_posts_7d_top") or []
    if top:
        lines.append("\n## X 直近7日 インプレッション上位5")
        for p in top:
            lines.append(f"- imp={p.get('impressions')} like={p.get('likes')} click={p.get('url_link_clicks')} "
                         f"[{p.get('kind')}]: {(p.get('body') or '').replace(chr(10), ' / ')[:120]}")
    fol = rows.get("x_followers") or []
    if fol:
        seq = ", ".join(f"{f['measured_on'][5:]}={f['followers']}" for f in reversed(fol))
        lines.append(f"\n## Xフォロワー推移（古→新）: {seq}")

    # --- PV ---
    pv = rows.get("pv") or []
    if pv:
        ip_total = Counter(r.get("ip_address") for r in pv)
        human = [r for r in pv if ip_total[r.get("ip_address")] <= _HUMAN_MAX_PV_PER_IP]
        d1 = _iso(now - timedelta(hours=24))
        last24 = [r for r in human if (r.get("occurred_at") or "") >= d1]
        prev7 = [r for r in human if (r.get("occurred_at") or "") < d1]
        uniq24 = len({r.get("visitor_id") for r in last24 if r.get("visitor_id")})
        lines.append(f"\n## サイトPV（bot除外・1IP>{_HUMAN_MAX_PV_PER_IP}PV/8日を機械として除外した人間推定値）")
        lines.append(f"- 直近24h: {len(last24)} PV / ユニーク {uniq24}（前7日平均 {len(prev7)/7:.0f} PV/日）"
                     f"　除外前の生PV24h: {sum(1 for r in pv if (r.get('occurred_at') or '') >= d1)}")
        def _bucket(path: str) -> str:
            parts = (path or "/").split("?")[0].split("/")
            return "/" + parts[1] + ("/*" if len(parts) > 2 and parts[2] else "") if len(parts) > 1 and parts[1] else "/"
        lines.append("- 24hのセクション別PV: " + ", ".join(
            f"{k}={v}" for k, v in Counter(_bucket(r.get("path")) for r in last24).most_common(8)))
        lines.append("- 24hの上位ページ:")
        for path, n in Counter(r.get("path") or "-" for r in last24).most_common(10):
            lines.append(f"  - {n:4d}  {path[:90]}")

    # --- ランキング ---
    rk = rows.get("rankings") or []
    if rk:
        label = rows.get("rankings_date") or now.astimezone(JST).strftime("%Y-%m-%d")
        cnt = Counter((r.get("recommend") or "").split(" ")[0] or "-" for r in rk)
        lines.append(f"\n## ランキング（{label}、{len(rk)}銘柄）推奨ラベル内訳: " +
                     ", ".join(f"{k}={v}" for k, v in cnt.most_common(8)))
    return "\n".join(lines)


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
        # 日次のログ確認に high は過剰（1回あたり出力$0.4前後）。medium で十分な粒度の
        # 指摘が返る。深掘りしたい日は workflow_dispatch で個別に上げればよい。
        output_config={"effort": "medium"},
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
    ap.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY"), help="owner/name")
    ap.add_argument("--dry-run", action="store_true", help="Issue/LINEに送らず標準出力のみ")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=args.since_hours)
    date_label = now.astimezone(JST).strftime("%Y-%m-%d")

    if not args.repo:
        print("--repo または GITHUB_REPOSITORY が必要です")
        return 1
    runs_by_workflow = list_recent_runs(since, args.repo)
    total = 0
    for wf, runs in runs_by_workflow.items():
        # runs は新しい順。失敗runは全文、成功runは最新1本だけ通常圧縮、
        # 同じワークフローの2本目以降の成功runは重要行だけに落とす（入力の大半が
        # 毎時実行の同一ログの繰り返しになるのを防ぐ）。
        seen_success = False
        for r in runs:
            jobs, log = fetch_run_detail(r["databaseId"], args.repo)
            r["_jobs_text"] = format_jobs(jobs)
            if r.get("conclusion") != "success":
                r["_log_text"] = condense_log(log, _MAX_CHARS_PER_RUN)
            elif not seen_success:
                r["_log_text"] = condense_log(log, _MAX_CHARS_PER_SUCCESS_RUN)
                seen_success = True
            else:
                r["_log_text"] = condense_log(
                    log, _MAX_CHARS_PER_REPEAT_RUN, signal_only=True
                ) or "（重要行なし・静かに成功）"
        total += len(runs)
        print(f"[gh] {wf}: {len(runs)} run(s), 圧縮後 {sum(len(r['_log_text']) for r in runs):,}字")

    if total == 0:
        print("対象runが無いためレビューをスキップします")
        return 0

    attach_reference_runs(runs_by_workflow, args.repo, now)
    snapshot = summarize_snapshot(fetch_snapshot_rows(now), now)
    ga4 = summarize_ga4(fetch_ga4_metrics())
    prev = fetch_previous_review(args.repo)
    review_input = (build_review_input(runs_by_workflow, date_label)
                    + "\n\n" + snapshot + "\n\n" + ga4)
    if prev:
        review_input += "\n\n" + prev
    print(f"[review] 入力 {len(review_input):,} 文字")
    try:
        review_md = review_with_claude(review_input)
    except Exception as e:
        # 見張り役が黙って落ちるのが一番まずい（Anthropic APIの利用上限に当たると
        # このレビューも一緒に止まる。実例: 2026-08-24）。レビューできなかったこと自体を通知する。
        print(f"[review] ⚠ レビュー生成に失敗: {e}")
        if not args.dry_run:
            from lib import notify

            notify.error("日次ログレビュー",
                         f"{date_label} のレビューを生成できませんでした（ログの確認は手動で必要）。",
                         detail=str(e))
        return 1

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
