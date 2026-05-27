#!/usr/bin/env python3
"""PDCA Orchestrator: 1日1サイクルの自律改善ループ
   Fund Manager (haiku) → Analyst (sonnet) → Engineer (python) → commit/push
"""
import sys, os, re, json, subprocess, smtplib, glob
from pathlib import Path
from datetime import date
from email.mime.text import MIMEText
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PDCA_DIR = Path(__file__).resolve().parent
PYTHON   = sys.executable
FEEDBACK = PDCA_DIR / "feedback.md"
LOG      = PDCA_DIR / "pdca_log.md"
BASELINE = PDCA_DIR / "baseline_metrics.json"
TODAY    = date.today().isoformat()

load_dotenv(BASE_DIR / ".env", override=True)

try:
    import anthropic
    client = anthropic.Anthropic()
except ImportError:
    print("anthropic パッケージが必要です: pip install anthropic"); sys.exit(1)


# ── ユーティリティ ─────────────────────────────────────────────────────────

# ── バックテスト期間定義 ──────────────────────────────────────────────────
# データが揃い次第 periods リストに追加していく
BACKTEST_PERIODS = [
    # (ラベル, 開始日, 終了日) — データが存在する期間のみ有効
    ("bear_2024",    "2024-07-01", "2024-10-01"),   # 2024年8月円キャリー崩壊
    ("q2_2025",      "2025-05-14", "2025-08-14"),   # 1年前の3ヶ月
]
# 10年データ取得後に追加予定:
# ("covid_crash",  "2020-02-01", "2020-05-01"),
# ("covid_bull",   "2020-06-01", "2021-12-31"),
# ("rate_hike",    "2022-01-01", "2022-12-31"),
# ("bull_2023",    "2023-01-01", "2024-06-30"),


def _run_one_backtest(start, end):
    """1期間のバックテストを実行してメトリクス dict を返す"""
    r = subprocess.run(
        [PYTHON, "tools/backtest.py", "--start", start, "--end", end],
        capture_output=True, text=True, cwd=str(BASE_DIR), timeout=1800,
        env={**os.environ, "STOCK_ALERT_HOME": str(BASE_DIR)},
    )
    out = r.stdout + r.stderr
    m = {}
    for pat, key in [
        (r'平均リターン: ([+-]?\d+\.?\d*)%',             'avg_return'),
        (r'勝率（\+0%以上）: \d+/\d+ = (\d+\.?\d*)%',    'win_rate'),
        (r'大勝率（\+15%以上）: \d+/\d+ = (\d+\.?\d*)%', 'big_win_rate'),
    ]:
        found = re.search(pat, out)
        if found:
            m[key] = float(found.group(1))
    return m


def run_backtest():
    """全期間を実行して平均メトリクスを返す。1期間でも成功すれば継続。"""
    all_results = {}
    for label, start, end in BACKTEST_PERIODS:
        print(f"  [{label}] {start} → {end} ...")
        m = _run_one_backtest(start, end)
        if m:
            all_results[label] = m
            print(f"    avg={m.get('avg_return')}%  win={m.get('win_rate')}%  big={m.get('big_win_rate')}%")
        else:
            print(f"    [スキップ] データ不足または失敗")

    if not all_results:
        return {}, ""

    # 全期間の平均を複合スコアとして使用
    keys = ['avg_return', 'win_rate', 'big_win_rate']
    composite = {}
    for k in keys:
        vals = [v[k] for v in all_results.values() if k in v]
        composite[k] = round(sum(vals) / len(vals), 2) if vals else None

    # 期間別の詳細も保持
    composite['periods'] = {
        label: m for label, m in all_results.items()
    }
    raw_summary = json.dumps({label: m for label, m in all_results.items()}, ensure_ascii=False)
    return composite, raw_summary


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

NOTIFY_TO   = "dosankoure@gmail.com"

GOAL_AVG    = 3.0   # 改善目標: avg_return > 3%
GOAL_WIN    = 45.0  # 改善目標: win_rate > 45%
GOAL_BIGWIN = 20.0  # 改善目標: big_win_rate > 20%

def fund_manager(metrics, baseline, feedback):
    bsl = baseline or metrics

    # 目標未達なら問答無用で improve（LLM 判断に頼らない）
    avg  = metrics.get('avg_return',  -9999)
    win  = metrics.get('win_rate',        0)
    big  = metrics.get('big_win_rate',    0)
    if avg < GOAL_AVG or win < GOAL_WIN or big < GOAL_BIGWIN:
        return {
            "action": "improve",
            "reason": (
                f"目標未達 (avg={avg}%<{GOAL_AVG}% / win={win}%<{GOAL_WIN}% / "
                f"big={big}%<{GOAL_BIGWIN}%) → 強制 improve"
            )
        }

    prompt = f"""あなたは株式予測モデルのファンドマネージャーです。

【今日のbacktest(bear mode: 2024年8月下落相場)】
avg_return={avg}%  win_rate={win}%  big_win_rate={big}%

【ベースライン（これまでの最良値）】
avg_return={bsl.get('avg_return','N/A')}%  win_rate={bsl.get('win_rate','N/A')}%  big_win_rate={bsl.get('big_win_rate','N/A')}%

【人間フィードバック】
{feedback}

改善目標(avg>{GOAL_AVG}% / win>{GOAL_WIN}% / big>{GOAL_BIGWIN}%)はすべて達成済み。
今日さらにパラメータを調整すべきか、現状維持か判断してください。
JSON1行のみで回答: {{"action":"improve"|"skip","reason":"..."}}"""
    result = parse_json(call_claude("claude-haiku-4-5-20251001", prompt, 200))
    return result or {"action": "improve", "reason": "parse error → fallback improve"}


def load_today_signals():
    """最新ランキング CSV から S買いシグナル銘柄を返す"""
    try:
        import pandas as pd
        csvs = sorted(glob.glob(str(BASE_DIR / "data/rankings/ranking_*.csv")))
        if not csvs:
            return []
        df = pd.read_csv(csvs[-1], encoding='utf-8-sig')
        sbuy = df[df['推奨'] == '🥇 S買い'].copy()
        if sbuy.empty:
            return []
        cols = ['順位', '銘柄コード', '銘柄名', '直近株価(円)',
                'ネット(%)', '上昇確率(%)', '下落確率(%)', 'ボラ(%)', '損切り幅(%)']
        cols = [c for c in cols if c in sbuy.columns]
        return sbuy[cols].to_dict(orient='records')
    except Exception as e:
        print(f"  [signal load error] {e}")
        return []


def fund_manager_stock_advice(signals, reports):
    """アナリストレポートを読んでFund Managerが最終的な購入推奨を決定する"""
    rows = "\n".join(
        f"  {s.get('銘柄コード')} {s.get('銘柄名')}  "
        f"net={s.get('ネット(%)')}%  rise={s.get('上昇確率(%)')}%  "
        f"drop={s.get('下落確率(%)')}%  vol={s.get('ボラ(%)')}%  "
        f"損切={s.get('損切り幅(%)')}%"
        for s in signals
    )
    reports_text = "\n\n".join(
        f"【{code} アナリストレポート】\n{report}"
        for code, report in reports.items()
    )
    prompt = f"""あなたはファンドマネージャーです。
オーナーの目標: 株式運用で3年以内に1億円を作り、家を買う。

アナリストが各銘柄を徹底調査し、以下のレポートを提出しました。
モデルのシグナルとアナリストの見解を総合して、実際に買うべき銘柄を決定してください。

【モデル S買いシグナル一覧】
{rows}

【アナリストレポート】
{reports_text}

判断基準:
1. アナリストが「買い推奨」で、モデルのネットスコアも高い銘柄を優先
2. リスク（ボラ・損切り幅）が許容範囲内か確認
3. アナリストが「売り推奨」や重大なリスクを指摘した銘柄は除外

上位1〜3銘柄を選び、「なぜ買うべきか（モデル+ファンダメンタルズの両面から）」を
それぞれ2〜3文で説明してください。
最後に「今日の一押し」を1つ選んで明示してください。
日本語テキストで回答（JSON不要）。"""
    try:
        return call_claude("claude-haiku-4-5-20251001", prompt, 800)
    except Exception as e:
        return f"評価エラー: {e}"


def send_buy_notification(signals, advice):
    """購入推奨をGmailで通知"""
    gmail_addr = os.getenv("GMAIL_ADDRESS")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail_addr or not gmail_pass:
        print("  [通知スキップ] GMAIL_ADDRESS / GMAIL_APP_PASSWORD 未設定")
        return

    codes = ", ".join(f"{s.get('銘柄コード')} {s.get('銘柄名')}" for s in signals)
    body = f"""📈 本日の S買いシグナル銘柄が出ました

{codes}

━━━━━━━━━━━━━━━━
Fund Manager のコメント
━━━━━━━━━━━━━━━━
{advice}

━━━━━━━━━━━━━━━━
※ 投資判断は自己責任でお願いします。損切り価格を必ず確認してから購入してください。
"""
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"📈 S買いシグナル {len(signals)}銘柄 — {TODAY}"
    msg["From"]    = gmail_addr
    msg["To"]      = NOTIFY_TO

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_addr, gmail_pass)
            server.sendmail(gmail_addr, NOTIFY_TO, msg.as_string())
        print(f"  → 購入通知メールを送信: {NOTIFY_TO}")
    except Exception as e:
        print(f"  [メール送信エラー] {e}")


# ══════════════════════════════════════════════════════════════════════
#  証券アナリスト（Analyst）
#  企業の財務状況・経営戦略・業界動向を調査し、FM へ推奨レポートを提出
# ══════════════════════════════════════════════════════════════════════

def analyst_research_stock(code, name, net_score, vol):
    """1銘柄を徹底調査して買い/売り推奨レポートを返す（web検索×3回）"""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            messages=[{"role": "user", "content": f"""あなたは証券アナリストです。
企業の財務状況、経営戦略、業界動向などを徹底的に調査・分析する専門家として、
以下の銘柄をIR取材レベルで調査し、ファンドマネージャーへの推奨レポートを作成してください。

【調査対象】{code} {name}
【モデルスコア】ネットスコア={net_score}%  ボラティリティ={vol}%

以下の観点で web 検索して調査してください:
1. 財務状況 — 直近の決算、売上高・利益の成長率、PER/PBR/ROE、キャッシュフロー
2. 経営戦略・IR — 経営者コメント、中期経営計画、直近のプレスリリース・IR資料
3. 業界動向 — セクターのトレンド、競合他社との比較、逆風・追い風要因
4. リスク要因 — 決算ミス、地政学リスク、法規制変更など

【レポート形式】
- 各観点を2〜3文で要約
- 総合評価: 「強く買い推奨」「買い推奨」「中立」「売り推奨」のいずれかを明記
- 推奨理由を1〜2文で結論として書く
"""}],
        )
        texts = [b.text for b in response.content if hasattr(b, 'text') and b.text]
        return '\n'.join(texts) if texts else "（調査結果なし）"
    except Exception as e:
        return f"（調査エラー: {type(e).__name__}）"


def analyst_reports(signals):
    """S買いシグナル銘柄をアナリストが全件調査してレポート dict を返す"""
    reports = {}
    for s in signals:
        code = s.get('銘柄コード', '')
        name = s.get('銘柄名', '')
        net  = s.get('ネット(%)', '')
        vol  = s.get('ボラ(%)', '')
        print(f"  → アナリスト調査中: {code} {name} ...")
        reports[code] = analyst_research_stock(code, name, net, vol)
        print(f"    完了")
    return reports


# ══════════════════════════════════════════════════════════════════════
#  数量アナリスト（Quant Analyst）
#  モデルパラメータの改善提案 — config.py を毎日チューニング
# ══════════════════════════════════════════════════════════════════════

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


def quant_analyst(metrics, baseline):
    """数量アナリスト（上位版）: config.py・rf_train_v3.py の両方を改善対象にできる"""
    bsl    = baseline or metrics
    config = (BASE_DIR / "config.py").read_text('utf-8')
    train  = (BASE_DIR / "core" / "rf_train_v3.py").read_text('utf-8')
    fi_raw = (BASE_DIR / "feature_importance.json").read_text('utf-8')
    fi     = json.loads(fi_raw)
    # 重要度上位10特徴量を整形
    rise_top = sorted(fi.get('rise', {}).items(), key=lambda x: -x[1])[:10]
    drop_top = sorted(fi.get('drop', {}).items(), key=lambda x: -x[1])[:10]
    fi_str = "上昇モデル: " + ", ".join(f"{k}={v:.3f}" for k, v in rise_top)
    fi_str += "\n下落モデル: " + ", ".join(f"{k}={v:.3f}" for k, v in drop_top)

    # AUC スコアを読む
    auc_str = "不明"
    try:
        auc = json.loads((BASE_DIR / "baseline_auc.json").read_text('utf-8'))
        auc_str = f"上昇AUC={auc.get('rise',0):.4f}  下落AUC={auc.get('drop',0):.4f}"
    except Exception:
        pass

    print("  → 最新論文を検索中...")
    research = _research_papers()
    print(f"  → 検索完了: {research[:80].strip()}...")

    prompt = f"""あなたは世界トップレベルのクオンツ研究者です。
日本株XGBoostモデルを「3年で1億円を稼ぐ」レベルまで引き上げてください。

【現在のモデル性能】
バックテスト複合スコア: avg={metrics.get('avg_return')}%  win={metrics.get('win_rate')}%  big={metrics.get('big_win_rate')}%
ベースライン:          avg={bsl.get('avg_return')}%  win={bsl.get('win_rate')}%  big={bsl.get('big_win_rate')}%
モデルAUC: {auc_str}

【特徴量重要度】
{fi_str}

【最新研究からの知見】
{research}

【config.py（スクリーニング・フィルター定数）】
{config}

【rf_train_v3.py（モデル学習コード）抜粋】
{train[train.find('def train_model'):train.find('def main')]}

改善の方向性（自由に判断せよ）:
- XGBoostのハイパーパラメータ調整（n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight など）
- スクリーニング条件の見直し（config.py）
- ALPHA_THRESHOLD / DROP_ALPHA_THRESHOLD の調整（rf_train_v3.py の定数）
- 特徴量重要度が低い特徴量のweight調整アイデア

変更禁止:
- BEAR_MARKET_THRESHOLD, HOT_MARKET_THRESHOLD（市場分類の根幹）
- コメント行

変更できるファイル: "config.py" または "rf_train_v3.py"
※ rf_train_v3.py を変更した場合、モデル再学習（30〜60分）が必要になります。
  それでも効果が大きいと判断した場合は迷わず提案してください。

JSON のみで回答（fileフィールドで対象ファイルを指定）:
{{"file":"config.py","changes":[{{"param_name":"...","old_value":...,"new_value":...,"reason":"..."}}]}}
または
{{"file":"rf_train_v3.py","changes":[{{"param_name":"...","old_value":...,"new_value":...,"reason":"..."}}]}}"""

    raw    = call_claude("claude-sonnet-4-6", prompt, 800)
    parsed = parse_json(raw)
    if parsed and "changes" in parsed:
        return parsed
    if parsed and "param_name" in parsed:
        return {"file": "config.py", "changes": [parsed]}
    return None


def _apply_changes(file_path, changes):
    """ファイルにパラメータ変更を一括適用。(new_text, applied_list) を返す"""
    orig     = file_path.read_text('utf-8')
    new_text = orig
    applied  = []
    for ch in changes:
        p  = ch['param_name']
        nv = ch['new_value']
        pat      = rf'^({re.escape(p)}\s*=\s*)[\d.+\-]+'
        replaced = re.sub(pat, rf'\g<1>{nv}', new_text, flags=re.MULTILINE)
        if replaced != new_text:
            applied.append(ch)
            new_text = replaced
        else:
            print(f"  [WARN] {p} が見つからずスキップ")
    return orig, new_text, applied


def _retrain_model():
    """rf_train_v3.py を実行してモデルを再学習（30〜60分）"""
    print("  → モデル再学習中（30〜60分かかります）...")
    r = subprocess.run(
        [PYTHON, "-u", "core/rf_train_v3.py"],
        capture_output=True, text=True, cwd=str(BASE_DIR), timeout=7200,
        env={**os.environ, "STOCK_ALERT_HOME": str(BASE_DIR)},
    )
    out = r.stdout + r.stderr
    success = "保存完了" in out or "✅" in out
    print(f"  → 再学習{'成功' if success else '失敗'}")
    return success, out


def engineer(proposal, before_metrics, baseline):
    """
    proposal = {
      "file": "config.py" | "rf_train_v3.py",
      "changes": [{param_name, old_value, new_value, reason}, ...]
    }
    rf_train_v3.py の場合は変更後にモデル再学習を実行する。
    """
    target_name = proposal.get('file', 'config.py')
    target_path = BASE_DIR / target_name if target_name == 'config.py' \
                  else BASE_DIR / 'core' / 'rf_train_v3.py'
    changes = proposal.get('changes', [])

    if not changes:
        return False, "変更リストが空"

    orig, new_text, applied = _apply_changes(target_path, changes)

    if not applied:
        return False, f"{target_name}: すべてのパラメータが見つからず変更失敗"

    target_path.write_text(new_text, 'utf-8')

    # rf_train_v3.py 変更時はモデル再学習が必要
    if target_name == 'rf_train_v3.py':
        ok, train_out = _retrain_model()
        if not ok:
            target_path.write_text(orig, 'utf-8')
            return False, f"rf_train_v3.py 変更後の再学習失敗、revert"

    after, _ = run_backtest()

    if not after:
        target_path.write_text(orig, 'utf-8')
        return False, "backtest失敗、revert"

    bsl = baseline if baseline else before_metrics
    improved = any([
        after.get('avg_return',    -9999) > bsl.get('avg_return',    -9999),
        after.get('win_rate',          0) > bsl.get('win_rate',          0),
        after.get('big_win_rate',      0) > bsl.get('big_win_rate',      0),
    ])

    change_summary = " / ".join(
        f"{c['param_name']} {c.get('old_value','?')}→{c['new_value']}" for c in applied
    )
    commit_files = [target_name if target_name == 'config.py' else f'core/{target_name}',
                    'pdca/baseline_metrics.json']
    if target_name == 'rf_train_v3.py':
        commit_files += ['rf_model.pkl', 'rf_drop_model.pkl']

    if improved:
        save_baseline(after)
        msg = (
            f"pdca: [{target_name}] {change_summary} | "
            f"avg {bsl.get('avg_return','?')}→{after.get('avg_return','?')}% "
            f"win {bsl.get('win_rate','?')}→{after.get('win_rate','?')}% [skip ci]"
        )
        git_commit_push([f for f in commit_files if (BASE_DIR / f).exists() or '/' in f], msg)
        return True, (
            f"✅ 採用 [{target_name}]: {change_summary} | "
            f"avg {bsl.get('avg_return')}%→{after.get('avg_return')}%  "
            f"win {bsl.get('win_rate')}%→{after.get('win_rate')}%  "
            f"big {bsl.get('big_win_rate')}%→{after.get('big_win_rate')}%"
        )
    else:
        target_path.write_text(orig, 'utf-8')
        return False, (
            f"❌ 改善なし [{target_name}]: {change_summary} | "
            f"avg {after.get('avg_return')}% (bsl {bsl.get('avg_return')}%)、revert"
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
    periods_info = metrics.pop('periods', {})
    print(f"  複合スコア: avg={metrics.get('avg_return')}%  win={metrics.get('win_rate')}%  big={metrics.get('big_win_rate')}%")
    log(f"- metrics: {json.dumps(metrics)}  periods: {json.dumps(periods_info, ensure_ascii=False)}")

    baseline = load_baseline()
    if not baseline:
        save_baseline(metrics)
        baseline = metrics
        log("- 初回実行: baseline設定")
        print("  初回: baselineを現在値で設定")

    # Step1.5: S買いシグナル銘柄チェック → アナリスト調査 → FM判断 → 購入通知
    print("Step1.5: S買いシグナル 確認中...")
    signals = load_today_signals()
    if signals:
        codes = [s.get('銘柄コード') for s in signals]
        print(f"  → {len(signals)}銘柄 発見: {codes}")
        print("Step1.5a: アナリスト 各銘柄を調査中...")
        reports = analyst_reports(signals)
        print("Step1.5b: Fund Manager レポートを読んで推奨銘柄を決定中...")
        advice = fund_manager_stock_advice(signals, reports)
        print(f"  → FM推奨:\n{advice}")
        send_buy_notification(signals, advice)
        log(f"- signals: {len(signals)}銘柄 通知済み {codes}")
    else:
        print("  → 本日S買いシグナルなし")
        log("- signals: なし")

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

    # Step3: 数量アナリスト（Quant Analyst） — モデルパラメータ改善
    print("Step3: Quant Analyst 分析中...")
    prop = quant_analyst(metrics, baseline)
    if not prop:
        log("- analyst: parse error"); push_log(); return
    for ch in prop.get('changes', []):
        print(f"  → {ch['param_name']}: {ch.get('old_value')}→{ch['new_value']}  ({ch.get('reason','')})")
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
