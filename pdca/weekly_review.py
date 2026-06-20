#!/usr/bin/env python3
"""週次ピアレビュー: 5ロール相互評価
   ロール:
     FM（ファンドマネージャー）、Quant（数量アナリスト）、Consultant（マーケットコンサル）、
     Engineer（エンジニア）、Human（オーナー・評価を受けるのみ）
   毎週月曜 10:00 JST に GitHub Actions から自動実行
"""
import sys, os, re, json, subprocess, requests
from pathlib import Path
from datetime import date, timedelta

import activity  # アクティビティログ

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

def call_claude(prompt: str, max_tokens: int = 1200) -> str:
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


def load_model_state() -> dict:
    """現在のモデル設定・特徴量重要度・閾値を読み込む"""
    state = {}
    try:
        import sys as _sys
        _sys.path.insert(0, str(BASE_DIR))
        # config.py から設定を読む
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", BASE_DIR / "config.py")
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        state["forecast_days"]     = getattr(cfg, "FORECAST", "不明")
        state["rise_threshold"]    = getattr(cfg, "RISE_THRESHOLD", "不明")
        state["bear_threshold"]    = getattr(cfg, "BEAR_MARKET_THRESHOLD", "不明")
    except Exception:
        state["forecast_days"] = "不明"

    # 特徴量重要度トップ5
    try:
        fi_path = BASE_DIR / "feature_importance.json"
        if fi_path.exists():
            fi = json.loads(fi_path.read_text())
            rise_imp = fi.get("rise", {})
            top5 = sorted(rise_imp.items(), key=lambda x: -x[1])[:5]
            state["top5_features"] = ", ".join(f"{k}({v:.3f})" for k, v in top5)
        else:
            state["top5_features"] = "不明"
    except Exception:
        state["top5_features"] = "不明"

    # 最適閾値
    try:
        thr_path = BASE_DIR / "optimal_thresholds.json"
        if thr_path.exists():
            thr = json.loads(thr_path.read_text())
            state["rise_threshold_opt"]  = thr.get("rise", "不明")
            state["drop_threshold_opt"]  = thr.get("drop", "不明")
    except Exception:
        pass

    return state


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


# ── 共通コンテキスト ────────────────────────────────────────────────────────────

def build_ctx(model_state: dict, trajectory: list, actions: dict) -> str:
    traj_str = " → ".join(
        f"avg{m.get('avg_return')}% 勝率{m.get('win_rate')}% α{m.get('nk_alpha','?')}%"
        for m in trajectory[-5:]
    ) if trajectory else "（データなし）"

    return f"""【システム概要】
目標: 日本株モデルで元本300万円を10年で1億円（年率+42%）。
戦略: TSE全銘柄をXGBoostでスコアリングし、上位N銘柄を{model_state.get('forecast_days','?')}日保有。
モデル: 49次元特徴量（テクニカル32 + ファンダメンタル11 + クロスセクション6）

【現在のモデル設定】
- 予測ホライズン: {model_state.get('forecast_days','?')}日
- 上昇閾値: +{model_state.get('rise_threshold','?')}%
- 弱気相場判定: 日経20日 < {model_state.get('bear_threshold','?')}%
- 重要度TOP5特徴量: {model_state.get('top5_features','不明')}

【今週の成績推移（最新5件）】
{traj_str}

【今週のアクション】採用:{actions['adopted']}件 / 却下:{actions['rejected']}件 / 買いシグナル:{actions['signals']}回
"""


# ── 評価プロンプト（深度強化版）────────────────────────────────────────────────

def fm_structural_audit(log_text: str, model_state: dict, trajectory: list, actions: dict) -> str:
    """FM: 設計整合性の監査（毎週必須チェック）"""
    ctx = build_ctx(model_state, trajectory, actions)
    last = trajectory[-1] if trajectory else {}
    nk_alpha = last.get("nk_alpha", "不明")

    prompt = f"""{ctx}

あなたはファンドマネージャー（FM）です。
数字の善し悪しを報告するだけでなく、**モデル設計の構造的問題を発見する**のが今週の仕事です。

【今週の記録】
{log_text[:2000]}

## 必須チェック項目（全項目に必ず答えること）

### 1. 予測期間と保有期間の整合性
- モデルの予測ホライズン（{model_state.get('forecast_days','?')}日）と実際の保有日数は一致しているか？
- ずれがある場合、それはパフォーマンスにどう影響しているか？

### 2. 日経平均比アルファの診断
- 直近の日経アルファ: {nk_alpha}%
- 日経に負けている場合、その構造的原因として何が考えられるか？（「運が悪かった」は不可）
- 相場レジーム（強気/弱気）と戦略の相性は適切か？

### 3. 特徴量の有効性チェック
- 重要度TOP5: {model_state.get('top5_features','不明')}
- 上位特徴量はモデルが学習すべき信号を捉えているか、それともノイズか？
- 重要度が0に近い特徴量が残っていないか？（削除候補）

### 4. スクリーナー/フィルターのバイアス診断
- 現在の銘柄選択プロセスは「まだ上がっていない株」を選べているか？
- それとも「すでに上昇済みの乗り遅れ株」を選んでいないか？

### 5. 来週の最重要アクション（1つだけ）
上記チェックで発見した最も深刻な問題に対する具体的な改善策を1つ提案する。
「パラメータを少し変える」ではなく、設計レベルの変更を検討すること。

回答は各項目200字以内で、定量的な根拠を必ず含めること。"""
    return call_claude(prompt, 1500)


def quant_model_integrity(log_text: str, model_state: dict, trajectory: list, actions: dict) -> str:
    """Quant: モデル整合性・特徴量品質の深掘り分析"""
    ctx = build_ctx(model_state, trajectory, actions)

    prompt = f"""{ctx}

あなたは数量アナリスト（Quant）です。
今週の数字変化だけでなく、**モデルが正しい問題を学習しているかを批判的に検証**してください。

【今週の記録】
{log_text[:2000]}

## 分析項目

### 1. ラベル設計の妥当性
- 現在の上昇ラベル: {model_state.get('forecast_days','?')}日後に+{model_state.get('rise_threshold','?')}%以上
- このラベルは実際の運用目標（年率+42%）と整合しているか？
- 正例率（上昇ラベルが全体の何%か）は適切な範囲か？過少/過多では？

### 2. 特徴量リークの確認
- クロスセクション特徴量（同日全銘柄の相対順位）はバックテスト時に未来情報を使っていないか？
- PIT（Point-in-Time）ファンダメンタルは正しく機能しているか？

### 3. 過学習の兆候
- 学習データのAUCとテストデータのAUCの差は何か？（乖離が大きければ過学習）
- early_stopping_roundsは適切か？

### 4. 業績モメンタム特徴量（eps_growth, roe_trend）の状態
- 直近追加した特徴量の重要度は期待通りか？
- データカバレッジは十分か（全銘柄の何%に値があるか）？

### 5. 次に試すべき特徴量または設計変更（優先度付き）
Chan et al.(1996)の業績サプライズ、Gu et al.(2020)の機械学習資産価格論、
日本市場固有のアノマリー（TSE改革PBR、配当利回り）などを踏まえて提案すること。

各項目150字以内、数値根拠必須。"""
    return call_claude(prompt, 1500)


def consultant_macro_regime(log_text: str, model_state: dict, trajectory: list, actions: dict) -> str:
    """Consultant: マクロ・レジーム分析とモデル戦略の整合性チェック"""
    ctx = build_ctx(model_state, trajectory, actions)

    prompt = f"""{ctx}

あなたはマーケットコンサルタントです。
**現在の相場レジームとモデル戦略の整合性**を専門家として評価してください。

【今週の記録】
{log_text[:1500]}

## 評価項目

### 1. 相場レジーム診断
- 現在の市場環境（強気/弱気/過熱/リスクオフ）を具体的な根拠と共に判定
- 日経225のSMA63/SMA200との位置関係、実現ボラティリティから判断すること

### 2. モデル戦略とレジームの整合性
- 現在のモデル（モメンタム+ファンダ複合）はこのレジームで機能する設計か？
- Daniel & Moskowitz(2016)の「モメンタムクラッシュ」リスクは現在高いか低いか？

### 3. 日本市場の構造変化への対応
- TSE改革（PBR改善要求）の影響は現在の特徴量に反映されているか？
- 為替（円高/円安）・金利変化が現在のポートフォリオに与えるリスク

### 4. 相場環境に応じた推奨銘柄数の妥当性
- 現在のレジーム動的N（強気10/中立5/弱気3）は適切か？
- 今週の相場環境に対してどの設定が最適か？

### 5. 来週のリスク要因
具体的なイベント・指標発表・季節性から来週注意すべきリスクを2つ挙げること。

各項目150字以内、具体的なデータや文献根拠を含めること。"""
    return call_claude(prompt, 1500)


def engineer_evaluates(log_text: str, actions: dict, model_state: dict) -> str:
    """Engineer: 実装品質・テスト厳密性の評価"""
    prompt = f"""【システム概要】
日本株予測モデル（49次元XGBoost）の自動改善システム。

【今週の記録】
{log_text[:1500]}

【集計】採用:{actions['adopted']}件 / 却下:{actions['rejected']}件 / エラー:{actions['parse_error']}件
【モデル設定】予測期間:{model_state.get('forecast_days','?')}日 / 重要特徴量:{model_state.get('top5_features','?')}

あなたはEngineerです。**コードの品質と検証の厳密性**を評価してください。

## 評価項目

### 1. テストの厳密性
- bearバックテスト（2024年8月暴落期）でちゃんと検証されているか？
- ローリングバックテストの結果は日経比で正確に計算されているか？
- 「採用」判断の根拠は十分に定量的か？（印象でなくデータに基づいているか）

### 2. 実装のバグリスク
- 今週変更したコードでルックアヘッドバイアス（未来情報の混入）の懸念はあるか？
- 特徴量の次元数変更（47→49）後、全パイプラインで整合が取れているか？

### 3. QuantとFMへの評価
- Quantの提案は実装可能な精度で書かれていたか？（曖昧な指示ではないか）
- FMの採用/却下判断は一貫した基準に基づいていたか？

### 4. 来週の技術的リスク
今週の変更を踏まえ、来週のCI/CDで起きうる問題を具体的に2つ挙げること。

各項目150字以内、具体例必須。"""
    return call_claude(prompt, 1200)


def evaluate_human(log_text: str, feedback_text: str, trajectory: list, model_state: dict) -> str:
    traj_str = " → ".join(
        f"avg{m.get('avg_return')}% α{m.get('nk_alpha','?')}%"
        for m in trajectory[-5:]
    ) if trajectory else "（データなし）"

    last = trajectory[-1] if trajectory else {}
    nk_alpha = last.get("nk_alpha", "不明")

    prompt = f"""【システム概要】
日本株予測モデル（XGBoost 49次元）の自動PDCA改善システム。
目標: 元本300万 → 10年で1億円（年率+42%）。

【オーナーの方針（feedback.md）】
{feedback_text[:1000]}

【今週の記録】
{log_text[:1000]}

【パフォーマンス推移】
{traj_str}
直近の日経アルファ: {nk_alpha}%

あなたはAIチーム全員（FM・Quant・Consultant・Engineer）の代表として、
オーナーに率直なフィードバックをします。

## AIチームからオーナーへ

### 成果として報告できること（1〜2つ）
数値的根拠付きで、今週実際に改善できたことを報告する。
「作業した」ではなく「数値がこう変わった」で示すこと。

### オーナーが決断すべき設計上の問題（1〜2つ）
AIチームでは判断できない、オーナーの意思決定が必要な設計上の問題を明示する。
例:「63日保有に戻すか/21日のまま続けるか」「スクリーナー廃止の是非」など
データと根拠を示した上で選択肢を提示し、オーナーに決断を求める。

### 来週AIチームに期待するアクション（2つ）
具体的なタスク。「改善する」ではなく「○○をバックテストで検証し、αが+X%改善すれば採用」まで書く。

率直かつ簡潔に（各項目200字以内）。問題は隠さないこと。"""
    return call_claude(prompt, 1000)


def synthesize_actions(evals: dict, trajectory: list, actions: dict, model_state: dict) -> str:
    first = trajectory[0]  if trajectory else {}
    last  = trajectory[-1] if trajectory else {}
    delta_avg = round(last.get("avg_return", 0) - first.get("avg_return", 0), 2) if trajectory else 0
    nk_alpha  = last.get("nk_alpha", "不明")

    summaries = "\n\n".join(
        f"【{role}の診断】\n{text[:500]}"
        for role, text in evals.items()
    )

    prompt = f"""【今週の状況】
予測期間: {model_state.get('forecast_days','?')}日 / 日経アルファ: {nk_alpha}% / 採用:{actions['adopted']}件
平均リターン変化: {'+' if delta_avg>=0 else ''}{delta_avg}%

【各ロールの診断（抜粋）】
{summaries}

上記の診断を統合して「来週やること」を決定してください。

## 判断基準
- 「パラメータを少し変える」レベルの改善は不採用
- 日経アルファ改善に直結するか、設計の構造的問題を解決するものを優先
- 検証方法（bearバックテストで何%改善すれば採用）まで明記すること

## 来週やること（優先度順に4〜6個）
以下の形式で:
- 担当: 具体的なアクション（採用条件: bearバックテストでαが+N%以上）

例:
- Quant: eps_growthの正規化範囲を[-1,3]→[-0.5,2]に変更（採用条件: bearバックテストでαが+1%以上改善）
- FM: 相場がbullレジームの週は保有銘柄数を10→15に一時拡大（採用条件: rolling 90日でαが+2%以上）
- Engineer: 特徴量重要度0.001以下の特徴量を削除してモデルを軽量化（採用条件: AUCが±0.005以内）

箇条書きのみ。曖昧な表現は禁止。"""
    return call_claude(prompt, 600)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print(f"=== 週次ピアレビュー {ISO_WEEK} ({TODAY}) ===")

    log_text      = extract_last_week_log()
    feedback_text = load_feedback_text()
    trajectory    = parse_metrics_trajectory(log_text)
    actions       = count_actions(log_text)
    model_state   = load_model_state()

    print(f"  ログ:{len(log_text.splitlines())}行  指標:{len(trajectory)}件  "
          f"採用:{actions['adopted']} 却下:{actions['rejected']} シグナル:{actions['signals']}回")
    print(f"  モデル設定: FORECAST={model_state.get('forecast_days')}日 "
          f"/ TOP5特徴量: {model_state.get('top5_features','不明')[:60]}")

    review_aid = activity.start("System", "週次チームレビュー",
                                "FM・Quant・マーケットコンサル・Engineer が構造監査中…")

    first = trajectory[0]  if trajectory else {}
    last  = trajectory[-1] if trajectory else {}

    print("Step1: FM → 設計整合性監査（構造的問題の発見）...")
    fm_eval = fm_structural_audit(log_text, model_state, trajectory, actions)

    print("Step2: Quant → モデル整合性・特徴量品質分析...")
    qa_eval = quant_model_integrity(log_text, model_state, trajectory, actions)

    print("Step3: Consultant → マクロ・レジーム分析...")
    sa_eval = consultant_macro_regime(log_text, model_state, trajectory, actions)

    print("Step4: Engineer → 実装品質・テスト厳密性評価...")
    eng_eval = engineer_evaluates(log_text, actions, model_state)

    print("Step5: AIチーム → Human フィードバック...")
    human_fb = evaluate_human(log_text, feedback_text, trajectory, model_state)

    print("Step6: 来週アクション統合...")
    evals = {
        "FM（設計監査）":          fm_eval,
        "Quant（モデル整合性）":   qa_eval,
        "Consultant（レジーム）":  sa_eval,
        "Engineer（実装品質）":    eng_eval,
        "AIチーム → Human":       human_fb,
    }
    next_actions = synthesize_actions(evals, trajectory, actions, model_state)

    # 数値サマリー
    def d(v1, v2): return round(v2 - v1, 2) if v1 is not None and v2 is not None else 0
    def s(v): return f"+{v}" if v >= 0 else str(v)

    delta_avg = d(first.get("avg_return"), last.get("avg_return"))
    delta_win = d(first.get("win_rate"),   last.get("win_rate"))
    delta_big = d(first.get("big_win_rate"), last.get("big_win_rate"))
    nk_alpha  = last.get("nk_alpha", "N/A")

    # Markdownレポート
    report = f"""# 週次ピアレビュー {ISO_WEEK}（{TODAY}）

## 今週の数字
| 指標 | 週初 | 週末 | 変化 |
|------|------|------|------|
| 平均リターン | {first.get('avg_return','?')}% | {last.get('avg_return','?')}% | {s(delta_avg)}% |
| 勝率         | {first.get('win_rate','?')}%   | {last.get('win_rate','?')}%   | {s(delta_win)}% |
| 大勝率       | {first.get('big_win_rate','?')}% | {last.get('big_win_rate','?')}% | {s(delta_big)}% |
| 日経アルファ | — | {nk_alpha}% | — |

採用 {actions['adopted']} / 却下 {actions['rejected']} / スキップ {actions['skipped']} / 買いシグナル {actions['signals']} 回

**モデル設定**: FORECAST={model_state.get('forecast_days','?')}日 / 重要度TOP5: {model_state.get('top5_features','不明')}

---

## FM → 設計整合性監査（構造的問題の発見）

{fm_eval}

---

## Quant → モデル整合性・特徴量品質分析

{qa_eval}

---

## Consultant → マクロ・レジーム分析

{sa_eval}

---

## Engineer → 実装品質・テスト厳密性評価

{eng_eval}

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
    activity.finish(review_aid, "done",
                    f"{ISO_WEEK} のレビュー完了（来週やること {len([l for l in next_actions.splitlines() if l.strip().startswith('-')])}件）",
                    next_actions)
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
