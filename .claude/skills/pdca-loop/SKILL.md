---
name: pdca-loop
description: モデル・フィルターの改善案を Plan→Do→Check→Act の1サイクルで自律的に回す。改善提案・PDCA・自動改善・戦略改善ループを求められたとき、または定期実行のRoutineから呼ばれたときに使う。
---

# PDCAループ（stock-alert 改善サイクル）

CLAUDE.md の「Dev Cycle」「改善マージ規律」「マージ規律（§8）」をそのまま1サイクルとして自律実行するスキル。
**1回の起動につき仮説は1つだけ**。大規模リファクタや特徴量定義の変更は行わない（CLAUDE.md §0 Think Small / No Hallucination）。

## 事前チェック（毎回）

1. `git status` で作業ツリーがクリーンか確認。汚れていれば中断しユーザーに報告。
2. `git fetch origin main && git checkout -B pdca/<短い施策名> origin/main` で**必ず最新mainから**ブランチを切る（stale base防止、CLAUDE.md §8）。
3. `dev_log.md` の直近10〜20件を読み、**既に「不採用」と記録された仮説は繰り返さない**。

## Plan（仮説を1つ立てる）

- 対象は「パラメータ・閾値の調整」「フィルター条件の微調整」など小さいものに限定する。
  - 例: `config.py` の閾値、`passes_buy_filter`/`recommend_from_scores` の境界値、S買い条件の微調整
- 触ってはいけないもの（CLAUDE.md厳守）:
  - `lib/utils.py` の64次元特徴量の定義
  - ハードフィルター: `down_streak > 3日`, `drawdown60 < -15%`
  - βフィルターのロジック自体（日経強気時 β≥0.4）
- `feature_importance.json` や直近の `dev_log.md` の負け筋（大負けした銘柄・時期）を根拠に仮説を1文で言語化してから着手する。

## Do（最小差分で実装）

- 仮説に対応する最小の差分のみをコミット対象にする。ついで修正・リファクタは混ぜない。

## Check（bear-backtestスキルで前後比較）

- `bear-backtest` スキルの手順で **変更前(baseline)と変更後**の bear バックテストを両方実行し数値を比較する（未実施なら先にbaselineを取得してから変更する）。
- `python3 tests/test_*.py` を実行し、既存テストがデグレしていないことを確認する。

## Act（マージ可否の判定・機械的に従う）

CLAUDE.md §改善マージ規律を機械的に適用する：

- **平均リターン・勝率・大勝率のいずれも改善しない場合 → マージ禁止**。
  - `git checkout main -- .` 相当で変更を破棄しブランチも削除（実験コードを残さない、CLAUDE.md §7）。
  - `dev_log.md` に「不採用」として仮説・結果・却下理由を1エントリ追記してmainに直接コミット（次回以降の重複試行を防ぐログとして残す）。
- **1つ以上改善し、既存テストもパス → 採用**。
  - `dev_log.md` に bear-backtest スキルの報告フォーマットで結果を追記。
  - 機能追加・変更なら README.md も同一コミットで更新（CLAUDE.md §7）。
  - コミットしてブランチを push し、**draft PR を作成する**（本番の配信パイプライン・実ユーザーへのLINE配信に影響するため、mainへの直接マージやauto-mergeは行わない。最終マージはCLAUDE.md §8の確認〔最新main取り込み・コンフリクトなし・デグレなし〕を経てから人が実行する）。

## 自動実行（Routineから呼ばれた場合）

- 上記フローをそのまま実行し、最後に「Plan仮説 / Check結果 / Act判定（採用してPR作成 or 不採用でdev_log記録のみ）」を短く報告する。
- 改善余地が見当たらない回は無理に変更を作らず、「今回は見送り」と報告して終了してよい（無理な改善の量産は禁止）。
