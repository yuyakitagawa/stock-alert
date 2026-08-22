---
name: jp-copy
description: kujira-watch / X / note 向けの日本語コピー（X投稿本文・サイト説明文・note導入文・プロフィール文・CTA）を書く／直すときに使う。copywriting・ogilvy・stop-slop の原則を日本語とX文字数制限（270単位）に合わせて適用する。「コピーを書いて」「投稿文を改善」「説明文を直して」「キャッチコピー」で起動。
---

# jp-copy — 日本語コピー作成（copywriting + ogilvy + stop-slop の日本語・X適用版）

## 最初に読む
1. `.claude/skills/copywriting/SKILL.md` の「Copywriting Principles」「Writing Style Rules」
2. `.claude/skills/ogilvy/SKILL.md` の「Headlines」「Body Copy」
3. `.claude/skills/stop-slop/SKILL.md` の「Core Rules」（仕上げ時）

## このプロジェクトの前提（聞かずに使う）
- 商品: kujira-watch.com（ブランド名「大口投資家の監視ブログ」）。EDINET大量保有報告書＝一次情報を毎日集計・解説。
- 読者: 日本株の個人投資家。初心者〜中級。「誰が何を買ったか」を手軽に知りたい。
- 唯一の信頼根拠: **5%超の株主は売買の開示が法律で義務** → 噂でなく公開データ。この一文はコピーの「事実」として優先的に使う。
- 投資判断を煽らない。「買い推奨」「爆上げ」「必見」は禁止（方針=GARP・誠実）。
- 数字は実データのみ。ダミー数値・架空の実績・架空の声を作らない（`Honest over sensational`）。

## 日本語への翻訳ルール
| 原則（英語スキル） | 日本語での実装 |
|---|---|
| Rhetorical question | 1行目を読者主語の疑問文にする「先週、大口投資家が買い増した日本株は？」 |
| Include brand + promise in headline | 見出しに「大口投資家／大量保有報告書」と「わかる・毎日・無料」のどれかを入れる |
| Specific over vague | 「注目」「話題」「最新」→ 件数・社名・日付に置き換える |
| No exclamation points | 「！」禁止。絵文字は行頭の記号用途に1〜2個まで |
| Active over passive | 「〜されています」→「〜しています／集計しています」 |
| Facts sell | 1投稿・1段落に「開示義務」「件数」「日付」のいずれか1つ以上 |
| CTA | 「続きはプロフィールのリンクから」固定（URLは入れない＝X課金回避） |

## X投稿の制約（必ず守る）
- 上限 **270単位**（全角2・半角1）。書いたら必ず `web.x_post_format.weighted_len()` で実測する:
  `venv/bin/python3 -c "from web.x_post_format import weighted_len; print(weighted_len(open('/dev/stdin').read()))" <<'T' ... T`
- 超えたら削る順序: ①証券コード ②投資家の2人目以降 ③銘柄4位以下 ④説明の1文。**1行目（疑問文）と開示義務の事実は最後まで残す**。
- 1行目だけで内容が伝わること（タイムラインでは1行目しか見えない）。
- ハッシュタグは末尾1行に既存の `#日本株 #大量保有報告書` をそのまま。

## 出力形式
1. 案を**2〜3本**（疑問文型／事実型／数字型）
2. 各案に単位数（X用）または文字数
3. どれを推すかと理由を1行
4. 仕上げに stop-slop の Quick Checks を当てる（「〜することで」「〜の世界へ」「重要です」「ぜひ」の連発を除去）
