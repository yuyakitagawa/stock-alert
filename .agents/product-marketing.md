# Product Marketing Context

**Document version:** v1（コードベースからの自動ドラフト。要レビュー）
**Last updated:** 2026-08-22

> marketingskills の全スキルが最初に読む共通コンテキスト。更新は `/product-marketing` スキルで行う。

## Product Overview
**One-liner:** EDINET大量保有報告書をもとに「クジラ」（大口投資家）の売買を毎時検知し、速報記事・X・YouTube Shortsで届けるメディア
**What it does:** 機関投資家・アクティビスト・インサイダー・自社株買いなどの開示を自動で記事化（kujira-watch.com）。銘柄ページ（約2,978銘柄）・投資家ページ・ランキング・週次まとめ・日付アーカイブで横断検索できる。別系統として、機械学習による下落確率ランキング（stock-alert）をLINE/メールで配信する。
**Product category:** 株式投資情報メディア／開示情報（EDINET）解説ブログ
**Product type:** メディア（無料・広告モデル）。Next.js + microCMS + Supabase、Vercel
**Business model:** 無料閲覧。収益化はAdSense（月間PVが現状の5倍＝目安3,500〜5,500に達したら導入）

## Target Audience
**Target companies:** N/A（B2C）
**Decision-makers:** 日本株の個人投資家（中長期・GARP志向）、開示情報を自分で追う中級者、金融系ライター・アナリスト
**Primary use case:** 「誰が・どの銘柄を・どれだけ買った/売ったか」を開示当日のうちに知る
**Jobs to be done:**
- 大口の動きを見逃さず、自分の保有/監視銘柄への影響を即日把握する
- 投資家名や銘柄名から過去の保有推移をまとめて確認する
- 開示の一次情報を読む手間を省き、要点だけ受け取る
**Use cases:**
- 朝/夜のルーティンで「本日のクジラ」を確認
- 気になる銘柄ページで大量保有の履歴を見る
- アクティビストの新規取得を週次で追う

## Personas
| Persona | Cares about | Challenge | Value we promise |
|---------|-------------|-----------|------------------|
| 個人投資家（中級） | 大口の売買方向・比率・金額 | EDINETは読みにくく網羅が無理 | 開示当日に要点を記事化 |
| 銘柄リサーチャー | 銘柄×投資家の履歴 | 情報が散在 | 銘柄/投資家ページで一括閲覧 |
| 速報派のXユーザー | いち早い検知 | タイムラインのノイズ | 毎時の自動速報・日次サマリー |

## Problems & Pain Points
**Core problem:** 大量保有報告書は毎日大量に出るが、一次情報は読みにくく、誰が重要な開示かを判断するのに時間がかかる
**Why alternatives fall short:**
- EDINET本体: 検索・通知が弱く、金額換算や過去推移が無い
- 証券会社の適時開示通知: 大量保有報告書は対象外か埋もれる
- 有料情報サービス: 個人には高額、速報性も毎時ではない
**What it costs them:** 重要な買い集め/売却に気づくのが数日遅れる。調査に毎日30分以上かかる
**Emotional tension:** 「自分だけ知らなかった」不安、情報の真偽への疑い

## Competitive Landscape
**Direct:** 大量保有報告書まとめ系サイト・Xアカウント — 更新が手動で遅い、銘柄/投資家の横断ページが無い
**Secondary:** 株探・みんかぶ等の総合株式情報サイト — 大量保有は一機能で深掘りが無い
**Indirect:** 自分でEDINETを毎日確認する — 時間がかかり継続できない

## Differentiation
**Key differentiators:**
- 平日9〜21時 毎時の自動検知・記事化（開示当日に出る）
- 投資家ページ（保有推移・過去売買の集約）と銘柄ページ（約2,978件）の双方向リンク
- 答え合わせ投稿（3ヶ月前の開示のその後）で結果を隠さない
- 英語版ページ、構造化データ、RSSなど検索/AI引用対応
**How we do it differently:** EDINET APIを直接監視し、Claudeで記事と動画台本を生成、人手を介さず配信
**Why that's better:** 速い・網羅的・継続する
**Why customers choose us:** 当日中に「誰が何を買ったか」を一覧で追える唯一に近い無料サイト

## Objections
| Objection | Response |
|-----------|----------|
| 自動生成の記事は信用できるのでは | 一次情報（EDINET）への出典リンクと開示日を必ず明記。誤りは削除せず訂正を出す |
| 投資助言なのでは | データ提供のみで助言はしない。将来株価・目標株価・売買推奨は書かない |
| 情報が多すぎる | 注目枠（金額順）、週次まとめ、投資家別ページで絞り込める |

**Anti-persona:** デイトレード・短期の値動き予想を求める人、「買い時」を教えてほしい人

## Switching Dynamics
**Push:** EDINETの読みにくさ、情報の遅れ
**Pull:** 毎時の速報と投資家/銘柄の履歴が無料
**Habit:** 慣れた総合サイト・SNSで済ませている
**Anxiety:** AI生成記事の正確性、サイトの継続性

## Customer Language
**How they describe the problem:**
- 「大量保有報告書、全部見きれない」
- 「あのファンドが買ってたの今知った」
**How they describe us:**
- 「クジラの動きが当日わかる」
**Words to use:** クジラ、大口投資家、大量保有報告書、保有比率、新規取得、買い増し、売却、アクティビスト、自社株買い、推定・約（推定値には必ず付ける）
**Words to avoid:** 必ず／確実／絶対／買い時／推奨／暴騰／暴落／爆益、目標株価、特定銘柄の売買の勧め、企業・個人への評価的形容
**Glossary:** クジラ＝相場を動かすほどの資金力を持つ大口投資家。大量保有報告書＝上場株式を5%超保有した際にEDINETへ提出する開示。答え合わせ＝3ヶ月前の開示のその後の株価を報告するX投稿

## Brand Voice
**Tone:** 淡々・事実ベース・中立。相場急落日も数字だけを出す
**Style:** 短文、事実と推定を書き分ける、一次情報へ出典リンク
**Personality:** 正確、速い、誠実（誤りは訂正で残す）、控えめ、継続的

## Proof Points
**Key metrics:** 銘柄ページ約2,978件、平日毎時の自動更新、英語版あり
**Notable customers/logos:** なし
**Testimonials:** なし（要収集）
**Value themes:** 速報性（当日）、網羅性（全開示を記事化）、検証性（答え合わせ）

## Goals
**Primary business goal:** 月間PVを現状の5倍（3,500〜5,500）に伸ばしてAdSense導入
**Key conversion action:** X（@kujira_watch）フォロー、サイト再訪（RSS/ブックマーク）。将来はLINE公式アカウント友だち追加
**Current metrics:** X: 最終指標はフォロワー数。中間指標はプロフィールクリック率→平均インプレッション→平均ブックマーク→リンククリック（いいね数は使わない）。サイト: GA4（X経由は `utm_source=x&utm_medium=profile`）、Vercel Analytics

## Changelog
- v1 (2026-08-22): README・docs/x_operation_rules.md・aboutページからの自動ドラフト作成
