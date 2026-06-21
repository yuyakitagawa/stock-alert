# 設計: スマホからの指示・実行基盤（Mobile Instruction）

## 0. このドキュメントの位置づけ
「スマホからバックテスト等を実行したい」という当初の要望と、その実現手段として構築中の
**LINE Bot + GitHub Actions** が、同じ一つの仕組みであることを整理するための設計書。
作業が行ったり来たりするため、全体像・判断根拠・現状をここに集約する。
（日々の細かい進捗・チェックリストは `docs/progress_line_bot.md` 側で管理）

---

## 1. 背景と目的

### 当初の要望（出発点）
- PCを開かずに**スマホから** stock-alert を操作したい。
- 代表的なユースケース:
  1. **バックテストの実行**（`tools/backtest.py bear` / bear-backtest スキル）と結果確認
  2. フィルター条件・パラメータの調整指示
  3. スクリーニング/ランキングの再実行・結果確認
  4. その他「○○して」という自然言語での依頼

### 課題
- これらは本来ターミナルで `python3 ...` を叩く作業。スマホには向かない。
- GitHub Actions の `workflow_dispatch` は手動実行できるが、毎回フォームを開くのは面倒で、
  自然言語の「相談」ができない。

### ゴール
> スマホのチャット（LINE）に日本語で書くだけで、
> Claude が内容を理解し、必要なら GitHub Actions 上でコード変更・実行まで行う。

---

## 2. 全体アーキテクチャ

```
┌─────────┐   ①メッセージ    ┌──────────────────────────┐
│ スマホ   │ ───────────────▶ │ Vercel: stock-alert-web      │
│  LINE   │                  │  /api/line (route.ts)        │
│         │ ◀─────────────── │   ├ 署名検証                  │
└─────────┘   ⑤返信           │   ├ 会話履歴(Supabase)        │
                              │   ├ Claude API 呼び出し ②     │
                              │   └ [ACTION]判定 ③            │
                              └────────────┬─────────────┘
                                           │ ④ workflow_dispatch
                                           ▼
                              ┌──────────────────────────┐
                              │ GitHub Actions: instruct.yml │
                              │  Claude CLI で指示を実行      │
                              │  （backtest等）→ 自動commit   │
                              └──────────────────────────┘
```

### 処理フロー
1. ユーザーがLINEにメッセージ送信 → LINE Platform が `/api/line` にWebhook POST。
2. **アクセス制御**: 送信者が `LINE_OWNER_USER_ID` と一致するか確認。不一致は拒否（Actionsを起動させない）。
   未設定時はブートストラップとして送信者に自分のuserIdを返すだけ。
3. `/api/line` が Claude API（Haiku）にメッセージ＋会話履歴＋システムプロンプトを渡す。
4. Claude の返答が `[ACTION]` で始まる＝**作業が必要な指示**と判定。
5. その場合 GitHub の `instruct.yml` を `workflow_dispatch` で起動（入力=ユーザーの原文）。
   - Actions側では Claude CLI が指示を解釈し、バックテスト実行やコード修正を行い、変更があれば自動commit。
6. **結果通知**: Actions完了時、Claude出力の末尾要約を `/api/line/notify`（`INTERNAL_SEND_SECRET`認証）
   へPOST → **Push API** でオーナーのLINEへ結果を届ける（reply tokenは失効が早く非同期返信に使えないため）。

### zenn記事(line-to-claude-code)から取り込んだアイデア
SaaS構成（Vercel+Actions）は維持し、設計アイデアのみ移植した（記事本体のCloudflare Tunnel/Hono/Bun/MCP
＝ローカルClaude Code駆動の構成は不採用）。
- **オーナー限定アクセス**: 記事のペアリング制限に倣い、`LINE_OWNER_USER_ID` で単一ユーザーに制限。
- **Push APIで非同期結果通知**: 記事同様、reply tokenの即時失効を避けPush APIで結果を返す。
- 署名検証は生body（JSONパース前）で実施済み（記事の要点と一致）。
5. LINE へ返信（即答 or 「実行を開始しました」）。会話履歴は Supabase に保存。

---

## 3. コンポーネント詳細

| コンポーネント | 場所 | 役割 |
|---|---|---|
| `instruct.yml` | `stock-alert/.github/workflows/` | workflow_dispatch で指示を受け、Claude CLI が実行→自動commit |
| `/api/line` | `stock-alert-web/src/app/api/line/route.ts` | LINE Webhook受け口。署名検証・Claude呼び出し・Actions起動 |
| `line_conversations` | Supabase テーブル | ユーザーごとのマルチターン会話履歴 |
| LINE公式アカウント | LINE Developers Console | チャネル。Webhook URL = `https://stock-alert-web.vercel.app/api/line` |
| 環境変数 | Vercel | `LINE_CHANNEL_SECRET` / `LINE_CHANNEL_ACCESS_TOKEN` / `GITHUB_TOKEN` ほか |

### 「即答」と「実行」の振り分け
- **即答系**（質問・相談・状況確認）: Claude が直接答えてLINEに返す。Actions起動なし。
- **実行系**（バックテスト・修正・再ランキング）: Claude応答が `[ACTION]` で始まる → Actions起動。
  - システムプロンプトで「作業が必要なら先頭に `[ACTION]` を付ける」と指示している。

---

## 4. なぜこの構成か（設計判断）

- **LINEを選んだ理由**: スマホで最も自然に使える日本語チャットUI。通知も届く。
  （Web/Telegram も候補だったが、ユーザーの普段使いに合わせLINEに決定）
- **GitHub Actions で実行する理由**: 既存のCI/CD・秘密情報・Python環境がすでに整っている。
  Vercel上で重いPython処理（バックテスト）は動かさず、Actionsに委譲する。
- **Vercelを経由する理由**: LINE Webhookは常時稼働HTTPSが必要。既存の `stock-alert-web` を流用。

---

## 5. セキュリティ

- **LINE署名検証**: `x-line-signature` を `LINE_CHANNEL_SECRET` でHMAC-SHA256検証。不一致は401。
  （検証スキップは禁止。過去にデバッグ目的でスキップしようとしたが安全上却下した経緯あり）
- **GitHub Actions の Claude CLI** は `--dangerously-skip-permissions` で動く。
  → だからこそ `/api/line` の署名検証が**唯一の入口ガード**として重要。
- **秘密情報**はVercel/GitHub Secretsに保管。コード・ログ・URLに平文で出さない。

---

## 6. 現状の実装状況

### 完了
- [x] `instruct.yml` 作成
- [x] `/api/line/route.ts` 作成（署名検証・会話履歴・Claude呼び出し・[ACTION]判定）
- [x] Supabase `line_conversations` テーブル作成
- [x] Vercel環境変数登録（SECRET は LINE側 `17e45…221e` と一致, secretLen=32 確認済）
- [x] LINE設定（応答メッセージOFF / Webhook ON / URL設定）
- [x] **Root Directory が誤って `frontend` になっていた問題を空欄に修正**（404の根本原因）
- [x] `vercel --prod` でCLI本番デプロイ成功（`stock-alert-web-q3vbxd4xt-…`）
- [x] `/api/line` GET が `{"hasSecret":true,"secretLen":32,…}` を返すこと確認（ルート稼働確認）

### 未解決・残課題
- [ ] `/api/line` の **HTTPステータス404 と body正常JSON の矛盾**を解明
      （`vercel inspect stock-alert-web.vercel.app` で本番エイリアスが最新デプロイを指すか確認。
       エッジキャッシュ/エイリアス伝播 or `?cb=` クエリの扱いを疑う）
- [ ] GitHub push での**自動デプロイが効いていない疑い**（Git連携設定の確認）
- [ ] LINE Console で Webhook「Verify」成功
- [ ] スマホLINEで実メッセージE2Eテスト
- [ ] **デバッグコード削除**（`route.ts` の `export async function GET` と `[LINE] sig verify failed` ログ）

---

## 7. 動作確認コマンド

```bash
# ルートが本番に乗っているか（200 + JSONなら成功）
curl -s "https://stock-alert-web.vercel.app/api/line"

# 本番エイリアスが指すデプロイの確認
cd ~/bussiness/stock-alert-web && npx vercel inspect stock-alert-web.vercel.app
```

---

## 8. 将来の拡張（メモ）
- バックテスト結果のグラフ/サマリをLINEに画像で返す。
- `instruct.yml` 実行完了時にLINEへ結果をプッシュ通知（現状は「開始しました」のみ）。
- 既存スキル（bear-backtest / weekly-review / web-republish）をLINEから直接トリガー。
