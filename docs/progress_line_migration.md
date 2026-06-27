# LINE webhook移行 & frontend削除 進捗

## ステップ
- [x] 1. LINE webhook を Supabase Edge Function に移行（frontend/app/api/line/webhook/route.ts → supabase/functions/line-webhook/）
- [x] 2. Supabase Edge Function デプロイ & 動作確認
- [x] 3. LINE Developers でwebhook URLを新URLに変更（ユーザー作業）
- [x] 4. frontend/ ディレクトリ全削除
- [x] 5. frontend_build.yml 削除
- [x] 6. web/export_to_web.py の frontend依存箇所を修正（ISR無効化・gen_risk_regime export削除）
- [x] 7. CLAUDE.md / README.md 更新
- [x] 8. コミット & プッシュ

## ユーザー作業（要対応）
- [ ] Supabase Edge Function に secrets を設定: LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN, ANTHROPIC_API_KEY
- [ ] LINE Developers で webhook URL を変更: `https://kxrgyguowxtjqexvmlgx.supabase.co/functions/v1/line-webhook`
