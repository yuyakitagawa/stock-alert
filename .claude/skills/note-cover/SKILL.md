---
name: note-cover
description: note記事用の表紙画像（1280x670）を生成する。絶景写真に黒帯＋極太白文字を左寄せで合成する。note表紙・サムネ・アイキャッチ・記事カバー画像の作成を求められたときに使う。
---

# note 表紙ジェネレータ

## 絶対ルール（必ず守ること）

1. **必ず `note_cover.py` スクリプトで合成せよ。** SVG・ウィジェット・show_widget・canvas-design・その他の方法で画像を作るな。
2. **出力は必ず実写写真ベースの PNG ファイル。** イラスト・ベクター・手描き風は禁止。
3. **スクリプトのパスは固定**: `/Users/kitagawayuuya/stock-alert/.claude/skills/note-cover/note_cover.py`
4. **Python は venv の絶対パスで実行**: `/Users/kitagawayuuya/stock-alert/venv/bin/python3`

## フォーマット（固定仕様・変更禁止）

- 背景: 実写の全面写真（中央 cover-crop）
- 黒帯: 横幅いっぱい・上下中央・不透明度 222/255（約87%）
- 文字: ヒラギノ角ゴ **W6（太字 / ttc index=2）**・**左寄せ**・2倍解像度（SS=2）で描いて1280×670に縮小
- 出力: 1280×670 PNG（note推奨 1.91:1）
- 保存先: `~/Desktop/note_cover_<slug>.png`

> **禁止事項**: index=0/1（W3）を使う、ストロークで太らせる、中央揃えにする、黒帯以外の装飾を加える

## 手順（この順序で必ず実行）

### Step 1. コピーを決める（2行・各15文字前後）
- 記事URLがあれば WebFetch でタイトルと結論を取得
- 記事の核心メッセージを **2行** に凝縮する（煽り＋本質の対比が効く）
- ユーザーがタイトルを指定した場合はそのまま2行に分割して使う

### Step 2. 背景写真を用意する
- フリー素材は **Pexels**（商用可・クレジット不要）を使う
- 記事テーマに合う絶景を **複数候補ダウンロード** → **Read で目視確認** → 最適な1枚を採用
- ユーザーが写真パスを渡した場合はそのファイルを使う

```bash
curl -sL "https://images.pexels.com/photos/<ID>/pexels-photo-<ID>.jpeg?auto=compress&cs=tinysrgb&w=1400" -o /tmp/cover_src.jpg
```

### Step 3. note_cover.py で合成する（これ以外の方法は禁止）

```bash
/Users/kitagawayuuya/stock-alert/venv/bin/python3 \
  /Users/kitagawayuuya/stock-alert/.claude/skills/note-cover/note_cover.py \
  --photo /tmp/cover_src.jpg \
  --line1 "1行目コピー" \
  --line2 "2行目コピー" \
  --out ~/Desktop/note_cover_<slug>.png
```

オプション（通常はデフォルトでOK）:
- `--pad-left 110` — 左余白(px)
- `--opacity 222` — 黒帯の不透明度(0-255)
- `--max-font 96` — 最大フォントサイズ(px)

### Step 4. 確認して報告
- 出力PNGを **Read で目視確認**（潰れ・はみ出し・滲みがないか）
- 保存先パスをユーザーに伝える

## 注意
- 写真は商用可ライセンスのみ（著作権のある画像を勝手に使わない）
- show_widget やウィジェットプレビューは使わない（外部画像URLを読めないため無意味）
- チャットに貼られた画像はファイルパスが不明なので、ユーザーにファイル保存を依頼する
