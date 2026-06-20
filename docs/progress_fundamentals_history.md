# 進捗: 過去ファンダメンタルのバックテスト対応

## 目的
backtest.py が6つのファンダ特徴量（PER/PBR/ROE/決算日/配当日/優待日）を
中立値(0.0/0.5)で埋めている問題を解消し、過去日時点の実値を再構成して
ファンダ入りモデルを正しく検証できるようにする。

## 設計
- **日付系3特徴量**（決算/配当/優待まで日数）: 発表日スケジュール + 確定月から
  任意の過去日で決定論的に計算（保存不要、スケジュールのみ）
- **PER/PBR/ROE**: 年度別 EPS・BPS・ROE + 発表日を `fundamentals_annual` に保存。
  backtest時に `PER = 過去株価 / その時点で既知のEPS` で再構成。

## データソース
kabutan.jp/stock/finance/?code=XXX
- 通期業績テーブル: 年度別 EPS + 配当 + 発表日(YY/MM/DD)
- 財務指標テーブル: 年度別 ROE・BPS(1株純資産)

## ステップ
- [x] 1. kabutanの過去データ構造を調査（EPS/ROE/発表日が年度別に取得可能と確認）
- [x] 2. progress + DBスキーマ作成（fundamentals_annual テーブル）
- [x] 3. 過去ファンダ取得スクリプト tools/fetch_fundamentals_history.py 作成
- [x] 4. 数銘柄でEPS/BPS/ROE/発表日のパースを検証
- [x] 5. 全銘柄(約3,734)の年度別ファンダを取得しDB格納（3,566銘柄 / 17,741行。EPS99%/ROE58%/BPS60%）
- [x] 6. backtest.py に point-in-time ファンダ再構成を組み込み（get_pit_fundamentals + 全呼び出し箇所）
- [ ] 7. ファンダ入りモデルを再学習（44次元、正しく完了させる）← 実行中
- [ ] 8. backtest bear + rolling で「ファンダあり vs なし」を比較
- [ ] 9. 結果を pdca_log.md / README に記録、コミット

## メモ
- 6/6 朝の再学習は head -20 パイプ切断で途中停止 → モデルは6/5の38次元のまま。要再実行。
- 財務指標テーブルのROE列インデックスは要確認（暫定: ヘッダー照合で特定）。
- ステップ5完了: PER中央値18.5、ROE中央値8.0% で妥当。fundamentals_annual に16,331行格納。
- ステップ6完了: extract_features_at に fundamentals 引数追加、rolling/bear 両経路で
  get_pit_fundamentals を呼ぶよう統一。中立padは get_pit_fundamentals 失敗時のみ。
