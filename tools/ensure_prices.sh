#!/usr/bin/env bash
# 判定日の終値（個別株・日経平均）が欠けていれば株価を取り直す。daily_alert.yml Step 0r と dip_alert.yml が使う。
# 欠け = 個別株が前の取引日の90%未満、または個別株があるのに日経平均が無い（dip_buy_alert.py --check-prices）。
# 取り直すのは大引け後〜日付が変わるまで（JST 15:30〜23:59）に始まったときだけ。30分おきに最大2回。待つ間に0時を越えたらそこで終わる。
# 日本時間0時を越えると当日の日足は数時間たっても返らない（11/9分: 03:07・04:14・05:20 JSTの3回とも
# 個別株2銘柄・日経平均なし）ので待たない。その日は翌日以降の便で dip_buy_alert.py が判定し直す（catchup_dates）。
# 終了コード: そろえば0、欠けたままなら1。
cd "$(dirname "$0")/.."
python web/dip_buy_alert.py --check-prices && exit 0
HM=$(TZ=Asia/Tokyo date +%H%M)
if [ "$((10#$HM))" -lt 1530 ]; then
  echo "日本時間0時を越えているため取り直さない（翌日以降の便で判定し直す）"
  exit 1
fi
for i in 1 2; do
  echo "当日の終値が欠けている。1800秒後に取り直す ($i/2)"
  sleep 1800
  if [ "$((10#$(TZ=Asia/Tokyo date +%H%M)))" -lt 1530 ]; then
    echo "待つ間に日本時間0時を越えたため取り直さない（翌日以降の便で判定し直す）"
    exit 1
  fi
  python tools/fetch_history.py --years 1
  python web/dip_buy_alert.py --check-prices && exit 0
done
exit 1
