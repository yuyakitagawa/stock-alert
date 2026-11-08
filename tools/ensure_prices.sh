#!/usr/bin/env bash
# 判定日の終値（個別株・日経平均）が欠けていれば株価を取り直す。daily_alert.yml Step 0r と dip_alert.yml が使う。
# 欠け = 個別株が前の取引日の90%未満、または個別株があるのに日経平均が無い（dip_buy_alert.py --check-prices）。
# 日本時間0時を越えるとYahooから当日の日足が取れなくなる（推定。11/4 00:13開始は23銘柄、11/5 00:14開始は2銘柄、
# 11/6 23:51開始は0時までに取れた1,073銘柄。01:53開始の11/2はそろった）。
# JST 0〜1時台なら 02:05 JST まで待って1回、それ以外は30分おきに最大2回取り直す。待ちは合計120分まで。
# 終了コード: そろえば0、取り直しても欠けていれば1。
cd "$(dirname "$0")/.."
WAITED=0
for i in 1 2; do
  python web/dip_buy_alert.py --check-prices && exit 0
  H=$(TZ=Asia/Tokyo date +%H); M=$(TZ=Asia/Tokyo date +%M)
  if [ "$H" -lt 2 ]; then
    WAIT=$(( (2 * 60 + 5 - (10#$H * 60 + 10#$M)) * 60 )); LAST=1
  else
    WAIT=1800; LAST=0
  fi
  if [ $((WAITED + WAIT)) -gt 7200 ]; then WAIT=$((7200 - WAITED)); fi
  WAITED=$((WAITED + WAIT))
  echo "当日の終値が欠けている。${WAIT}秒後に取り直す ($i/2)"
  sleep "$WAIT"
  python tools/fetch_history.py --years 1
  if [ "$LAST" = 1 ]; then break; fi
done
python web/dip_buy_alert.py --check-prices
