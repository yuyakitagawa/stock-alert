"use client";

import { useEffect, useState } from "react";
import Button from "@mui/material/Button";
import {
  WATCHLIST_CHANGE_EVENT,
  isWatched,
  toggleWatch,
  type WatchItemType,
} from "@/lib/watchlist";

// 投資家ページ・銘柄ページの見出し横に置く「ウォッチする」トグルボタン。
// localStorageはSSRでは読めないため、初期描画は常に未保存表示にし、
// マウント後に実際の保存状態へ同期する（ハイドレーション不一致を避ける）。
export default function WatchButton({
  type,
  id,
  label,
}: {
  type: WatchItemType;
  id: string;
  label: string;
}) {
  const [watched, setWatched] = useState(false);

  useEffect(() => {
    const sync = () => setWatched(isWatched(type, id));
    sync();
    window.addEventListener(WATCHLIST_CHANGE_EVENT, sync);
    return () => window.removeEventListener(WATCHLIST_CHANGE_EVENT, sync);
  }, [type, id]);

  return (
    <Button
      variant={watched ? "contained" : "outlined"}
      onClick={() => setWatched(toggleWatch({ type, id, label }))}
      aria-pressed={watched}
    >
      {watched ? "★ ウォッチ中" : "☆ ウォッチする"}
    </Button>
  );
}
