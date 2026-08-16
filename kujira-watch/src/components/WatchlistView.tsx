"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import ActionButton from "@/components/ActionButton";
import { formatDate } from "@/lib/format";
import {
  WATCHLIST_CHANGE_EVENT,
  getWatchlist,
  removeWatch,
  type WatchItem,
} from "@/lib/watchlist";

type LatestDates = { stocks: Record<string, string>; investors: Record<string, string> };

// /watchlist の本体。localStorageからの読み込みはマウント後にしか行えないため、
// 読み込み完了までは何も描画しない（「0件です」の誤フラッシュを避ける）。
export default function WatchlistView() {
  const [items, setItems] = useState<WatchItem[] | null>(null);
  const [latest, setLatest] = useState<LatestDates | null>(null);

  useEffect(() => {
    const sync = () => setItems(getWatchlist());
    sync();
    window.addEventListener(WATCHLIST_CHANGE_EVENT, sync);
    return () => window.removeEventListener(WATCHLIST_CHANGE_EVENT, sync);
  }, []);

  // 「保存したあとの新着に気づけない」対策として、各項目の最新開示日を取得して添える。
  // 取得失敗・遅延中でもリスト本体の表示は止めない。
  useEffect(() => {
    if (!items || items.length === 0) return;
    const stocks = items.filter((i) => i.type === "stock").map((i) => i.id);
    const investors = items.filter((i) => i.type === "investor").map((i) => i.id);
    const params = new URLSearchParams();
    if (stocks.length > 0) params.set("stocks", stocks.join(","));
    if (investors.length > 0) params.set("investors", investors.join(","));
    fetch(`/api/watchlist-latest?${params.toString()}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: LatestDates | null) => {
        if (data) setLatest(data);
      })
      .catch(() => {});
  }, [items]);

  if (items === null) return null;

  if (items.length === 0) {
    return (
      <div className="text-sm leading-relaxed text-foreground/70">
        <p className="mb-4">
          まだ何もウォッチしていません。投資家ページ・銘柄ページの「☆
          ウォッチする」ボタンで保存すると、ここから再訪できます（この端末のブラウザにのみ保存されます）。
        </p>
        <div className="flex flex-wrap gap-3">
          <ActionButton href="/investors">投資家一覧から探す</ActionButton>
          <ActionButton href="/stocks">銘柄一覧から探す</ActionButton>
        </div>
      </div>
    );
  }

  const investors = items.filter((i) => i.type === "investor");
  const stocks = items.filter((i) => i.type === "stock");

  const latestDateOf = (item: WatchItem): string | undefined =>
    item.type === "stock" ? latest?.stocks[item.id] : latest?.investors[item.id];

  const section = (title: string, list: WatchItem[], hrefOf: (item: WatchItem) => string) =>
    list.length > 0 && (
      <div className="mb-8 border-t border-rule pt-4">
        <h2 className="mb-2 text-sm font-bold text-brand-navy">{title}</h2>
        <List disablePadding dense>
          {list.map((item) => {
            const latestDate = latestDateOf(item);
            return (
              <ListItem
                key={`${item.type}-${item.id}`}
                disableGutters
                sx={{ py: 0.5, gap: 1, flexWrap: "wrap" }}
                secondaryAction={
                  <IconButton
                    edge="end"
                    size="small"
                    aria-label={`${item.label}をウォッチリストから外す`}
                    onClick={() => removeWatch(item.type, item.id)}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                }
              >
                <Link href={hrefOf(item)} className="text-sm text-brand-blue hover:underline">
                  {item.label}
                </Link>
                {latestDate && (
                  <span className="text-xs text-foreground/50">
                    最新開示: {formatDate(latestDate)}
                  </span>
                )}
              </ListItem>
            );
          })}
        </List>
      </div>
    );

  return (
    <div>
      {section("ウォッチ中の投資家", investors, (i) => `/investors/${encodeURIComponent(i.id)}`)}
      {section("ウォッチ中の銘柄", stocks, (i) => `/stocks/${i.id}`)}
    </div>
  );
}
