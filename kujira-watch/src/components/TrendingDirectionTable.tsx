"use client";

import { useMemo, useState } from "react";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import TrendingTable from "./TrendingTable";
import { selectDirection } from "@/lib/trendingStats";
import type { DirectionalTrendingEntry, TrendingDirection } from "@/lib/trendingStats";

// hrefとnote（事業内容）・sector（業種アイコン用）はサーバー側で解決してから渡す
// （ここはクライアントコンポーネントなので関数propsを境界を越えて渡せない）。
export type DirectionalTrendingItem = DirectionalTrendingEntry & {
  href: string | null;
  note?: string | null;
  sector?: string | null;
};

// 既定は「買い」。大口投資家がどの銘柄を買い集めているかを見に来る読者が大半で、
// 売り方向の開示まで混ぜると買いの動きが埋もれるため。
const DEFAULT_DIRECTION: TrendingDirection = "buy";

const OPTIONS: { value: TrendingDirection; label: string }[] = [
  { value: "buy", label: "買い" },
  { value: "sell", label: "売り" },
  { value: "both", label: "両方" },
];

// /trendingの銘柄一覧に売買方向の絞り込みを足したラッパー。3方向の件数は最初から
// 手元にあるので、切り替えでも追加のfetchは発生しない。
// ボタンの見た目は一覧ページの絞り込みナビ（FilterButtonNav）に合わせているが、
// あちらはURL遷移のリンク、こちらはその場で切り替えるボタンなので別実装にしている
// （searchParamsを読むとページがリクエストごとの動的レンダリングになるため）。
export default function TrendingDirectionTable({
  items,
  windowDays,
}: {
  items: DirectionalTrendingItem[];
  windowDays: number;
}) {
  const [direction, setDirection] = useState<TrendingDirection>(DEFAULT_DIRECTION);
  const shownItems = useMemo(() => selectDirection(items, direction), [items, direction]);

  return (
    <>
      <Stack
        role="group"
        aria-label="開示の売買方向で絞り込む"
        direction="row"
        className="no-scrollbar"
        sx={{
          gap: 1,
          mb: 2,
          flexWrap: { xs: "nowrap", sm: "wrap" },
          overflowX: { xs: "auto", sm: "visible" },
          "& .MuiButton-root": { flexShrink: 0 },
        }}
      >
        {OPTIONS.map((option) => (
          <Button
            key={option.value}
            type="button"
            variant={option.value === direction ? "contained" : "outlined"}
            aria-pressed={option.value === direction}
            onClick={() => setDirection(option.value)}
          >
            {option.label}
          </Button>
        ))}
      </Stack>

      {shownItems.length === 0 ? (
        <p className="text-sm text-ink-tertiary">
          直近{windowDays}日間で前期間より
          {direction === "both" ? "開示" : `${direction === "buy" ? "買い" : "売り"}の開示`}
          が増えた銘柄はありません。
        </p>
      ) : (
        // 絞り込みを変えたら「もっと見る」で足した分は数え直す（keyで作り直す）。
        <TrendingTable key={direction} items={shownItems} headLabel="銘柄" windowDays={windowDays} />
      )}
    </>
  );
}
