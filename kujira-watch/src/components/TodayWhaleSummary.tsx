"use client";

import Link from "next/link";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { formatDate, formatDealAmount } from "@/lib/format";

// TOPの冒頭に置く最新開示サマリー。以前は見出しを「今日の注目取引」で固定していたが、
// 実際の最新開示が何日付なのか・その日に何件いくら動いたのかは記事カードを読むまで
// 分からなかった。開示日と件数・金額を最初に大きく出すことで、毎日更新されている
// ことが一目で分かるようにする（Discover・リピーター向けの鮮度表示）。
export default function TodayWhaleSummary({
  date,
  href,
  count,
  buyCount,
  buyAmount,
  sellCount,
  sellAmount,
  disclosuresFixed,
}: {
  date: string;
  /** 取引日ページへのリンク。開示が少なくページを公開していない日はnull（lib/publishedPages.ts）。 */
  href: string | null;
  count: number;
  buyCount: number;
  buyAmount: number;
  sellCount: number;
  sellAmount: number;
  disclosuresFixed: boolean;
}) {
  const bigNumberSx = { fontWeight: 700, lineHeight: 1.1, fontSize: { xs: "1.5rem", sm: "1.75rem" } } as const;
  return (
    <Card
      variant="outlined"
      sx={{ mb: 4, borderTop: 2, borderBottom: 2, borderLeft: 0, borderRight: 0, borderColor: "primary.main", borderRadius: 0 }}
    >
      <Inner href={href}>
        {/* 「今日の〜」だと日付をまたいで見たときに実態とずれるため、データの開示日を主語にする。 */}
        <Typography variant="overline" sx={{ color: "brand.blue" }}>
          {formatDate(date)}の大口取引
        </Typography>
        <Box sx={{ mt: 1, display: "flex", alignItems: "baseline", columnGap: 1 }}>
          <Typography variant="h3" component="span" sx={{ ...bigNumberSx, color: "primary.main" }}>
            {count}
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>件の開示</Typography>
        </Box>
        {/* 合計金額だけでは買い優勢か売り優勢か分からないため、買い・売りの件数と金額を分けて出す。
            合計値なので小数第1位は読む意味が薄く、整数に丸めて桁を見やすくする。 */}
        <Box sx={{ mt: 1, display: "flex", flexWrap: "wrap", columnGap: 3, rowGap: 0.5 }}>
          <Box sx={{ display: "flex", alignItems: "baseline", columnGap: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 700, color: "primary.main" }}>買い</Typography>
            <Typography variant="h3" component="span" sx={{ ...bigNumberSx, color: "primary.main" }}>
              {formatDealAmount(Math.round(buyAmount))}
            </Typography>
            <Typography variant="body2" sx={{ color: "text.secondary" }}>{buyCount}件</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "baseline", columnGap: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 700, color: "error.main" }}>売り</Typography>
            <Typography variant="h3" component="span" sx={{ ...bigNumberSx, color: "error.main" }}>
              {formatDealAmount(Math.round(sellAmount))}
            </Typography>
            <Typography variant="body2" sx={{ color: "text.secondary" }}>{sellCount}件</Typography>
          </Box>
        </Box>
        {href && (
          <Typography variant="overline" sx={{ display: "block", mt: 1, color: "brand.blue" }}>
            この日の開示をすべて見る ›
          </Typography>
        )}
        {/* デイトレ層向けの鮮度表示: 何時に今日の開示が載るのかをTOPで明示する（edinet_blog.ymlの毎時実行に対応） */}
        <Typography variant="caption" sx={{ display: "block", mt: 0.5, color: "text.secondary" }}>
          {disclosuresFixed
            ? "この日の開示は確定（EDINETの受付は17時15分で終了）"
            : "受付中。新着開示は平日9時〜19時ごろに毎時自動更新"}
        </Typography>
      </Inner>
    </Card>
  );
}

// 取引日ページを公開していない日はカード全体をクリックさせない（404へ飛ばさない）。
function Inner({ href, children }: { href: string | null; children: React.ReactNode }) {
  if (!href) return <Box sx={{ py: 2 }}>{children}</Box>;
  return (
    <CardActionArea component={Link} href={href} sx={{ py: 2, px: { xs: 0, sm: 0 } }}>
      {children}
    </CardActionArea>
  );
}
