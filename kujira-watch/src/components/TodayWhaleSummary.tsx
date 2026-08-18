"use client";

import Link from "next/link";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { formatDate, formatDealAmount } from "@/lib/format";

// TOPの冒頭に置く「今日のクジラ」サマリー。見出しは常に「今日の注目取引」で固定だが、
// 実際の最新開示が何日付なのか・その日に何件いくら動いたのかは記事カードを読むまで
// 分からなかった。開示日と件数・金額を最初に大きく出すことで、毎日更新されている
// ことが一目で分かるようにする（Discover・リピーター向けの鮮度表示）。
export default function TodayWhaleSummary({
  date,
  count,
  amount,
  buyCount,
  sellCount,
}: {
  date: string;
  count: number;
  amount: number;
  buyCount: number;
  sellCount: number;
}) {
  const dateOnly = date.slice(0, 10);
  return (
    <Card
      variant="outlined"
      sx={{ mb: 4, borderTop: 2, borderBottom: 2, borderLeft: 0, borderRight: 0, borderColor: "primary.main", borderRadius: 0 }}
    >
      <CardActionArea component={Link} href={`/date/${dateOnly}`} sx={{ py: 2, px: { xs: 0, sm: 0 } }}>
        <Typography variant="overline" sx={{ color: "brand.blue" }}>
          今日の大口取引・{formatDate(date)}
        </Typography>
        <Box sx={{ mt: 1, display: "flex", flexWrap: "wrap", alignItems: "baseline", columnGap: 1.5, rowGap: 0.5 }}>
          <Typography variant="h3" component="span" sx={{ fontWeight: 700, lineHeight: 1, color: "primary.main" }}>
            {count}
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>件の開示</Typography>
          <Typography variant="h3" component="span" sx={{ fontWeight: 700, lineHeight: 1, color: "primary.main" }}>
            {formatDealAmount(amount)}
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            （買い{buyCount}件・売り{sellCount}件）
          </Typography>
        </Box>
        <Typography variant="overline" sx={{ display: "block", mt: 1, color: "brand.blue" }}>
          この日の開示をすべて見る ›
        </Typography>
        {/* デイトレ層向けの鮮度表示: 何時に今日の開示が載るのかをTOPで明示する（edinet_blog.ymlの毎時実行に対応） */}
        <Typography variant="caption" sx={{ display: "block", mt: 0.5, color: "text.secondary" }}>
          新着開示は平日9時〜19時ごろに毎時自動更新
        </Typography>
      </CardActionArea>
    </Card>
  );
}
