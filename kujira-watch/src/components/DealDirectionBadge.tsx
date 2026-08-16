"use client";

import Chip from "@mui/material/Chip";
import Tooltip from "@mui/material/Tooltip";
import Box from "@mui/material/Box";
import { isSellArticle } from "@/lib/format";
import { UI, type Locale } from "@/lib/i18n";

// 買い方向の記事にはバッジを出さない（従来通りの見た目を維持し、売り方向のみ目立たせる）。
export default function DealDirectionBadge({
  tags,
  locale = "ja",
  onDark = false,
}: {
  tags?: string;
  locale?: Locale;
  // ダーク地ではerror.mainの文字が沈むため、文字は白・売りの赤はドットのみで示す。
  onDark?: boolean;
}) {
  if (!isSellArticle(tags)) return null;
  const t = UI[locale];
  return (
    <Tooltip title={t.sellBadgeTitle}>
      <Chip
        size="small"
        variant="outlined"
        icon={<Box component="span" sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: "error.dark", ml: "6px !important" }} />}
        label={t.sellBadge}
        sx={{
          height: "auto",
          borderColor: "transparent",
          color: onDark ? "rgba(255, 255, 255, 0.92)" : "error.main",
          fontSize: "0.6875rem",
          fontWeight: 700,
          letterSpacing: "0.08em",
          "& .MuiChip-label": { px: 0.75, py: 0.25 },
        }}
      />
    </Tooltip>
  );
}
