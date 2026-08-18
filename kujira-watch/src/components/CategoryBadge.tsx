"use client";

import Link from "next/link";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import type { DealType } from "@/types/article";
import { DEAL_TYPE_COLORS, DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import type { Locale } from "@/lib/i18n";

export default function CategoryBadge({
  dealType,
  locale = "ja",
}: {
  dealType: DealType | undefined;
  locale?: Locale;
}) {
  if (!dealType) return null;
  const label = locale === "en" ? DEAL_TYPE_EN[dealType].label : dealType;
  const href =
    locale === "en"
      ? `/en/category/${DEAL_TYPE_EN[dealType].slug}`
      : `/category/${encodeURIComponent(dealType)}`;
  // 一覧のドット版(DealTypeBadge)と同じ分類色ドットを付ける。同じ「分類」を指す
  // バッジなのに、一覧=色ドット・記事詳細=枠付きChipで別物に見えていた。
  // 枠とリンクは記事詳細でカテゴリ一覧への導線であることを示すため残す。
  const dotColor = (DEAL_TYPE_COLORS[dealType] ?? DEAL_TYPE_COLORS.その他).dot;
  return (
    <Chip
      component={Link}
      href={href}
      clickable
      size="small"
      variant="outlined"
      icon={
        <Box
          component="span"
          sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: dotColor, ml: "8px !important" }}
        />
      }
      label={label}
      sx={{
        fontSize: "0.6875rem",
        fontWeight: 700,
        letterSpacing: "0.08em",
        color: "primary.main",
        borderColor: "rgba(22, 33, 58, 0.4)",
        "&:hover": { color: "brand.gold", borderColor: "brand.gold" },
      }}
    />
  );
}
