"use client";

import Link from "next/link";
import Chip from "@mui/material/Chip";
import type { DealType } from "@/types/article";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
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
  return (
    <Chip
      component={Link}
      href={href}
      clickable
      size="small"
      variant="outlined"
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
