"use client";

import Chip from "@mui/material/Chip";
import Tooltip from "@mui/material/Tooltip";
import Box from "@mui/material/Box";
import type { DealType } from "@/types/article";
import { DEAL_TYPE_COLORS, DEAL_TYPE_DESCRIPTIONS, DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import type { Locale } from "@/lib/i18n";

export default function DealTypeBadge({
  dealType,
  locale = "ja",
  onDark = false,
}: {
  dealType: DealType;
  locale?: Locale;
  // 注目カードなどダーク地に載せる場合。DEAL_TYPE_COLORS.textはライト背景用の濃色で
  // ダーク地ではコントラストが出ないため、文字は白・分類色はドットのみで示す。
  onDark?: boolean;
}) {
  if (!dealType) return null;
  const colors = DEAL_TYPE_COLORS[dealType] ?? DEAL_TYPE_COLORS.その他;
  const label = locale === "en" ? DEAL_TYPE_EN[dealType].label : dealType;
  const title = locale === "en" ? DEAL_TYPE_EN[dealType].description : DEAL_TYPE_DESCRIPTIONS[dealType];

  return (
    <Tooltip title={title}>
      <Chip
        size="small"
        variant="outlined"
        icon={<Box component="span" sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: colors.dot, ml: "6px !important" }} />}
        label={label}
        sx={{
          height: "auto",
          borderColor: "transparent",
          color: onDark ? "rgba(255, 255, 255, 0.92)" : colors.text,
          fontSize: "0.6875rem",
          fontWeight: 700,
          letterSpacing: "0.08em",
          "& .MuiChip-label": { px: 0.75, py: 0.25 },
        }}
      />
    </Tooltip>
  );
}
