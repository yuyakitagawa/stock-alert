"use client";

import Chip from "@mui/material/Chip";
import Tooltip from "@mui/material/Tooltip";
import Box from "@mui/material/Box";
import type { DealType } from "@/types/article";
import { DEAL_TYPE_DESCRIPTIONS, DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import type { Locale } from "@/lib/i18n";

const COLOR_MAP: Record<DealType, { dot: string; text: string }> = {
  個人: { dot: "#d97706", text: "#92400e" },
  創業家の資産管理会社: { dot: "#f97316", text: "#c2410c" },
  "公益/一般財団法人": { dot: "#0d9488", text: "#0f766e" },
  プライムブローカー: { dot: "#6366f1", text: "#4338ca" },
  アクティビスト: { dot: "#e11d48", text: "#be123c" },
  VC: { dot: "#ec4899", text: "#db2777" },
  "PE・メザニンファンド": { dot: "#c026d3", text: "#a21caf" },
  独立系ブティックAM: { dot: "#3b82f6", text: "#2563eb" },
  国内アセットマネジメント: { dot: "#2563eb", text: "#1d4ed8" },
  外資系伝統運用会社: { dot: "#7c3aed", text: "#6d28d9" },
  日系証券銀行: { dot: "#0284c7", text: "#0369a1" },
  事業会社: { dot: "#059669", text: "#047857" },
  その他: { dot: "#9ca3af", text: "rgba(32, 29, 26, 0.6)" },
};

export default function DealTypeBadge({
  dealType,
  locale = "ja",
}: {
  dealType: DealType;
  locale?: Locale;
}) {
  if (!dealType) return null;
  const colors = COLOR_MAP[dealType] ?? COLOR_MAP.その他;
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
          color: colors.text,
          fontSize: "0.6875rem",
          fontWeight: 700,
          letterSpacing: "0.08em",
          "& .MuiChip-label": { px: 0.75, py: 0.25 },
        }}
      />
    </Tooltip>
  );
}
