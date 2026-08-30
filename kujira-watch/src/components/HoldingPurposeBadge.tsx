"use client";

import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Tooltip from "@mui/material/Tooltip";
import { PURPOSE_COLORS, PURPOSE_DESCRIPTIONS, type HoldingPurpose } from "@/lib/disclosures";

// 大量保有報告書の「保有目的」欄を5区分に寄せたバッジ。
// 同じ「5%取得」でも、純投資と重要提案行為等では読み手にとっての意味がまるで違う。
// 分類はあくまで自由記述からの機械判定なので、原文はバッジの隣に併記する側で出す。
export default function HoldingPurposeBadge({ purpose }: { purpose: HoldingPurpose }) {
  const color = PURPOSE_COLORS[purpose];
  return (
    <Tooltip title={PURPOSE_DESCRIPTIONS[purpose]}>
      <Chip
        size="small"
        variant="outlined"
        icon={
          <Box
            component="span"
            sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: color, ml: "6px !important" }}
          />
        }
        label={purpose}
        sx={{
          height: "auto",
          borderColor: "transparent",
          bgcolor: `${color}14`,
          color,
          fontSize: "0.6875rem",
          fontWeight: 700,
          letterSpacing: "0.08em",
          "& .MuiChip-label": { px: 0.75, py: 0.25 },
        }}
      />
    </Tooltip>
  );
}
