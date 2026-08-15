"use client";

import Chip from "@mui/material/Chip";
import Tooltip from "@mui/material/Tooltip";
import { UI, type Locale } from "@/lib/i18n";

function starsLabel(stars: number): string {
  return "★".repeat(stars) + "☆".repeat(5 - stars);
}

export function scoreToStars(score: number): number {
  if (score >= 90) return 5;
  if (score >= 75) return 4;
  if (score >= 55) return 3;
  if (score >= 35) return 2;
  return 1;
}

// カード・一覧用のコンパクトなクジラ注目度バッジ。売り記事・未算出の記事はscore未設定のため
// 何も表示しない（DealDirectionBadgeと同じ「無ければ出さない」方針）。
export default function AttentionScoreBadge({
  score,
  locale = "ja",
}: {
  score?: number;
  locale?: Locale;
}) {
  if (score == null) return null;
  const t = UI[locale];
  const stars = scoreToStars(score);
  return (
    <Tooltip title={t.attentionScoreTooltip}>
      <Chip
        size="small"
        variant="outlined"
        label={`${t.attentionScoreLabel} ${score} ${starsLabel(stars)}`}
        sx={{
          height: "auto",
          borderColor: "transparent",
          bgcolor: "rgba(184, 134, 58, 0.12)",
          color: "brand.gold",
          fontSize: "0.6875rem",
          fontWeight: 700,
          letterSpacing: "0.04em",
          "& .MuiChip-label": { px: 0.75, py: 0.25 },
        }}
      />
    </Tooltip>
  );
}
