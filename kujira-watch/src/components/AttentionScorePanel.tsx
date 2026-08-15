"use client";

import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import HelpOutlineIcon from "@mui/icons-material/Help";
import { UI, type Locale } from "@/lib/i18n";
import { scoreToStars } from "./AttentionScoreBadge";

// 記事詳細ページ用のクジラ注目度パネル。score未設定(売り記事・旧記事)の場合は何も描画しない。
export default function AttentionScorePanel({
  score,
  reasons,
  locale = "ja",
}: {
  score?: number;
  reasons?: string;
  locale?: Locale;
}) {
  if (score == null) return null;
  const t = UI[locale];
  const stars = scoreToStars(score);
  // 注目度の理由は日本語のみ生成されるため(lib/attention_score.py)、EN版では
  // 他コンテンツと同様に日英混在を避けてスコア・星のみ表示する。
  const reasonList =
    locale === "ja"
      ? reasons
          ?.split(",")
          .map((r) => r.trim())
          .filter(Boolean)
      : undefined;

  return (
    <Box
      sx={{
        mb: 4,
        p: { xs: 2.5, sm: 3 },
        borderRadius: 1,
        border: 1,
        borderColor: "rgba(184, 134, 58, 0.35)",
        bgcolor: "rgba(184, 134, 58, 0.06)",
      }}
    >
      <Stack direction="row" alignItems="center" spacing={0.5} sx={{ mb: 1 }}>
        <Typography variant="overline" sx={{ color: "brand.gold", fontWeight: 700, letterSpacing: "0.08em" }}>
          {t.attentionScoreLabel}
        </Typography>
        <Tooltip title={t.attentionScoreTooltip}>
          <HelpOutlineIcon sx={{ fontSize: 14, color: "text.disabled", cursor: "help" }} />
        </Tooltip>
      </Stack>
      <Stack direction="row" alignItems="baseline" spacing={1.5} sx={{ mb: reasonList?.length ? 1.5 : 0 }}>
        <Typography component="span" sx={{ fontSize: "2rem", fontWeight: 800, color: "brand.gold", lineHeight: 1 }}>
          {score}
        </Typography>
        <Typography component="span" sx={{ fontSize: "1.125rem", color: "brand.gold", letterSpacing: "0.05em" }}>
          {"★".repeat(stars) + "☆".repeat(5 - stars)}
        </Typography>
      </Stack>
      {reasonList && reasonList.length > 0 && (
        <Stack component="ul" sx={{ m: 0, pl: 2.5, listStyle: "disc" }}>
          {reasonList.map((reason) => (
            <Typography key={reason} component="li" variant="body2" sx={{ color: "text.secondary" }}>
              {reason}
            </Typography>
          ))}
        </Stack>
      )}
    </Box>
  );
}
