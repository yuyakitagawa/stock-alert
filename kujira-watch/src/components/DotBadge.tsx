"use client";

import Link from "next/link";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Tooltip from "@mui/material/Tooltip";
import type { SxProps, Theme } from "@mui/material/styles";

// サイト共通のバッジプリミティブ。「色ドット＋短いラベル」という形をここ1箇所で定義する。
// 分類(DealTypeBadge)・売買方向(DealDirectionBadge)・保有目的(HoldingPurposeBadge)・
// カテゴリ導線(CategoryBadge)がそれぞれ同じsxを複製しており、片方だけ直すと見た目が
// ズレていたため統合した。書式(サイズ・字間・太さ)は theme.ts の MuiChip 既定に置く。
export default function DotBadge({
  label,
  dotColor,
  color,
  tint = false,
  bordered = false,
  tooltip,
  href,
  sx,
}: {
  label: string;
  /** 左のドットの色。分類そのものを表す色はドットだけが担う。 */
  dotColor: string;
  /** ラベル文字の色。 */
  color: string;
  /** ドット色の薄い面をラベル背面に敷く（保有目的バッジ）。 */
  tint?: boolean;
  /** 枠線を出す（リンクとして押せることを示す場合）。 */
  bordered?: boolean;
  tooltip?: string;
  /** 指定するとリンクChipになる。 */
  href?: string;
  sx?: SxProps<Theme>;
}) {
  const linkProps = href ? ({ component: Link, href, clickable: true } as const) : {};
  const chip = (
    <Chip
      {...linkProps}
      icon={
        <Box
          component="span"
          sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: dotColor, ml: "6px !important" }}
        />
      }
      label={label}
      sx={[
        {
          height: "auto",
          color,
          borderColor: bordered ? "rgba(22, 33, 58, 0.4)" : "transparent",
          ...(tint ? { bgcolor: `${dotColor}14` } : null),
          "& .MuiChip-label": { px: 0.75, py: 0.25 },
        },
        ...(Array.isArray(sx) ? sx : [sx]),
      ]}
    />
  );
  return tooltip ? <Tooltip title={tooltip}>{chip}</Tooltip> : chip;
}
