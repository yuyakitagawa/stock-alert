"use client";

import Link from "next/link";
import Button from "@mui/material/Button";
import type { ButtonProps } from "@mui/material/Button";

// サイト共通のCTAボタン。「この日の記事を見る」「前後の月」「共有」など、
// 本文中のテキストリンクではない“押す導線”はすべてこのボタンに統一する
// （見た目・リップル・タップ領域をページ間で揃えるため）。
// 配色・角丸・フォントは theme.ts の MuiButton で一元管理している。
// Server Componentから component={Link}（関数）を渡すことはできないため、
// クライアント境界をこのファイルに置く（呼び出し側はサーバー/クライアントどちらでも可）。
export default function ActionButton({
  href,
  children,
  external = false,
  variant,
  selected = false,
  fullWidth,
}: {
  href: string;
  children: React.ReactNode;
  // 外部サイトへのリンク（別タブで開く）。
  external?: boolean;
  variant?: ButtonProps["variant"];
  // フィルターの選択状態など、押下中であることを示したいとき。
  selected?: boolean;
  fullWidth?: boolean;
}) {
  const resolvedVariant = variant ?? (selected ? "contained" : "outlined");

  if (external) {
    return (
      <Button
        component="a"
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        variant={resolvedVariant}
        fullWidth={fullWidth}
      >
        {children}
      </Button>
    );
  }

  return (
    <Button
      component={Link}
      href={href}
      variant={resolvedVariant}
      fullWidth={fullWidth}
      aria-current={selected ? "page" : undefined}
    >
      {children}
    </Button>
  );
}
