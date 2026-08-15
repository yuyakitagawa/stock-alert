"use client";

import { useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { ADSENSE_CLIENT, ADSENSE_SLOTS, type AdPlacement } from "@/lib/adsense";
import { UI, type Locale } from "@/lib/i18n";

declare global {
  interface Window {
    adsbygoogle?: unknown[];
  }
}

// AdSenseの広告枠。掲載位置ごとに別の広告ユニット(スロットID)を使い、
// AdSense管理画面で「記事一覧の途中」と「ページ下部」の収益を分けて見られるようにする。
//
// - infeed: 記事一覧(オートスクロール)の取引日グループの区切り
// - bottom: 記事詳細・銘柄ページ等、コンテンツの末尾
//
// 記事と見分けが付かない置き方はAdSenseのポリシー違反になるため、必ず「広告」ラベルを添える。
export default function AdUnit({
  placement,
  locale = "ja",
}: {
  placement: AdPlacement;
  locale?: Locale;
}) {
  const insRef = useRef<HTMLModElement>(null);
  const slot = ADSENSE_SLOTS[placement];

  useEffect(() => {
    const el = insRef.current;
    // 同じ<ins>を二度pushするとAdSenseが「already have ads in them」で例外を投げる。
    // React StrictModeの二重実行を避けるため、処理済みマーカーが付いた枠はスキップする。
    if (!el || el.dataset.adsbygoogleStatus) return;
    try {
      (window.adsbygoogle = window.adsbygoogle ?? []).push({});
    } catch {
      // 広告ブロッカー等でローダーが読めない場合は無視する（本文の表示は止めない）。
    }
  }, []);

  if (!ADSENSE_CLIENT || !slot) return null;

  return (
    <Box
      component="aside"
      sx={placement === "bottom" ? { mt: 6 } : { mb: 8 }}
    >
      <Typography
        variant="overline"
        component="p"
        sx={{ mb: 0.5, color: "text.disabled", lineHeight: 1.2 }}
      >
        {UI[locale].adLabel}
      </Typography>
      <ins
        ref={insRef}
        className="adsbygoogle"
        style={{ display: "block" }}
        data-ad-client={ADSENSE_CLIENT}
        data-ad-slot={slot}
        data-ad-format="auto"
        data-full-width-responsive="true"
      />
    </Box>
  );
}
