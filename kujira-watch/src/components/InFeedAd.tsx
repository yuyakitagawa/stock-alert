"use client";

import { useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { ADSENSE_CLIENT, ADSENSE_INFEED_SLOT } from "@/lib/adsense";
import { UI, type Locale } from "@/lib/i18n";

declare global {
  interface Window {
    adsbygoogle?: unknown[];
  }
}

// 記事一覧（オートスクロール）の途中に差し込む広告枠。
// 記事カードと見分けが付かない置き方はAdSenseのポリシー違反になるため、
// 取引日グループの境目にだけ置き、必ず「広告」ラベルを添える。
export default function InFeedAd({ locale = "ja" }: { locale?: Locale }) {
  const insRef = useRef<HTMLModElement>(null);
  const enabled = Boolean(ADSENSE_CLIENT && ADSENSE_INFEED_SLOT);

  useEffect(() => {
    const el = insRef.current;
    // 同じ<ins>を二度pushするとAdSenseが「already have ads in them」で例外を投げる。
    // React StrictModeの二重実行を避けるため、処理済みマーカーが付いた枠はスキップする。
    if (!el || el.dataset.adsbygoogleStatus) return;
    try {
      (window.adsbygoogle = window.adsbygoogle ?? []).push({});
    } catch {
      // 広告ブロッカー等でローダーが読めない場合は無視する（記事一覧の表示は止めない）。
    }
  }, []);

  if (!enabled) return null;

  return (
    <Box component="aside" sx={{ mb: 8 }}>
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
        data-ad-slot={ADSENSE_INFEED_SLOT}
        data-ad-format="auto"
        data-full-width-responsive="true"
      />
    </Box>
  );
}
