"use client";

import { useEffect, useState } from "react";
import Typography from "@mui/material/Typography";
import { UI, type Locale } from "@/lib/i18n";

export default function VisitCounter({ locale = "ja" }: { locale?: Locale }) {
  const [count, setCount] = useState<number | null>(null);
  const t = UI[locale];

  useEffect(() => {
    fetch("/api/counter", { method: "POST" })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && typeof data.count === "number") {
          setCount(data.count);
        }
      })
      .catch(() => {
        // カウンター取得に失敗しても表示自体は諦める（サイト閲覧を妨げない）
      });
  }, []);

  return (
    <Typography
      variant="caption"
      component="p"
      aria-hidden={count === null}
      sx={{ fontFamily: "var(--font-geist-mono)", letterSpacing: "0.05em", color: "text.disabled", fontVariantNumeric: "tabular-nums" }}
    >
      {t.totalVisitsLabel}
      {count !== null ? count.toLocaleString(locale === "en" ? "en-US" : "ja-JP") : "…"}
    </Typography>
  );
}
