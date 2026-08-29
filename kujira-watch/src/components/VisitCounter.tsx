"use client";

import { useEffect, useState } from "react";
import Typography from "@mui/material/Typography";
import { UI } from "@/lib/i18n";

// increment=false は現在値の表示のみ（加算しない）。ヘッダーで既に加算しているページ内の
// 2か所目（ハンバーガーメニュー）で使う。
export default function VisitCounter({
  increment = true,
}: {
  increment?: boolean;
}) {
  const [count, setCount] = useState<number | null>(null);
  const t = UI;

  useEffect(() => {
    fetch("/api/counter", { method: increment ? "POST" : "GET" })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && typeof data.count === "number") {
          setCount(data.count);
        }
      })
      .catch(() => {
        // カウンター取得に失敗しても表示自体は諦める（サイト閲覧を妨げない）
      });
  }, [increment]);

  return (
    <Typography
      variant="caption"
      component="p"
      aria-hidden={count === null}
      sx={{ fontFamily: "var(--font-geist-mono)", letterSpacing: "0.05em", color: "text.disabled", fontVariantNumeric: "tabular-nums" }}
    >
      {t.totalVisitsLabel}
      {count !== null ? count.toLocaleString("ja-JP") : "…"}
    </Typography>
  );
}
