"use client";

import { useEffect, useState } from "react";
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
    <p className="font-mono text-[11px] tracking-wider text-foreground/40 tabular-nums" aria-hidden={count === null}>
      {t.totalVisitsLabel}
      {count !== null ? count.toLocaleString(locale === "en" ? "en-US" : "ja-JP") : "…"}
    </p>
  );
}
