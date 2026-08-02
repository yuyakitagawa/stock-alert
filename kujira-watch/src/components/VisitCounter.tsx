"use client";

import { useEffect, useState } from "react";

export default function VisitCounter() {
  const [count, setCount] = useState<number | null>(null);

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

  if (count === null) return null;

  return (
    <p className="mt-3 font-mono text-[11px] tracking-wider text-gray-400">
      累計訪問数: {count.toLocaleString("ja-JP")}
    </p>
  );
}
