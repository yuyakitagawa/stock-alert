"use client";
import { signalStyle } from "@/lib/signals";
import type { Recommend } from "@/lib/types";
import { useLang } from "@/contexts/LanguageContext";
import { SIGNAL_LABELS } from "@/lib/i18n";

interface Props {
  value: Recommend;
  size?: "sm" | "md";
}

export default function RecommendBadge({ value, size = "sm" }: Props) {
  const { lang } = useLang();
  const s = signalStyle(value);
  const pad = size === "md" ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-xs";
  const label = SIGNAL_LABELS[value]?.[lang] ?? value;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-bold border ${pad} ${s.bg} ${s.text} ${s.border}`}
      suppressHydrationWarning
    >
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${s.dot}`} />
      <span suppressHydrationWarning>{label}</span>
    </span>
  );
}
