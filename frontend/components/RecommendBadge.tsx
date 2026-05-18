import { signalStyle } from "@/lib/signals";
import type { Recommend } from "@/lib/types";

interface Props {
  value: Recommend;
  size?: "sm" | "md";
}

export default function RecommendBadge({ value, size = "sm" }: Props) {
  const s = signalStyle(value);
  const pad = size === "md" ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-xs";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-bold border ${pad} ${s.bg} ${s.text} ${s.border}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${s.dot}`} />
      {value}
    </span>
  );
}
