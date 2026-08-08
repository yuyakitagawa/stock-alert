import { isSellArticle } from "@/lib/format";

// 買い方向の記事にはバッジを出さない（従来通りの見た目を維持し、売り方向のみ目立たせる）。
export default function DealDirectionBadge({ tags }: { tags?: string }) {
  if (!isSellArticle(tags)) return null;
  return (
    <span
      title="保有比率が減少した売り方向（譲渡・売却等）の開示です"
      className="kicker inline-flex items-center gap-1.5 text-rose-700"
    >
      <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full bg-rose-600" />
      売り
    </span>
  );
}
