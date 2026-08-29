import ActionButton from "./ActionButton";

// その日の記事カードを見せた後の導線として、次の日付見出しの直前（右寄せ）に置く。
// 開示が少ない日の取引日ページは公開していない（404）ので、その日は何も出さない
// （公開判定は lib/publishedPages.ts）。
export default function DealDateSeeMoreLink({ href }: { href: string | null }) {
  if (!href) return null;
  return (
    <div className="mt-4 text-right">
      <ActionButton href={href}>この日の記事を見る</ActionButton>
    </div>
  );
}
