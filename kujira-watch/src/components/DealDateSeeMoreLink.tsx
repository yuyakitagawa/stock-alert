import ActionButton from "./ActionButton";

// その日の記事カードを見せた後の導線として、次の日付見出しの直前（右寄せ）に置く。
export default function DealDateSeeMoreLink({ date }: { date: string }) {
  return (
    <div className="mt-4 text-right">
      <ActionButton href={`/date/${date}`}>この日の記事を見る</ActionButton>
    </div>
  );
}
