import Link from "next/link";

export default function DealDateHeading({ date, label }: { date: string; label: string }) {
  return (
    <div className="mb-5">
      <h2 className="flex items-center gap-3 text-base font-bold text-brand-navy">
        {label}
        <span aria-hidden className="h-px flex-1 bg-rule" />
      </h2>
      <Link
        href={`/date/${date}`}
        className="mt-1 inline-block text-xs text-brand-blue transition-colors hover:text-brand-navy hover:underline"
      >
        この日の記事を見る ›
      </Link>
    </div>
  );
}
