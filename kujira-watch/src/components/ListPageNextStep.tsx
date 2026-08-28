import Link from "next/link";

// 一覧ページ（カテゴリ・日付）の末尾に置く「次に見るページ」。
//
// なぜ必要か（2026-08-27のGA4実測、28日間）:
//   カテゴリページは入口10セッションで直帰100%・滞在0秒、日付ページも入口の直帰100%。
//   どちらも記事一覧を出して終わりで、末尾に次へ進む導線が1つも無く、来た人を全員失っていた。
export default function ListPageNextStep({
  links,
}: {
  links: { href: string; label: string }[];
}) {
  if (links.length === 0) return null;
  return (
    <nav
      aria-label="次に見るページ"
      className="mt-10 border-t border-rule pt-4"
    >
      <p className="mb-3 text-xs font-bold text-foreground/50">次に見る</p>
      <div className="flex flex-wrap gap-2 text-sm">
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className="rounded border border-rule px-3 py-1.5 text-foreground/70 hover:border-brand-blue hover:text-brand-blue"
          >
            {link.label} ›
          </Link>
        ))}
      </div>
    </nav>
  );
}
