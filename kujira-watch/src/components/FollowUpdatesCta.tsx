// 投資家ページ・銘柄ページに置く「この対象の更新だけ追う」導線。
// アカウント機能を持たない方針のため、受け皿は対象単位のRSSひとつに絞る
// （SNSへの誘導はサイト全体でフッターのみに集約した）。ページごとのRSSは
// /investors/[filer]/feed.xml・/stocks/[code]/feed.xml が返す（EDINET開示
// そのものが源なので、記事化されなかった小さな開示も取りこぼさない）。
export default function FollowUpdatesCta({
  feedUrl,
  targetLabel,
}: {
  feedUrl: string;
  targetLabel: string;
}) {
  return (
    <section className="mb-10 rounded-md border border-rule bg-section-tint p-4">
      <h2 className="text-sm font-bold text-brand-navy">{targetLabel}の更新を追う</h2>
      <p className="mt-1 text-xs leading-relaxed text-ink-tertiary">
        新しい開示が出たときに気づけるよう、RSSリーダー（Feedly・Inoreader等）に登録できます。
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <a
          href={feedUrl}
          className="rounded-md border border-brand-blue px-3 py-1.5 text-xs font-bold text-brand-blue hover:bg-brand-blue hover:text-white"
        >
          RSSで新着を受け取る
        </a>
      </div>
    </section>
  );
}
