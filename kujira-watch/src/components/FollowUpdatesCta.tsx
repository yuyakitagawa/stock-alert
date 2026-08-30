import { X_FOLLOW_URL, X_SCREEN_NAME } from "@/lib/site";

// 投資家ページ・銘柄ページに置く「この対象の更新だけ追う」導線。
// アカウント機能を持たない方針のため、フォローの受け皿はRSS（対象単位）と
// 公式X（サイト全体）の2つ。ページごとのRSSは /investors/[filer]/feed.xml・
// /stocks/[code]/feed.xml が返す（EDINET開示そのものが源なので、記事化されなかった
// 小さな開示も取りこぼさない）。
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
        大きな動きは公式X（@{X_SCREEN_NAME}）でも毎日お知らせしています。
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <a
          href={feedUrl}
          className="rounded-md border border-brand-blue px-3 py-1.5 text-xs font-bold text-brand-blue hover:bg-brand-blue hover:text-white"
        >
          RSSで新着を受け取る
        </a>
        <a
          href={X_FOLLOW_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-md border border-rule px-3 py-1.5 text-xs font-bold text-ink-secondary hover:border-brand-blue hover:text-brand-blue"
        >
          Xでフォローする
        </a>
      </div>
    </section>
  );
}
