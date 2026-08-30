import { formatDate, toDateAttr } from "@/lib/format";

/**
 * ページの公開日・更新日をWebPageとして宣言する。
 * 記事(Article)以外のページはこれが無いと日付を持たず、AI検索から鮮度不明＝古い情報として
 * 扱われうる。渡す日付はページが実際に反映しているデータの日付にすること
 * （ビルド時刻を入れると、中身が変わっていない日も「更新した」と嘘をつくことになる）。
 */
export function PageDatesJsonLd({
  url,
  date,
  datePublished,
}: {
  url: string;
  date: string;
  datePublished?: string;
}) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebPage",
    "@id": url,
    url,
    datePublished: toDateAttr(datePublished ?? date),
    dateModified: toDateAttr(date),
  };
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }}
    />
  );
}

type Props = {
  /** 反映済みデータの最終日。"2026-08-29" でも ISO 文字列でも可。 */
  date: string;
  /** ページの絶対URL。WebPage構造化データの@id/urlに使う。 */
  url: string;
  /** 表示ラベル。 */
  label?: string;
  /** ページの初出日。省略時はdateと同じ扱いにする。 */
  datePublished?: string;
  className?: string;
};

/**
 * ページの鮮度を人にも機械にも同じ値で見せる。可視の日付をtime要素で機械可読にし、
 * 同じ値をWebPage(dateModified)としても出す。
 */
export default function DataUpdatedAt({
  date,
  url,
  label = "最終更新",
  datePublished,
  className = "",
}: Props) {
  const modified = toDateAttr(date);
  return (
    <>
      <PageDatesJsonLd url={url} date={date} datePublished={datePublished} />
      <p className={`text-xs text-ink-tertiary ${className}`.trim()}>
        {label}: <time dateTime={modified}>{formatDate(modified)}</time>
      </p>
    </>
  );
}
