import type { Metadata } from "next";
import Link from "next/link";
import AdUnit from "@/components/AdUnit";
import ActionButton from "@/components/ActionButton";
import BuybackImpactCalculator from "@/components/tools/BuybackImpactCalculator";
import { chapterById, chapterNumber, chapterPath } from "@/lib/textbook";
import { toolBySlug, toolPath } from "@/lib/tools";
import { SITE_NAME, SITE_URL } from "@/lib/site";

const tool = toolBySlug("buyback-impact")!;
const url = `${SITE_URL}${toolPath(tool.slug)}`;

export const metadata: Metadata = {
  title: tool.title,
  description: tool.description,
  alternates: { canonical: url },
  openGraph: { title: tool.title, description: tool.description, url },
};

export default function BuybackImpactPage() {
  const chapter = chapterById(tool.chapterId);
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
        { "@type": "ListItem", position: 2, name: "計算ツール", item: `${SITE_URL}/tools` },
        { "@type": "ListItem", position: 3, name: tool.title, item: url },
      ],
    },
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      name: tool.title,
      description: tool.description,
      url,
      applicationCategory: "FinanceApplication",
      operatingSystem: "All",
      inLanguage: "ja",
      isAccessibleForFree: true,
      offers: { "@type": "Offer", price: "0", priceCurrency: "JPY" },
      provider: { "@type": "Organization", name: SITE_NAME, url: SITE_URL },
    },
  ];

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-ink-tertiary">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <Link href="/tools" className="hover:text-brand-blue">計算ツール</Link>
        {" / "}
        <span className="text-ink-secondary">{tool.title}</span>
      </nav>

      <h1 className="mb-2 text-2xl font-bold text-brand-navy sm:text-3xl">{tool.title}</h1>
      <p className="mb-8 text-sm leading-relaxed text-ink-secondary">
        自社株買いの決定開示に載っている取得枠と、発行済株式総数・株価を入れると、
        発行済株式に対する比率、時価総額に対する比率、EPSの押し上げ率、1日平均出来高の何日ぶんかが出ます。
        「何億円か」ではなく「発行済の何%か」で比べるための道具です。
      </p>

      <BuybackImpactCalculator />

      <section className="mb-8">
        <h2 className="mb-3 text-xl font-bold text-brand-navy">計算式</h2>
        <ul className="list-disc pl-5">
          <li className="mb-2 text-sm leading-relaxed text-ink-secondary">
            <strong className="font-bold text-ink">実際に取得しうる株数</strong> = min（取得価額の上限 ÷ 株価、取得株数の上限）。
            決議されるのは2つの上限で、先に当たった方で実際の取得株数が決まります。
          </li>
          <li className="mb-2 text-sm leading-relaxed text-ink-secondary">
            <strong className="font-bold text-ink">発行済株式に対する比率</strong> = 実際に取得しうる株数 ÷ 発行済株式総数（自己株式を除く）。
          </li>
          <li className="mb-2 text-sm leading-relaxed text-ink-secondary">
            <strong className="font-bold text-ink">EPS押し上げ率</strong> = 1 ÷（1 − 発行済比率）− 1。
            1株当たり利益の計算では自己株式が分母から除かれるため、利益が変わらなくても比率のぶんEPSが上がります。
            5%の取得なら約5.3%、10%なら約11.1%です。
          </li>
          <li className="mb-2 text-sm leading-relaxed text-ink-secondary">
            <strong className="font-bold text-ink">出来高の何日ぶんか</strong> = 実際に取得しうる株数 ÷ 1日平均出来高。
            市場買付の場合に、需要としてどれだけの大きさかを測る目安になります。
          </li>
        </ul>
        <p className="mb-0 text-xs leading-relaxed text-ink-tertiary">
          取締役会が決めるのは上限であり、枠を使い切らずに取得期間が終わることもあります。
          ここで出るのは枠を使い切った場合の最大値で、実際の進捗は毎月の「自己株式の取得状況に関するお知らせ」で開示されます。
          投資判断はご自身の責任でお願いします。
        </p>
      </section>

      {chapter && (
        <section className="mb-10 rounded-md border border-rule bg-section-tint p-4">
          <h2 className="mb-2 text-base font-bold text-brand-navy">この数字の意味を読む</h2>
          <p className="mb-3 text-sm leading-relaxed text-ink-secondary">
            取得枠・消却・取得方法の読み方は、教科書の第{chapterNumber(chapter.id)}章「{chapter.title}」で解説しています。
          </p>
          <div className="flex flex-wrap gap-3">
            <ActionButton href={chapterPath(chapter.id)}>{chapter.title}を読む</ActionButton>
            <ActionButton href="/buybacks">直近の自社株買いを見る</ActionButton>
          </div>
        </section>
      )}

      <AdUnit placement="bottom" />
    </article>
  );
}
