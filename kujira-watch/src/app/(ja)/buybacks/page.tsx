import type { Metadata } from "next";
import Link from "next/link";
import AdUnit from "@/components/AdUnit";
import ArticleCard from "@/components/ArticleCard";
import FaqAccordionList from "@/components/FaqAccordionList";
import SectorIcon from "@/components/SectorIcon";
import { getCompanyBriefs, type CompanyBrief } from "@/lib/companyInfo";
import { formatAmountOku, getRecentBuybackDecisions, type BuybackDecision } from "@/lib/buybacks";
import type { FaqItem } from "@/lib/faqData";
import { formatDate } from "@/lib/format";
import { getArticleList } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";

export const revalidate = 3600;

// 集計窓（30日。/trendingと/activistsは7日窓）。
const WINDOW_DAYS = 30;
// 最新一覧の描画上限（30日で決定は100件前後。超えた分は件数を明記する）。
const LIST_LIMIT = 100;

const url = `${SITE_URL}/buybacks`;
const title = "自社株買い注目銘柄";

export const metadata: Metadata = {
  title,
  description:
    "上場企業が取締役会で決議した自社株買い（自己株式取得）の取得枠を、TDnet適時開示の原文PDFから毎日抽出。直近30日の決定を開示日の新しい順に、上限金額・発行済株式比率・取得期間・方法・消却の有無つきで一覧します。",
  alternates: { canonical: url },
  openGraph: { title, url },
};

const FAQS: FaqItem[] = [
  {
    category: "自社株買い",
    question: "自社株買いの「上限」とは何ですか？",
    answer:
      "取締役会が決議した「ここまでは買ってよい」という枠（取得する株式の総数・取得価額の総額の上限）です。実際の取得額はこれを下回ることがあり、株価が上がれば枠を使い切る前に株数上限に達することも、取得期間内に枠を消化しないこともあります。進捗は毎月の「自己株式の取得状況に関するお知らせ」で開示されます。",
  },
  {
    category: "自社株買い",
    question: "発行済株式比率が高いと何が違うのですか？",
    answer:
      "発行済株式総数（自己株式を除く）に対する取得上限の割合です。たとえば10%なら、枠を使い切った場合に1株当たり利益（EPS）が計算上およそ1割押し上がり、市場に出回る株数もその分減ります。上限金額が同じでも、時価総額の小さい会社ほど比率は高くなり、需給への影響は大きくなります。",
  },
  {
    category: "自社株買い",
    question: "ToSTNeT-3（立会外買付）とは何ですか？",
    answer:
      "東京証券取引所の自己株式立会外買付取引で、翌営業日の取引開始前（午前8時45分）に前日終値で一括して買い付ける方法です。市場でじわじわ買う通常の市場買付と違い、1日で取得が完了します。大株主が保有株を売却する受け皿として使われることが多く、開示に「主要株主の異動」が併記される場合は売り手が存在するサインです。",
  },
  {
    category: "自社株買い",
    question: "「消却」の有無は何を意味しますか？",
    answer:
      "取得した自己株式を消滅させて発行済株式総数そのものを減らすことです。消却しない自己株式は金庫株として残り、将来の株式報酬やM&Aの対価に再利用される（市場に戻る）可能性があります。消却まで決議している会社は、株数を恒久的に減らす意思がより明確といえます。",
  },
  {
    category: "自社株買い",
    question: "データはどこから取っていますか？更新頻度は？",
    answer:
      "TDnet（適時開示情報閲覧サービス）に公表された「自己株式取得に係る事項の決定に関するお知らせ」等の原文PDFを、当サイトのシステムが毎日読み取り、上限株数・上限金額・発行済比率・取得期間・取得方法を抽出しています。月次の取得状況報告（進捗）は一覧に含めていません。各銘柄ページでは進捗を含む自社株買い開示の履歴を確認できます。",
  },
  // 以下3件は本文にあった「自社株買いの数字の見方」セクションをQ&A形式へ移設したもの
  // （2026-08-23。本文の読み物セクションよりFAQに集約した方が探しやすいため）。
  {
    category: "自社株買い",
    question: "自社株買いの数字はどこから見ればよいですか？",
    answer:
      "まず発行済株式比率から見ます。上限金額は会社の規模に比例するため、需給インパクトは発行済株式比率で比べるのが基本です。5%を超える枠は大きく、10%超は株主構成を変えるレベルです。上限金額は規模感の把握用として、比率と併せて見てください。",
  },
  {
    category: "自社株買い",
    question: "取得期間の長さからは何が読み取れますか？",
    answer:
      "会社の買い方の本気度・スタイルです。ToSTNeT-3は翌営業日に一括で取得が終わる方式で、大株主の売却の受け皿であることが多い一方、1年近い取得期間の市場買付は株価を見ながら機動的に買う枠です。なお枠は取得の約束ではないため、実際の取得は毎月の取得状況報告（各銘柄ページの履歴）で確認してください。",
  },
  {
    category: "自社株買い",
    question: "大量保有報告書と合わせてどう読めばよいですか？",
    answer:
      "同じ銘柄で大量保有報告書の売り（保有比率の低下）と自社株買いが同時期に出ていれば、会社が大株主の売却を吸収している構図が読み取れます。銘柄ランキングや各銘柄ページで大口投資家の動きと突き合わせて確認できます。",
  },
];

function Frame({ d }: { d: BuybackDecision }) {
  const amount = formatAmountOku(d.maxAmountYen);
  return (
    <>
      <span className="font-bold text-brand-navy">{amount ?? "-"}</span>
      {d.ratioPct !== null && <span className="ml-2 text-xs text-foreground/60">発行済の{d.ratioPct}%</span>}
    </>
  );
}

export default async function BuybacksPage() {
  const [decisions, { contents: articles }] = await Promise.all([
    getRecentBuybackDecisions(WINDOW_DAYS).catch(() => [] as BuybackDecision[]),
    getArticleList({ dealType: "自社株買い", limit: 6 }).catch(() => ({ contents: [] })),
  ]);

  // 銘柄の業種アイコン用（会社ロゴは持てないため業種で代替）。決定一覧の銘柄ぶんを一括取得。
  const briefs = await getCompanyBriefs([...new Set(decisions.map((d) => d.code))]).catch(
    () => new Map<string, CompanyBrief>()
  );
  const sectorOf = (code: string) => briefs.get(code)?.sector;

  const listed = decisions.slice(0, LIST_LIMIT);

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "トップ", item: SITE_URL },
      { "@type": "ListItem", position: 2, name: title, item: url },
    ],
  };
  const faqJsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: FAQS.map((f) => ({
      "@type": "Question",
      name: f.question,
      acceptedAnswer: { "@type": "Answer", text: f.answer },
    })),
  };

  return (
    <div>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }} />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">{title}</span>
      </nav>

      <div className="mb-4 sm:mb-6">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          <span className="hidden sm:inline">
            上場企業が決議した自社株買いの取得枠を、TDnet開示の原文PDFから{SITE_NAME}が毎日抽出。
          </span>
          直近{WINDOW_DAYS}日の決定を開示日の新しい順に並べています（数字の見方は
          <a href="#faq" className="text-brand-blue hover:underline">FAQ</a>）。
        </p>
      </div>

      <section className="mb-10">
        <h2 className="mb-1 text-xl font-bold text-brand-navy">最新の自社株買い決定</h2>
        <p className="mb-3 text-xs text-foreground/60">
          開示日の新しい順。
          {decisions.length > LIST_LIMIT && `上位${LIST_LIMIT}件を表示（ほか${decisions.length - LIST_LIMIT}件は各銘柄ページでご確認ください）。`}
        </p>
        {listed.length === 0 ? (
          <p className="text-sm text-foreground/60">直近{WINDOW_DAYS}日の決定はありません。</p>
        ) : (
          /* 以前はMUI Table（minWidth 720px）でスマホは横スクロールが必要だった。
             他の一覧ページと同じ1件1カード（.card-grid-wide）にして全項目を折り返しで収める。 */
          <ul className="card-grid card-grid-wide">
            {listed.map((d) => {
              const period =
                d.periodFrom && d.periodTo
                  ? d.periodFrom === d.periodTo
                    ? formatDate(d.periodFrom)
                    : `${formatDate(d.periodFrom)}〜${formatDate(d.periodTo)}`
                  : null;
              return (
                <li key={`${d.code}-${d.disclosedAt}-${d.title}`} className="card">
                  <div className="flex items-start gap-2">
                    <SectorIcon sector={sectorOf(d.code)} />
                    <div className="min-w-0 grow">
                      <span className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                        <Link href={`/stocks/${d.code}`} className="font-medium text-brand-blue hover:underline">
                          {d.stockName}（{d.code}）
                        </Link>
                        {d.willCancel && (
                          <span className="rounded bg-brand-navy/10 px-1.5 py-0.5 text-[10px] font-bold text-brand-navy">
                            消却
                          </span>
                        )}
                        <span className="text-xs text-foreground/50">{formatDate(d.disclosedAt.slice(0, 10))}</span>
                      </span>
                      <span className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-xs text-foreground/60">
                        <span className="text-sm">
                          <Frame d={d} />
                        </span>
                        {d.maxShares !== null && <span>{d.maxShares.toLocaleString("ja-JP")}株</span>}
                        {d.method && <span>{d.method}</span>}
                        {period && <span>{period}</span>}
                        {d.docUrl && (
                          <a
                            href={d.docUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-brand-blue hover:underline"
                          >
                            原文PDF ↗
                          </a>
                        )}
                      </span>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {articles.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-3 text-xl font-bold text-brand-navy">自社株買いの解説記事</h2>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {articles.map((article) => (
              <ArticleCard key={article.id} article={article} />
            ))}
          </div>
        </section>
      )}

      <section id="faq" className="mb-8 border-t border-rule pt-4">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">よくある質問</h2>
        <FaqAccordionList faqs={FAQS} />
      </section>

      <AdUnit placement="bottom" />
    </div>
  );
}
