import type { Metadata } from "next";
import Link from "next/link";
import Box from "@mui/material/Box";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import AdUnit from "@/components/AdUnit";
import ArticleCard from "@/components/ArticleCard";
import FaqAccordionList from "@/components/FaqAccordionList";
import {
  formatAmountOku,
  getMonthlyBuybackCounts,
  getRecentBuybackDecisions,
  type BuybackDecision,
} from "@/lib/buybacks";
import type { FaqItem } from "@/lib/faqData";
import { formatDate } from "@/lib/format";
import { getArticleList } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";

export const revalidate = 3600;

// 集計窓（/activists・/trendingと同じ30日）。
const WINDOW_DAYS = 30;
// ランキングの表示件数。上限金額は大型株、発行済比率は小型株が上位に来やすく、
// 2つ並べることで「規模」と「需給インパクト」の両面から注目銘柄を拾える。
const RANK_LIMIT = 15;
// 最新一覧の描画上限（30日で決定は100件前後。超えた分は件数を明記する）。
const LIST_LIMIT = 100;

const url = `${SITE_URL}/buybacks`;
const title = "自社株買い注目銘柄";

export const metadata: Metadata = {
  title,
  description:
    "上場企業が取締役会で決議した自社株買い（自己株式取得）の取得枠を、TDnet適時開示の原文PDFから毎日抽出。直近30日の決定を発行済株式比率・上限金額の大きい順にランキングし、取得期間・方法・消却の有無つきで一覧します。",
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

function RankingList({ items, valueOf }: { items: BuybackDecision[]; valueOf: (d: BuybackDecision) => string }) {
  return (
    <ol className="divide-y divide-rule border-y border-rule">
      {items.map((d, i) => (
        <li key={`${d.code}-${d.disclosedAt}`} className="flex items-baseline gap-3 py-2.5">
          <span className="w-6 shrink-0 text-right text-sm font-bold text-foreground/40">{i + 1}</span>
          <div className="min-w-0 flex-1">
            <Link href={`/stocks/${d.code}`} className="font-medium text-brand-blue hover:underline">
              {d.stockName}（{d.code}）
            </Link>
            <span className="ml-2 text-xs text-foreground/50">{formatDate(d.disclosedAt.slice(0, 10))}</span>
            {d.willCancel && <span className="ml-2 rounded bg-brand-navy/10 px-1.5 py-0.5 text-[10px] font-bold text-brand-navy">消却</span>}
          </div>
          <span className="shrink-0 text-sm font-bold text-brand-navy">{valueOf(d)}</span>
        </li>
      ))}
    </ol>
  );
}

export default async function BuybacksPage() {
  const [decisions, monthly, { contents: articles }] = await Promise.all([
    getRecentBuybackDecisions(WINDOW_DAYS).catch(() => [] as BuybackDecision[]),
    getMonthlyBuybackCounts().catch(() => []),
    getArticleList({ dealType: "自社株買い", limit: 6 }).catch(() => ({ contents: [] })),
  ]);

  const totalYen = decisions.reduce((s, d) => s + (d.maxAmountYen ?? 0), 0);
  const byAmount = [...decisions]
    .filter((d) => d.maxAmountYen !== null)
    .sort((a, b) => (b.maxAmountYen ?? 0) - (a.maxAmountYen ?? 0))
    .slice(0, RANK_LIMIT);
  const byRatio = [...decisions]
    .filter((d) => d.ratioPct !== null)
    .sort((a, b) => (b.ratioPct ?? 0) - (a.ratioPct ?? 0))
    .slice(0, RANK_LIMIT);
  const largest = byAmount[0];
  const cancelCount = decisions.filter((d) => d.willCancel).length;
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

  const tile = (label: string, value: string, sub?: string) => (
    <div className="rounded-lg border border-rule bg-surface/60 px-4 py-3">
      <div className="text-xs text-foreground/50">{label}</div>
      <div className="mt-1 text-xl font-bold text-brand-navy">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-foreground/60">{sub}</div>}
    </div>
  );

  return (
    <div>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }} />
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">{title}</span>
      </nav>

      <div className="mb-6">
        <h1 className="text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>
        <p className="mt-2 text-sm leading-relaxed text-foreground/70">
          上場企業が取締役会で決議した自社株買い（自己株式取得）の取得枠を、TDnet適時開示の原文PDFから
          {SITE_NAME}のシステムが毎日抽出しています。会社自身が自社株を買う「最大のクジラ」の動きを、
          直近{WINDOW_DAYS}日の決定について発行済株式比率・上限金額の大きい順に並べました。
          月次の取得状況報告（進捗）は含みません。数字の見方は下部の
          <a href="#faq" className="text-brand-blue hover:underline">よくある質問</a>をご覧ください。
        </p>
      </div>

      <section className="mb-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {tile(`直近${WINDOW_DAYS}日の決定`, `${decisions.length}件`)}
        {tile("上限金額の合計", formatAmountOku(totalYen) ?? "-")}
        {tile(
          "最大の取得枠",
          largest ? formatAmountOku(largest.maxAmountYen) ?? "-" : "-",
          largest ? `${largest.stockName}（${largest.code}）` : undefined
        )}
        {tile("消却を同時決議", `${cancelCount}件`, decisions.length > 0 ? `決定の${Math.round((cancelCount / decisions.length) * 100)}%` : undefined)}
      </section>

      <div className="mb-10 grid grid-cols-1 gap-8 lg:grid-cols-2">
        <section>
          <h2 className="mb-1 text-xl font-bold text-brand-navy">発行済株式比率ランキング</h2>
          <p className="mb-3 text-xs text-foreground/60">
            取得上限が発行済株式総数（自己株式を除く）に占める割合。比率が高いほど1株当たりの押し上げ効果と需給インパクトが大きい。
          </p>
          {byRatio.length === 0 ? (
            <p className="text-sm text-foreground/60">直近{WINDOW_DAYS}日に比率が開示された決定はありません。</p>
          ) : (
            <RankingList items={byRatio} valueOf={(d) => `${d.ratioPct}%`} />
          )}
        </section>
        <section>
          <h2 className="mb-1 text-xl font-bold text-brand-navy">上限金額ランキング</h2>
          <p className="mb-3 text-xs text-foreground/60">
            取得価額の総額の上限。大型株が上位に来やすいため、規模感の把握用。比率ランキングと併せて見る。
          </p>
          {byAmount.length === 0 ? (
            <p className="text-sm text-foreground/60">直近{WINDOW_DAYS}日に上限金額が開示された決定はありません。</p>
          ) : (
            <RankingList items={byAmount} valueOf={(d) => formatAmountOku(d.maxAmountYen) ?? "-"} />
          )}
        </section>
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
          <TableContainer sx={{ borderTop: 1, borderColor: "divider" }}>
            <Table size="small" sx={{ minWidth: 720, "& .MuiTableCell-root": { borderColor: "divider" } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: "text.disabled" }}>開示日</TableCell>
                  <TableCell sx={{ color: "text.disabled" }}>銘柄</TableCell>
                  <TableCell sx={{ color: "text.disabled" }}>取得枠（上限）</TableCell>
                  <TableCell sx={{ color: "text.disabled" }}>方法・期間</TableCell>
                  <TableCell sx={{ color: "text.disabled" }}>原文</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {listed.map((d) => {
                  const period =
                    d.periodFrom && d.periodTo
                      ? d.periodFrom === d.periodTo
                        ? formatDate(d.periodFrom)
                        : `${formatDate(d.periodFrom)}〜${formatDate(d.periodTo)}`
                      : null;
                  return (
                    <TableRow key={`${d.code}-${d.disclosedAt}-${d.title}`}>
                      <TableCell sx={{ whiteSpace: "nowrap", color: "text.secondary" }}>
                        {formatDate(d.disclosedAt.slice(0, 10))}
                      </TableCell>
                      <TableCell>
                        <Link href={`/stocks/${d.code}`} className="text-brand-blue hover:underline">
                          {d.stockName}（{d.code}）
                        </Link>
                        {d.willCancel && <span className="ml-2 text-[10px] font-bold text-brand-navy">消却</span>}
                      </TableCell>
                      <TableCell sx={{ whiteSpace: "nowrap" }}>
                        <Frame d={d} />
                        {d.maxShares !== null && (
                          <span className="block text-xs text-foreground/50">{d.maxShares.toLocaleString("ja-JP")}株</span>
                        )}
                      </TableCell>
                      <TableCell sx={{ color: "text.secondary" }}>
                        <span className="block text-xs">{d.method ?? "-"}</span>
                        {period && <span className="block text-xs text-foreground/60">{period}</span>}
                      </TableCell>
                      <TableCell sx={{ whiteSpace: "nowrap" }}>
                        {d.docUrl && (
                          <a href={d.docUrl} target="_blank" rel="noopener noreferrer" className="text-xs text-brand-blue hover:underline">
                            PDF ↗
                          </a>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </section>

      {monthly.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-1 text-xl font-bold text-brand-navy">月別の決定件数</h2>
          <p className="mb-3 text-xs text-foreground/60">
            取得枠の新設（決定開示）の件数。上限合計は取得枠の抽出を始めた2026年8月中旬以降の月のみ。
          </p>
          <Box sx={{ overflowX: "auto" }}>
            <Table size="small" sx={{ maxWidth: 480, "& .MuiTableCell-root": { borderColor: "divider" } }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: "text.disabled" }}>月</TableCell>
                  <TableCell align="right" sx={{ color: "text.disabled" }}>決定件数</TableCell>
                  <TableCell align="right" sx={{ color: "text.disabled" }}>上限合計</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {monthly.map((m) => (
                  <TableRow key={m.month}>
                    <TableCell sx={{ whiteSpace: "nowrap" }}>
                      {m.month.replace("-", "年")}月{m.isPartial && <span className="ml-1 text-xs text-foreground/50">（集計中）</span>}
                    </TableCell>
                    <TableCell align="right">{m.decisions}件</TableCell>
                    <TableCell align="right" sx={{ color: m.totalYen > 0 ? "text.primary" : "text.disabled" }}>
                      {formatAmountOku(m.totalYen) ?? "-"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        </section>
      )}

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

      <section className="mb-10">
        <h2 className="mb-3 text-xl font-bold text-brand-navy">自社株買いの数字の見方</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-foreground/80">
          <li>
            <strong>比率を先に見る。</strong>
            上限金額は会社の規模に比例するので、需給インパクトは発行済株式比率で比べる。5%を超える枠は大きく、10%超は株主構成を変えるレベル。
          </li>
          <li>
            <strong>期間の長さで本気度を測る。</strong>
            ToSTNeT-3は翌営業日に一括で終わる（大株主の売却の受け皿であることが多い）。1年近い取得期間の市場買付は、株価を見ながら機動的に買う枠。
          </li>
          <li>
            <strong>消却まで決めているか。</strong>
            消却しない自己株式は将来のM&Aや株式報酬で市場に戻りうる。消却の同時決議は株数を恒久的に減らす意思表示。
          </li>
          <li>
            <strong>枠は約束ではない。</strong>
            実際の取得は毎月の取得状況報告で確認する。各銘柄ページの「自社株買い（TDnet適時開示）」に進捗を含む履歴がある。
          </li>
          <li>
            <strong>大株主の動きと合わせて読む。</strong>
            同じ銘柄で<Link href="/trending" className="text-brand-blue hover:underline">大量保有報告書</Link>の売り（比率低下）と自社株買いが同時期に出ていれば、会社が大株主の売却を吸収している構図。
          </li>
        </ul>
      </section>

      <section id="faq" className="mb-8 border-t border-rule pt-4">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">よくある質問</h2>
        <FaqAccordionList faqs={FAQS} />
      </section>

      <AdUnit placement="bottom" />
    </div>
  );
}
