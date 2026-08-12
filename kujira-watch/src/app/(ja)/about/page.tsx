import type { Metadata } from "next";
import Link from "next/link";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { getAboutPage } from "@/lib/microcms";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { DEAL_TYPES } from "@/types/article";

const title = "運営者情報・免責事項";
const description = "本サイトの運営方針、データソース、免責事項について。";

export const metadata: Metadata = {
  title,
  description,
  alternates: {
    canonical: `${SITE_URL}/about`,
    languages: { ja: `${SITE_URL}/about`, en: `${SITE_URL}/en/about` },
  },
};

export const revalidate = 60;

export default async function AboutPage() {
  const about = await getAboutPage();

  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{about.heroTitle}</h1>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">大口投資家とは</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          株式や各種金融市場において、市場価格を動かすほどの巨大な資金を運用する投資主体です。
        </p>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm leading-relaxed text-foreground/70">
          <li>
            <strong className="text-brand-navy">機関投資家（国内・海外）</strong>:
            年金基金（GPIFなど）、投資信託、生命保険会社、政府系ファンド
          </li>
          <li>
            <strong className="text-brand-navy">ヘッジファンド</strong>:
            短期〜中長期で絶対収益を狙うファンド（空売りやデリバティブを多用）
          </li>
          <li>
            <strong className="text-brand-navy">アクティビスト（物言う株主）</strong>:
            企業に増配や自社株買い、経営改善を要求するファンド
          </li>
          <li>
            <strong className="text-brand-navy">富裕層・個人大口（クジラ・ウルフ等）</strong>:
            個人枠でありながら億単位で動かす投資家
          </li>
        </ul>
        <p className="mt-3 text-sm leading-relaxed text-foreground/70">
          {SITE_NAME}ではこれらの投資家を13種類に分類しています。詳しくは
          <Link href="#dealtype-glossary" className="text-brand-blue hover:underline">
            用語集
          </Link>
          をご覧ください。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">大口投資家の動きを追う意味</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-foreground/70">
          <li>
            <strong className="text-brand-navy">資金の流れ（トレンド）の把握</strong>:
            株価の持続的な上昇・下落は大口投資家の買増し・売却によって作られます。トレンドに逆らわない
            「順張り」の判断材料として重要です。
          </li>
          <li>
            <strong className="text-brand-navy">銘柄選定のスクリーニング</strong>:
            業績や成長性が大口プロ投資家の調査基準を満たしているかのフィルターになります。
          </li>
        </ul>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">大口投資家の動きとは</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          「大口投資家の動き」とは、機関投資家・アクティビストファンド・創業家の資産管理会社・
          自社株買いなど、相場を動かすほどの資金力を持つ投資主体（クジラ）が、上場企業の株式を
          いつ・どれくらいの規模で売買したかを指します。日本では、株式の5%以上を保有した投資家は
          EDINET（金融庁の開示システム）に大量保有報告書の提出が義務付けられており（5%ルール）、
          保有比率が1%以上増減した場合は変更報告書として再提出されます。{SITE_NAME}はこの公開情報を
          毎日追跡し、銘柄別・投資家分類別に整理して解説しています。直近の動きは
          <Link href="/weekly" className="text-brand-blue hover:underline">
            「今週の動き」ページ
          </Link>
          でまとめて確認できるほか、特定の投資家（ファンド）を軸に保有銘柄の推移を追いたい場合は
          <Link href="/investors" className="text-brand-blue hover:underline">
            「投資家一覧」ページ
          </Link>
          も参照できます。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">このサイトについて</h2>
        <div
          className="prose prose-sm max-w-none text-foreground/70"
          dangerouslySetInnerHTML={{ __html: about.profileBody }}
        />
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">データソースと更新方法</h2>
        <div
          className="prose prose-sm max-w-none text-foreground/70"
          dangerouslySetInnerHTML={{ __html: about.dataSources }}
        />
        {about.methodology && (
          <div
            className="prose prose-sm mt-3 max-w-none text-foreground/70"
            dangerouslySetInnerHTML={{ __html: about.methodology }}
          />
        )}
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">情報源について（EDINET）</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          本サイトが扱う大量保有報告書などの開示情報は、金融庁が運営する電子開示システム
          「EDINET（Electronic Disclosure for Investors&apos; NETwork）」の公開情報を一次情報源としています。
          ご自身で開示書類の原本を確認したい場合は、以下から直接検索できます。
        </p>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm leading-relaxed text-foreground/70">
          <li>
            <a
              href="https://disclosure2.edinet-fsa.go.jp/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-brand-blue hover:underline"
            >
              EDINET（書類検索）
            </a>
          </li>
          <li>
            <a
              href="https://disclosure2dl.edinet-fsa.go.jp/guide/static/disclosure/WZEK0110.html"
              target="_blank"
              rel="noopener noreferrer"
              className="text-brand-blue hover:underline"
            >
              EDINET API仕様書
            </a>
            （本サイトが開示情報の自動取得に利用している金融庁公式APIの仕様書です）
          </li>
        </ul>
        <p className="mt-3 text-sm leading-relaxed text-foreground/70">
          各記事の「出典」欄には、判明している場合、その記事の元になった個別の開示書類・適時開示への
          直接リンクを掲載しています。
        </p>
      </section>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-foreground/70">
          大量保有報告書のしくみや本サイトの使い方については
          <Link href="/faq" className="text-brand-blue hover:underline">
            よくある質問
          </Link>
          もあわせてご覧ください。
        </p>
      </section>

      <section id="dealtype-glossary" className="mb-6 scroll-mt-20">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">投資家分類（用語集）</h2>
        <p className="mb-3 text-sm leading-relaxed text-foreground/70">
          記事に付いているバッジは、大量保有報告書の提出者を以下のいずれかに分類したものです。
        </p>
        <dl className="space-y-3 text-sm">
          {DEAL_TYPES.map((dealType) => (
            <div key={dealType}>
              <dt className="font-semibold text-brand-navy">{dealType}</dt>
              <dd className="text-foreground/70">{DEAL_TYPE_DESCRIPTIONS[dealType]}</dd>
            </div>
          ))}
        </dl>
      </section>

      {about.faq && (
        <section className="mb-6">
          <h2 className="mb-2 text-lg font-bold text-brand-navy">よくある質問</h2>
          <div
            className="prose prose-sm max-w-none text-foreground/70"
            dangerouslySetInnerHTML={{ __html: about.faq }}
          />
        </section>
      )}

      <section>
        <h2 className="mb-2 text-lg font-bold text-brand-navy">免責事項</h2>
        <div
          className="prose prose-sm max-w-none text-foreground/70"
          dangerouslySetInnerHTML={{ __html: about.disclaimer }}
        />
      </section>
    </article>
  );
}
