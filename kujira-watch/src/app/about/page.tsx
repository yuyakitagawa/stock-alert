import type { Metadata } from "next";
import Link from "next/link";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { SITE_NAME, SITE_URL } from "@/lib/site";
import { DEAL_TYPES } from "@/types/article";

const title = "運営者情報・免責事項";
const description = "本サイトの運営方針、データソース、免責事項について。";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: `${SITE_URL}/about` },
};

export default function AboutPage() {
  return (
    <article className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-200">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy">{title}</h1>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">大口投資家とは</h2>
        <p className="text-sm leading-relaxed text-gray-700">
          株式や各種金融市場において、市場価格を動かすほどの巨大な資金を運用する投資主体です。
        </p>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm leading-relaxed text-gray-700">
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
        <p className="mt-3 text-sm leading-relaxed text-gray-700">
          {SITE_NAME}ではこれらの投資家を13種類に分類しています。詳しくは
          <Link href="#dealtype-glossary" className="text-brand-blue hover:underline">
            用語集
          </Link>
          をご覧ください。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">大口投資家の動きを追う意味</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-gray-700">
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
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">大口投資家の動きとは</h2>
        <p className="text-sm leading-relaxed text-gray-700">
          「大口投資家の動き」とは、機関投資家・アクティビストファンド・創業家の資産管理会社・
          自社株買いなど、相場を動かすほどの資金力を持つ投資主体（クジラ）が、上場企業の株式を
          いつ・どれくらいの規模で売買したかを指します。日本では、株式の5%以上を保有した投資家は
          EDINET（金融庁の開示システム）に大量保有報告書の提出が義務付けられており（5%ルール）、
          保有比率が1%以上増減した場合は変更報告書として再提出されます。{SITE_NAME}はこの公開情報を
          毎日追跡し、銘柄別・投資家分類別に整理して解説しています。直近の動きは
          <Link href="/weekly" className="text-brand-blue hover:underline">
            「今週の動き」ページ
          </Link>
          でまとめて確認できます。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">このサイトについて</h2>
        <p className="text-sm leading-relaxed text-gray-700">
          {SITE_NAME}は、EDINET大量保有報告書（5%ルール）などの公開情報をもとに、機関投資家・
          インサイダー・自社株買いといった「クジラ」（相場を動かすほどの資金力を持つ大口投資家を指す
          金融業界の俗称）の動きを監視し、解説するブログです。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">データソースと更新方法</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-gray-700">
          <li>取引情報はEDINET大量保有報告書（買い方向のみ）を日次で取得しています。</li>
          <li>
            金額規模は、発行済株式数×株価×保有比率変化から算出した推定値です。EDINET開示自体には
            金額の記載がないため、正確な金額を保証するものではありません。
          </li>
          <li>記事はAIが事実情報から生成した上で、必要に応じて人が内容を確認・修正しています。</li>
          <li>出典元（適時開示・プレスリリース等）が判明している記事には、記事内にリンクを掲載しています。</li>
        </ul>
      </section>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-gray-700">
          大量保有報告書のしくみや本サイトの使い方については
          <Link href="/faq" className="text-brand-blue hover:underline">
            よくある質問
          </Link>
          もあわせてご覧ください。
        </p>
      </section>

      <section id="dealtype-glossary" className="mb-6 scroll-mt-20">
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">投資家分類（用語集）</h2>
        <p className="mb-3 text-sm leading-relaxed text-gray-700">
          記事に付いているバッジは、大量保有報告書の提出者を以下のいずれかに分類したものです。
        </p>
        <dl className="space-y-3 text-sm">
          {DEAL_TYPES.map((dealType) => (
            <div key={dealType}>
              <dt className="font-semibold text-brand-navy">{dealType}</dt>
              <dd className="text-gray-700">{DEAL_TYPE_DESCRIPTIONS[dealType]}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section>
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">免責事項</h2>
        <p className="text-sm leading-relaxed text-gray-700">
          本サイトの内容は情報提供を目的としたものであり、特定の銘柄や投資判断を推奨・勧誘するものではありません。
          掲載情報の正確性・完全性を保証するものではなく、本サイトの情報に基づいて被ったいかなる損害についても
          運営者は責任を負いません。投資に関する最終判断はご自身の責任で行ってください。
        </p>
      </section>
    </article>
  );
}
