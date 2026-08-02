import type { Metadata } from "next";
import { SITE_NAME, SITE_URL } from "@/lib/site";

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
        <h2 className="mb-2 text-lg font-semibold text-brand-navy">このサイトについて</h2>
        <p className="text-sm leading-relaxed text-gray-700">
          {SITE_NAME}は、EDINET大量保有報告書（5%ルール）などの公開情報をもとに、機関投資家・
          インサイダー・自社株買いといった大口投資家の動きを監視し、解説するブログです。
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
