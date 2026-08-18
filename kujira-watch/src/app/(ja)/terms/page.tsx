import type { Metadata } from "next";
import Link from "next/link";
import { SITE_NAME, SITE_URL } from "@/lib/site";

const title = "利用規約";
const description = `${SITE_NAME}の利用条件、コンテンツの取り扱い、免責事項について定めたものです。`;

export const metadata: Metadata = {
  title,
  description,
  alternates: {
    canonical: `${SITE_URL}/terms`,
  },
};

export default function TermsPage() {
  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-foreground/70">
          本規約は、{SITE_NAME}（{SITE_URL}
          。以下「本サイト」）および本サイトが運営するSNSアカウント・動画チャンネル（以下あわせて「本サービス」）の利用条件を定めたものです。閲覧者は、本サービスを利用することで本規約に同意したものとみなします。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">本サービスの内容</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          本サービスは、EDINET等で公開されている大量保有報告書などの公開情報をもとに、大口投資家による株式売買の動向を記事・動画等の形式で提供するものです。掲載内容は公開情報の要約・解説であり、その正確性・完全性・最新性を保証するものではありません。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">投資判断に関する免責</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          本サービスが提供する情報は、投資勧誘・投資助言を目的としたものではありません。金融商品の価値の分析に基づく投資判断の助言を行うものではなく、特定の銘柄の売買を推奨するものでもありません。投資に関する最終的な判断は、閲覧者ご自身の責任において行ってください。本サービスの情報を利用したことにより生じたいかなる損害についても、運営者は一切の責任を負いません。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">知的財産権</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          本サービスに掲載される記事・動画・画像等のコンテンツの著作権は、運営者または正当な権利を有する第三者に帰属します。法令で認められる引用の範囲を超えて、無断で複製・転載・再配布することはできません。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">禁止事項</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          本サービスの利用にあたり、法令または公序良俗に違反する行為、本サービスの運営を妨害する行為、その他運営者が不適切と判断する行為を禁止します。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">規約の変更</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          運営者は、必要に応じて本規約を変更できるものとします。変更後の規約は、本ページに掲載した時点で効力を生じます。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">関連ページ</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          閲覧者の情報の取り扱いについては、
          <Link href="/privacy" className="text-brand-blue underline">
            プライバシーポリシー
          </Link>
          をご覧ください。
        </p>
      </section>
    </article>
  );
}
