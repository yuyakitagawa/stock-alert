import type { Metadata } from "next";
import Link from "next/link";
import { EN_SITE_URL, SITE_NAME_EN } from "@/lib/en";
import { SITE_URL } from "@/lib/site";

const title = "Terms of Use";
const description = `Terms of use, content reuse, and disclaimers for ${SITE_NAME_EN}.`;

export const metadata: Metadata = {
  title,
  description,
  alternates: {
    canonical: `${EN_SITE_URL}/terms`,
    languages: { ja: `${SITE_URL}/terms`, en: `${EN_SITE_URL}/terms` },
  },
};

const link = "text-brand-blue hover:underline";

// 日本語版 /terms の英語版（内容は同じ。2026-09-10 のAdSense再監査で英語版に無いことを検出）。
export default function EnTermsPage() {
  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-ink-secondary">
          These terms govern the use of {SITE_NAME_EN} ({EN_SITE_URL}), the English edition of {SITE_URL}, together
          with the social media accounts and video channel it operates (together, &quot;the service&quot;). By using
          the service, you agree to these terms.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">What the service provides</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          The service reports on share purchases and sales by large investors in Japanese listed companies, based on
          public disclosures such as the large-shareholding reports filed on EDINET. Articles summarize and explain
          public information; we do not guarantee that they are accurate, complete, or up to date. The Japanese
          edition is updated first, and corrections reach it before the English edition.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Not investment advice</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          Nothing on the service is a solicitation or investment advice. It is not advice on investment decisions
          based on an analysis of the value of financial instruments, and it does not recommend buying or selling
          any stock. Investment decisions are your own responsibility. The operator accepts no liability for any
          loss arising from use of the information on the service.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Intellectual property</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          Copyright in the articles, videos, images, and other content on the service belongs to the operator or to
          the third parties who hold the rights. Beyond quotation permitted by law, the content may not be copied,
          republished, or redistributed without permission.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Prohibited conduct</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          When using the service, you may not break the law or public order, interfere with the operation of the
          service, or do anything else the operator considers inappropriate.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Changes to these terms</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          The operator may change these terms when necessary. Revised terms take effect when they are posted on this
          page. If this English text and the{" "}
          <a href={`${SITE_URL}/terms`} hrefLang="ja" lang="ja" className={link}>
            Japanese terms
          </a>{" "}
          differ, the Japanese terms prevail.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Related pages</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          For how visitor information is handled, see the{" "}
          <Link href="/privacy" className={link}>
            Privacy Policy
          </Link>
          . To report an error or make a request, see{" "}
          <Link href="/contact" className={link}>
            Contact
          </Link>
          .
        </p>
      </section>
    </article>
  );
}
