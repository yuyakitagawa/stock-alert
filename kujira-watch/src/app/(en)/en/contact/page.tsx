import type { Metadata } from "next";
import Link from "next/link";
import { EN_SITE_URL, SITE_NAME_EN } from "@/lib/en";
import { SITE_URL, X_HANDLE, X_PROFILE_URL } from "@/lib/site";

const title = "Contact";
const description = `How to reach ${SITE_NAME_EN}: corrections, removal requests, and citation or media inquiries.`;

export const metadata: Metadata = {
  title,
  description,
  alternates: {
    canonical: `${EN_SITE_URL}/contact`,
    languages: { ja: `${SITE_URL}/contact`, en: `${EN_SITE_URL}/contact` },
  },
};

const link = "text-brand-blue hover:underline";

// 日本語版 /contact の英語版。英語版にだけ窓口・規約ページが無く、フッターも
// About と Privacy しか無かった（2026-09-10 のAdSense再監査で検出）。窓口は日本語版と同じ公式X。
export default function EnContactPage() {
  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-ink-secondary">
          {SITE_NAME_EN} accepts messages through direct messages or replies to its official X account,{" "}
          <a href={X_PROFILE_URL} target="_blank" rel="noopener noreferrer" className={link}>
            {X_HANDLE}
          </a>
          . The operator does not publish a name or email address, so this is the only contact point. Messages
          in English or Japanese are both fine.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Reporting an error</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          If you find a wrong number, company name, or holding ratio in an article, send the article URL. We check
          it against the original filing (EDINET large-shareholding reports, TDnet timely disclosures) and correct
          or remove the article if the error is confirmed. The English edition follows the Japanese edition; when
          the two disagree, the{" "}
          <a href={SITE_URL} hrefLang="ja" lang="ja" className={link}>
            Japanese edition
          </a>{" "}
          carries the latest corrected data.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Correction and removal requests</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          Companies and investors named in an article can request a correction or removal through the same
          channel. The site only covers information disclosed under Japan&apos;s Financial Instruments and Exchange
          Act, but please tell us if an article misstates a filing or describes it in a way the filing does not
          support.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Citation, reuse, and media inquiries</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          Requests to quote or reuse articles and charts, and media inquiries, go to the same contact point. The
          conditions for quoting are set out in the{" "}
          <Link href="/terms" className={link}>
            Terms of Use
          </Link>
          .
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">What we cannot answer</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          This site is not a registered investment adviser. We cannot answer questions about whether to buy or sell
          a particular stock, and we do not accept requests to feature a stock or to publish content that is not
          based on a filing.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Personal information</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          The site has no contact form and does not collect names or email addresses on its pages. Messages sent
          through X are used only to respond to the request. See the{" "}
          <Link href="/privacy" className={link}>
            Privacy Policy
          </Link>{" "}
          for details.
        </p>
      </section>

      <p className="text-sm leading-relaxed text-ink-secondary">
        <a href={X_PROFILE_URL} target="_blank" rel="noopener noreferrer" className={link}>
          Open {X_HANDLE} on X
        </a>
      </p>
    </article>
  );
}
