import type { Metadata } from "next";
import Link from "next/link";
import { EN_SITE_URL, SITE_NAME_EN } from "@/lib/en";
import { SITE_URL, X_PROFILE_URL } from "@/lib/site";

const title = "Privacy Policy";
const description = "How this site uses cookies for analytics and how to opt out.";

export const metadata: Metadata = {
  title,
  description,
  alternates: {
    canonical: `${EN_SITE_URL}/privacy`,
    languages: { ja: `${SITE_URL}/privacy`, en: `${EN_SITE_URL}/privacy` },
  },
};

const link = "text-brand-blue hover:underline";

export default function EnPrivacyPage() {
  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-ink-secondary">
          This policy describes how visitor information is handled on {SITE_NAME_EN} ({EN_SITE_URL},
          &quot;this site&quot;), the English edition of {SITE_URL}.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Analytics</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          This site uses the following tools to understand how it is being read. None of them collect
          personally identifying information such as names, email addresses, or phone numbers.
        </p>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-ink-secondary">
          <li>
            <strong className="text-brand-navy">Google Analytics (GA4)</strong>: collects traffic data
            using cookies. To opt out, use the{" "}
            <a href="https://tools.google.com/dlpage/gaoptout" target="_blank" rel="noopener noreferrer" className={link}>
              Google Analytics Opt-out Browser Add-on
            </a>
            .
          </li>
          <li>
            <strong className="text-brand-navy">Our own access log</strong>: records access time, host,
            path, IP address, and User-Agent to distinguish search-engine and AI crawlers from real browser
            visits. Browser visits are tagged with a random identifier (the <code>kw_vid</code> cookie) so
            that unique visitors are not double-counted; this identifier cannot be used to identify an
            individual.
          </li>
        </ul>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Advertising</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          The English edition does not serve third-party advertising. The Japanese edition uses Google
          AdSense; see its{" "}
          <a href={`${SITE_URL}/privacy`} hrefLang="ja" className={link}>
            privacy policy (Japanese)
          </a>
          .
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Disabling cookies</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          You can disable or delete cookies at any time in your browser settings. Some features of this
          site may not work correctly if you do.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Personal information</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          This site has no sign-up, contact form, or comment feature, and does not collect personal
          information directly from visitors. The operator does not use the information gathered by the
          analytics services above to identify individuals.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Disclaimer</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          For the site&apos;s disclaimer, data sources, and operator information, see{" "}
          <Link href="/about" className={link}>
            About &amp; Disclaimer
          </Link>
          . The operator takes no responsibility for information or services provided on external sites
          linked from this site.
        </p>
      </section>

      <section>
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Changes to this policy</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          This policy may be revised without notice as laws change or as the tools used by this site
          change. Questions can be sent to{" "}
          <a href={X_PROFILE_URL} target="_blank" rel="noopener noreferrer" className={link}>
            @kujira_watch on X
          </a>
          .
        </p>
      </section>
    </article>
  );
}
