import type { Metadata } from "next";
import Link from "next/link";
import { SITE_NAME_EN, SITE_URL } from "@/lib/site";

const title = "Privacy Policy";
const description =
  "How this site uses cookies for analytics and third-party advertising (Google AdSense), and how to opt out.";

export const metadata: Metadata = {
  title,
  description,
  alternates: {
    canonical: `${SITE_URL}/en/privacy`,
    languages: { ja: `${SITE_URL}/privacy`, en: `${SITE_URL}/en/privacy` },
  },
};

export default function EnPrivacyPage() {
  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-foreground/70">
          This policy describes how visitor information is handled on {SITE_NAME_EN} ({SITE_URL}
          , &quot;this site&quot;).
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">Advertising</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          This site uses Google AdSense, a third-party advertising service. Google and its partners
          use cookies to serve ads based on a visitor&apos;s prior visits to this site and other
          sites.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-foreground/70">
          You can turn off personalized advertising in{" "}
          <a
            href="https://myadcenter.google.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            Google Ads Settings
          </a>
          . To opt out of third-party vendors&apos; use of cookies more broadly, see{" "}
          <a
            href="https://www.aboutads.info/choices/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            www.aboutads.info
          </a>
          . How Google uses information from sites that use its services is described in{" "}
          <a
            href="https://policies.google.com/technologies/ads"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            Google&apos;s advertising policies
          </a>
          .
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">Analytics</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          This site uses the following tools to understand how it is being read. None of them
          collect personally identifying information such as names, email addresses, or phone
          numbers.
        </p>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-foreground/70">
          <li>
            <strong className="text-brand-navy">Google Analytics (GA4)</strong>: collects traffic
            data using cookies. To opt out, use the{" "}
            <a
              href="https://tools.google.com/dlpage/gaoptout"
              target="_blank"
              rel="noopener noreferrer"
              className="text-brand-blue hover:underline"
            >
              Google Analytics Opt-out Browser Add-on
            </a>
            .
          </li>
          <li>
            <strong className="text-brand-navy">Vercel Analytics / Speed Insights</strong>:
            aggregates page views and page-load performance.
          </li>
          <li>
            <strong className="text-brand-navy">Our own access log</strong>: records access time,
            path, and User-Agent to distinguish search-engine crawlers from real browser visits.
            Browser visits are tagged with a random identifier (the <code>kw_vid</code> cookie) so
            that unique visitors are not double-counted; this identifier cannot be used to identify
            an individual.
          </li>
        </ul>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">Disabling cookies</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          You can disable or delete cookies at any time in your browser settings. Some features of
          this site may not work correctly if you do.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">Personal information</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          This site has no sign-up, contact form, or comment feature, and does not collect personal
          information directly from visitors. The operator does not use the information gathered by
          the analytics and advertising services above to identify individuals.
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">Disclaimer</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          For the site&apos;s disclaimer, data sources, and operator information, see{" "}
          <Link href="/en/about" className="text-brand-blue hover:underline">
            About &amp; Disclaimer
          </Link>
          . The operator takes no responsibility for information or services provided on external
          sites, including advertisers&apos; sites, linked from this site.
        </p>
      </section>

      <section>
        <h2 className="mb-2 text-lg font-bold text-brand-navy">Changes to this policy</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          This policy may be revised without notice as laws change or as the tools used by this site
          change. Questions can be sent to{" "}
          <a
            href="https://x.com/kujira_watch"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            @kujira_watch on X
          </a>
          .
        </p>
      </section>
    </article>
  );
}
