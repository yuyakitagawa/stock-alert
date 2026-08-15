import type { Metadata } from "next";
import Link from "next/link";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import { SITE_NAME_EN, SITE_URL } from "@/lib/site";
import { DEAL_TYPES } from "@/types/article";
import { UI } from "@/lib/i18n";
import AdUnit from "@/components/AdUnit";

const t = UI.en;

export const metadata: Metadata = {
  title: t.aboutTitle,
  description: t.aboutMetaDescription,
  alternates: {
    canonical: `${SITE_URL}/en/about`,
    languages: { ja: `${SITE_URL}/about`, en: `${SITE_URL}/en/about` },
  },
};

export default function EnAboutPage() {
  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{t.aboutTitle}</h1>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">{t.aboutWhoHeading}</h2>
        <p className="text-sm leading-relaxed text-foreground/70">{t.aboutWhoIntro}</p>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm leading-relaxed text-foreground/70">
          {t.aboutWhoItems.map((item) => (
            <li key={item.term}>
              <strong className="text-brand-navy">{item.term}</strong>: {item.body}
            </li>
          ))}
        </ul>
        <p className="mt-3 text-sm leading-relaxed text-foreground/70">
          {`This site classifies these investors into 13 categories. See the `}
          <Link href="#dealtype-glossary" className="text-brand-blue hover:underline">
            glossary
          </Link>
          {` below for details.`}
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">{t.aboutWhyHeading}</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-foreground/70">
          {t.aboutWhyItems.map((item) => (
            <li key={item.term}>
              <strong className="text-brand-navy">{item.term}</strong>: {item.body}
            </li>
          ))}
        </ul>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">{t.aboutWhatHeading}</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          {`"Big-investor moves" refers to when and how much of a listed company's shares were bought or sold by an entity with market-moving capital (a "whale") — institutional investors, activist funds, founder-family holding companies, or share buybacks. In Japan, investors who acquire 5% or more of a company's shares must file a large-shareholding report with EDINET (the Financial Services Agency's disclosure system) — the so-called "5% rule" — and must re-file a change report if their holding ratio moves by 1 percentage point or more. This site tracks that public information daily and organizes it by stock and investor category.`}
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">{t.aboutSiteHeading}</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          {`${SITE_NAME_EN}${t.aboutSiteBody}`}
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">{t.aboutDataHeading}</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm leading-relaxed text-foreground/70">
          {t.aboutDataItems.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">{t.aboutSourceHeading}</h2>
        <p className="text-sm leading-relaxed text-foreground/70">{t.aboutSourceIntro}</p>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm leading-relaxed text-foreground/70">
          <li>
            <a
              href="https://disclosure2.edinet-fsa.go.jp/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-brand-blue hover:underline"
            >
              {t.aboutSourceEdinetLabel}
            </a>
          </li>
          <li>
            <a
              href="https://disclosure2dl.edinet-fsa.go.jp/guide/static/disclosure/WZEK0110.html"
              target="_blank"
              rel="noopener noreferrer"
              className="text-brand-blue hover:underline"
            >
              {t.aboutSourceApiLabel}
            </a>
            {t.aboutSourceApiNote}
          </li>
        </ul>
        <p className="mt-3 text-sm leading-relaxed text-foreground/70">{t.aboutSourceNote}</p>
      </section>

      <section id="dealtype-glossary" className="mb-6 scroll-mt-20">
        <h2 className="mb-2 text-lg font-bold text-brand-navy">{t.aboutGlossaryHeading}</h2>
        <p className="mb-3 text-sm leading-relaxed text-foreground/70">{t.aboutGlossaryIntro}</p>
        <Box component="dl" sx={{ m: 0, display: "flex", flexDirection: "column", gap: 1.5 }}>
          {DEAL_TYPES.map((dealType) => (
            <Box key={dealType}>
              <Typography component="dt" sx={{ fontWeight: 600, color: "primary.main" }}>
                {DEAL_TYPE_EN[dealType].label}
              </Typography>
              <Typography component="dd" variant="body2" sx={{ m: 0, color: "text.secondary" }}>
                {DEAL_TYPE_EN[dealType].description}
              </Typography>
            </Box>
          ))}
        </Box>
      </section>

      <section>
        <h2 className="mb-2 text-lg font-bold text-brand-navy">{t.aboutDisclaimerHeading}</h2>
        <p className="text-sm leading-relaxed text-foreground/70">{t.aboutDisclaimerBody}</p>
      </section>
      <AdUnit placement="bottom" locale="en" />
    </article>
  );
}
