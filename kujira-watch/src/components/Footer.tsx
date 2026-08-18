import Link from "next/link";
import { SITE_NAME, SITE_NAME_EN, X_PROFILE_URL, X_SCREEN_NAME, YOUTUBE_CHANNEL_URL } from "@/lib/site";
import type { Locale } from "@/lib/i18n";

// フッターは全ページ共通で出す。かつて「オートスクロールで一覧が伸びるとフッターに
// たどり着けない」という理由でハンバーガーメニューへ集約したが、メニューは開かないと
// 見えず、規約・プライバシー等の定番リンクがページ内に一切無い状態だった。
// メニュー(HeaderMenu)は残したまま、フッターも全ページに戻す。
type FooterLink = { href: string; label: string; external?: boolean };
type FooterGroup = { heading: string; links: FooterLink[] };

const JA_GROUPS: FooterGroup[] = [
  {
    heading: "主要ページ",
    links: [
      { href: "/", label: "TOP" },
      { href: "/disclosures", label: "開示速報" },
      { href: "/trending", label: "急増銘柄" },
      { href: "/ranking", label: "投資家ランキング" },
      { href: "/weekly", label: "週次トレンド" },
      { href: "/activists", label: "アクティビストの動き" },
    ],
  },
  {
    heading: "一覧・アーカイブ",
    links: [
      { href: "/investors", label: "投資家一覧" },
      { href: "/stocks", label: "銘柄一覧" },
      { href: "/monthly", label: "月別アーカイブ" },
    ],
  },
  {
    heading: "サイト情報",
    links: [
      { href: "/about", label: "このサイトについて" },
      { href: "/faq", label: "よくある質問" },
      { href: "/privacy", label: "プライバシーポリシー" },
      { href: "/terms", label: "利用規約" },
    ],
  },
  {
    heading: "フォロー",
    links: [
      { href: X_PROFILE_URL, label: `公式X（@${X_SCREEN_NAME}）`, external: true },
      { href: YOUTUBE_CHANNEL_URL, label: "公式YouTube", external: true },
      { href: "/feed.xml", label: "RSSフィード" },
      { href: "/en", label: "English" },
    ],
  },
];

const EN_GROUPS: FooterGroup[] = [
  {
    heading: "Contents",
    links: [{ href: "/en", label: "Top" }],
  },
  {
    heading: "Site info",
    links: [
      { href: "/en/about", label: "About this site" },
      { href: "/en/privacy", label: "Privacy policy" },
    ],
  },
  {
    heading: "Follow",
    links: [
      { href: X_PROFILE_URL, label: `Official X (@${X_SCREEN_NAME})`, external: true },
      { href: YOUTUBE_CHANNEL_URL, label: "Official YouTube", external: true },
      { href: "/", label: "日本語" },
    ],
  },
];

const DISCLAIMER_JA =
  "本サイトはEDINET大量保有報告書等の公開情報をもとにした解説であり、投資助言ではありません。掲載情報の正確性・完全性を保証するものではなく、投資に関する最終判断はご自身の責任で行ってください。";
const DISCLAIMER_EN =
  "This site is commentary based on public EDINET large-shareholding filings and is not investment advice. Accuracy and completeness are not guaranteed; make investment decisions at your own responsibility.";

export default function Footer({ locale = "ja" }: { locale?: Locale }) {
  const groups = locale === "en" ? EN_GROUPS : JA_GROUPS;
  const year = new Date().getFullYear();

  return (
    <footer className="mt-auto border-t border-rule bg-section-tint">
      <div className="mx-auto w-full max-w-3xl px-4 py-10">
        <nav
          aria-label={locale === "en" ? "Footer" : "フッターリンク"}
          className="grid grid-cols-2 gap-x-6 gap-y-8 sm:grid-cols-4"
        >
          {groups.map((group) => (
            <div key={group.heading}>
              <h2 className="kicker mb-2 text-brand-navy">{group.heading}</h2>
              <ul className="space-y-1.5 text-sm">
                {group.links.map((link) => (
                  <li key={link.href}>
                    {link.external ? (
                      <a
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-brand-blue hover:underline"
                      >
                        {link.label}
                      </a>
                    ) : (
                      <Link href={link.href} className="text-brand-blue hover:underline">
                        {link.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <p className="mt-8 border-t border-rule pt-4 text-xs leading-relaxed text-foreground/60">
          {locale === "en" ? DISCLAIMER_EN : DISCLAIMER_JA}
        </p>
        <p className="mt-3 text-xs text-foreground/50">
          © {year} {locale === "en" ? SITE_NAME_EN : SITE_NAME}
        </p>
      </div>
    </footer>
  );
}
