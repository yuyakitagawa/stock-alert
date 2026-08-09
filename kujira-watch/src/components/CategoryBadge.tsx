import Link from "next/link";
import type { DealType } from "@/types/article";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import type { Locale } from "@/lib/i18n";

export default function CategoryBadge({
  dealType,
  locale = "ja",
}: {
  dealType: DealType | undefined;
  locale?: Locale;
}) {
  if (!dealType) return null;
  const label = locale === "en" ? DEAL_TYPE_EN[dealType].label : dealType;
  const href =
    locale === "en"
      ? `/en/category/${DEAL_TYPE_EN[dealType].slug}`
      : `/category/${encodeURIComponent(dealType)}`;
  return (
    <Link
      href={href}
      className="kicker inline-flex items-center border-b border-brand-navy/40 text-brand-navy transition-colors hover:border-brand-gold hover:text-brand-gold"
    >
      {label}
    </Link>
  );
}
