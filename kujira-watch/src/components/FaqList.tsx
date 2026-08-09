"use client";

import { useState } from "react";

export type FaqCategory = { id: string; label: string };

export type FaqItem = {
  question: string;
  answer: string;
  category: string;
  render?: React.ReactNode;
};

// カテゴリごとに色分けした質問バッジ用。9カテゴリ固定のため、増減時はここも見直す。
const CATEGORY_COLORS: Record<string, string> = {
  basics: "text-brand-blue",
  terms: "text-violet-600",
  "term-supplement": "text-indigo-600",
  "regulation-deep-dive": "text-rose-600",
  "investor-players": "text-cyan-600",
  "investing-basics-extra": "text-amber-600",
  usage: "text-emerald-600",
  howto: "text-brand-gold",
  "about-site": "text-foreground/50",
};

// カテゴリタブは見た目上の絞り込みのみで、初期状態は必ず「すべて」（全件表示）。
// クローラー・AIOがJS実行前に読むSSR済みHTMLには常に全問が含まれるようにするため。
export default function FaqList({
  faqs,
  categories,
}: {
  faqs: FaqItem[];
  categories: FaqCategory[];
}) {
  const [active, setActive] = useState<string>("all");
  const filtered = active === "all" ? faqs : faqs.filter((faq) => faq.category === active);
  const categoryLabel = (id: string) =>
    categories.find((category) => category.id === id)?.label ?? id;

  return (
    <div>
      <div
        role="tablist"
        aria-label="カテゴリで絞り込み"
        className="no-scrollbar kicker mb-6 flex flex-nowrap items-center gap-x-4 gap-y-1 overflow-x-auto border-b border-rule pb-2 sm:flex-wrap sm:overflow-visible"
      >
        <button
          type="button"
          role="tab"
          aria-selected={active === "all"}
          onClick={() => setActive("all")}
          className={`shrink-0 border-b-2 pb-1 transition-colors ${
            active === "all"
              ? "border-brand-gold text-brand-navy"
              : "border-transparent text-brand-navy/50 hover:text-brand-navy"
          }`}
        >
          すべて（{faqs.length}）
        </button>
        {categories.map((category) => {
          const count = faqs.filter((faq) => faq.category === category.id).length;
          return (
            <button
              key={category.id}
              type="button"
              role="tab"
              aria-selected={active === category.id}
              onClick={() => setActive(category.id)}
              className={`shrink-0 border-b-2 pb-1 transition-colors ${
                active === category.id
                  ? "border-brand-gold text-brand-navy"
                  : "border-transparent text-brand-navy/50 hover:text-brand-navy"
              }`}
            >
              {category.label}（{count}）
            </button>
          );
        })}
      </div>
      <dl className="space-y-5 text-sm">
        {filtered.map((faq) => (
          <div key={faq.question}>
            <dt>
              <span
                className={`kicker mb-1 flex items-center gap-1.5 ${
                  CATEGORY_COLORS[faq.category] ?? "text-foreground/50"
                }`}
              >
                <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full bg-current" />
                {categoryLabel(faq.category)}
              </span>
              <span className="font-semibold text-brand-navy">{faq.question}</span>
            </dt>
            <dd className="mt-1 leading-relaxed text-foreground/70">{faq.render ?? faq.answer}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
