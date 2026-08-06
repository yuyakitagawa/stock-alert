"use client";

import { useState } from "react";

export type FaqCategory = { id: string; label: string };

export type FaqItem = {
  question: string;
  answer: string;
  category: string;
  render?: React.ReactNode;
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
            <dt className="font-semibold text-brand-navy">{faq.question}</dt>
            <dd className="mt-1 leading-relaxed text-foreground/70">{faq.render ?? faq.answer}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
