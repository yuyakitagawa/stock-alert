// ページ冒頭の主要な数値を定義リスト（<dl>）で構造化して置くためのボックス。
// 散文の中の数字より抜き出されやすく、AI検索（GEO）が回答に使う「事実」の受け皿になる。
// 値はすべてDBの集計値なのでLLM生成は不要＝ハルシネーションもAPIコストもゼロ。
// マークアップは既存の PriceAfterDisclosure / FilerReturnRecord の <dl> と揃えている
// （あちらは専用の数値パネルで見出し・注記の構成が違うため、共通化はせず形だけ揃える）。

export type Fact = {
  label: string;
  value: string;
  // 値の右に小さく添える補足（単位の言い換え・内訳など）。
  note?: string;
  tone?: "gain" | "loss";
};

export default function FactBox({ facts, caption }: { facts: Fact[]; caption?: string }) {
  if (facts.length === 0) return null;
  return (
    <section className="mb-8 rounded-md border border-rule bg-section-tint p-4">
      <dl className="m-0 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        {facts.map((fact) => {
          const color =
            fact.tone === "gain"
              ? "text-gain"
              : fact.tone === "loss"
                ? "text-loss"
                : "text-foreground/80";
          return (
            <div key={fact.label}>
              <dt className="text-[11px] text-foreground/40">{fact.label}</dt>
              <dd className={`m-0 mt-0.5 text-sm font-bold tabular-nums ${color}`}>
                {fact.value}
                {fact.note && (
                  <span className="ml-1 text-xs font-normal text-foreground/50">{fact.note}</span>
                )}
              </dd>
            </div>
          );
        })}
      </dl>
      {caption && <p className="mb-0 mt-3 text-[11px] leading-relaxed text-foreground/40">{caption}</p>}
    </section>
  );
}
