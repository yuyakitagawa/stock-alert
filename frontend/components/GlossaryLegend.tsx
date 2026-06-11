import { GLOSSARY, GLOSSARY_ORDER } from "@/lib/glossary";

/**
 * 用語の説明（開閉式）。表のヘッダーやスコアの専門用語を平易に解説する。
 * ホバーが使えないモバイルでも読めるよう <details> で実装。
 */
export default function GlossaryLegend({ keys }: { keys?: readonly string[] }) {
  const order = keys ?? GLOSSARY_ORDER;
  return (
    <details className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 group">
      <summary className="text-sm text-gray-400 cursor-pointer select-none hover:text-gray-200 list-none flex items-center gap-2">
        <span className="text-blue-400">ⓘ</span>
        用語の説明（PER・PBR・ネットスコアなど）
        <span className="text-gray-600 text-xs ml-auto group-open:hidden">タップで開く ▾</span>
      </summary>
      <dl className="mt-3 space-y-2.5">
        {order.map((k) => {
          const t = GLOSSARY[k];
          if (!t) return null;
          return (
            <div key={k} className="flex flex-col sm:flex-row sm:gap-3">
              <dt className="text-sm font-semibold text-gray-200 sm:w-40 shrink-0">{t.term}</dt>
              <dd className="text-sm text-gray-500 leading-relaxed">{t.long}</dd>
            </div>
          );
        })}
      </dl>
    </details>
  );
}
