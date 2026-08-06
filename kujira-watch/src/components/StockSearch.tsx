"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { StockSearchResult } from "@/lib/microcms";

// 企業名・証券コードで検索し、選択(またはEnter)で /stocks/[code] に遷移する。
// レスポンスを都度撃たないよう、入力停止から300ms待ってからAPIを叩く。
const DEBOUNCE_MS = 300;

export default function StockSearch() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<StockSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const close = () => {
    setOpen(false);
    setQuery("");
    setResults([]);
  };

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    if (!open) return;

    function handlePointerDown(event: PointerEvent) {
      if (!containerRef.current?.contains(event.target as Node)) close();
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") close();
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const trimmedQuery = query.trim();

  useEffect(() => {
    if (!trimmedQuery) return;

    const timer = setTimeout(() => {
      setLoading(true);
      fetch(`/api/stocks/search?q=${encodeURIComponent(trimmedQuery)}`)
        .then((res) => (res.ok ? res.json() : { results: [] }))
        .then((data) => setResults(data.results ?? []))
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [trimmedQuery]);

  const visibleResults = trimmedQuery ? results : [];

  const goToStock = (stockCode: string) => {
    close();
    router.push(`/stocks/${stockCode}`);
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (visibleResults[0]) goToStock(visibleResults[0].stockCode);
  };

  return (
    <div ref={containerRef} className="relative shrink-0">
      <button
        type="button"
        aria-label="企業名・証券コードで検索"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="flex h-9 w-9 items-center justify-center text-brand-navy hover:text-brand-gold"
      >
        {/* viewBoxをマイナスオフセットし、円+持ち手の絵が隣のハンバーガー(☰)と
            ボタン内で同じ視覚的重心に来るよう補正している（0 0 24 24のままだと左上に寄って見える）。 */}
        <svg
          aria-hidden
          viewBox="-1.5 -1.5 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-5 w-5"
        >
          <circle cx="11" cy="11" r="7" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      </button>

      {open && (
        <div className="absolute right-0 top-11 z-50 w-72 rounded-md border border-rule bg-paper shadow-xl sm:w-80">
          <form onSubmit={handleSubmit} className="border-b border-rule p-2">
            <input
              ref={inputRef}
              type="text"
              inputMode="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="企業名 or 証券コードで検索"
              className="w-full rounded border border-rule bg-transparent px-3 py-2 text-sm text-brand-navy outline-none focus:border-brand-blue"
            />
          </form>

          <div className="max-h-80 overflow-y-auto">
            {loading && (
              <p className="px-3 py-3 text-xs text-foreground/50">検索中…</p>
            )}
            {!loading && trimmedQuery && visibleResults.length === 0 && (
              <p className="px-3 py-3 text-xs text-foreground/50">
                「{query}」に一致する銘柄が見つかりません
              </p>
            )}
            {!loading &&
              visibleResults.map((result) => (
                <button
                  key={result.stockCode}
                  type="button"
                  onClick={() => goToStock(result.stockCode)}
                  className="flex w-full items-center justify-between px-3 py-2.5 text-left text-sm text-brand-navy hover:bg-section-tint"
                >
                  <span className="truncate">{result.stockName}</span>
                  <span className="ml-2 shrink-0 font-mono text-xs text-foreground/50">
                    {result.stockCode}
                  </span>
                </button>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
