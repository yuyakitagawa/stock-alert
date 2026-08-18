"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import SearchIcon from "@mui/icons-material/Search";
import type { StockSearchResult } from "@/lib/microcms";
import { UI, type Locale } from "@/lib/i18n";

// 企業名・証券コード・投資家名で検索し、選択(またはEnter)で /stocks/[code] または
// /investors/[filer] に遷移する。
// レスポンスを都度撃たないよう、入力停止から300ms待ってからAPIを叩く。
const DEBOUNCE_MS = 300;

// 検索結果の1件。銘柄と投資家(EDINET提出者)を同じドロップダウンにグループ表示する。
export type SearchOption =
  | { type: "stock"; stockCode: string; stockName: string }
  | { type: "investor"; filerName: string };

// パネル(Autocomplete/TextField/CircularProgress)は虫眼鏡をタップするまで表示されない。
// MUIの中でも重い部類なので、初期JSから外して開いたときに読み込む。
// ssr:false なのは、閉じている状態のHTMLにパネルの分の出力を含める意味が無いため。
const StockSearchPanel = dynamic(() => import("./StockSearchPanel"), { ssr: false });

export default function StockSearch({ locale = "ja" }: { locale?: Locale }) {
  const t = UI[locale];
  const router = useRouter();
  const [open, setOpen] = useState(false);
  // Autocompleteは遅延読み込みなので、読み込み終わるまでの数十msは平常時の素のinputを
  // 出したままにする。先に消すと、その間のキー入力が行き場を失って落ちる。
  const [panelReady, setPanelReady] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchOption[]>([]);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const close = () => {
    setOpen(false);
    setPanelReady(false);
    setQuery("");
    setResults([]);
  };

  // パネル(Autocomplete)がマウントされた合図。autoFocusで入力欄のフォーカスも
  // そちらへ移るので、このタイミングで素のinputを引っ込める。
  const handlePanelReady = useCallback(() => setPanelReady(true), []);

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
        .then((res) => (res.ok ? res.json() : { results: [], investors: [] }))
        .then((data) => {
          const stocks: SearchOption[] = ((data.results ?? []) as StockSearchResult[]).map(
            (r) => ({ type: "stock", ...r })
          );
          // 英語版には投資家ページ(/investors)が無いため、投資家の結果は日本語版のみ表示する。
          const investors: SearchOption[] =
            locale === "en"
              ? []
              : ((data.investors ?? []) as { filerName: string }[]).map((i) => ({
                  type: "investor",
                  filerName: i.filerName,
                }));
          setResults([...stocks, ...investors]);
        })
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [trimmedQuery, locale]);

  const goTo = (option: SearchOption) => {
    close();
    if (option.type === "stock") {
      router.push(locale === "en" ? `/en/stocks/${option.stockCode}` : `/stocks/${option.stockCode}`);
    } else {
      router.push(`/investors/${encodeURIComponent(option.filerName)}`);
    }
  };

  return (
    <Box ref={containerRef} sx={{ position: "relative", flexShrink: 0 }}>
      {/* モバイルはヘッダーの横幅をロゴに使いたいので虫眼鏡のまま。
          PC(md以上)は常時表示の検索窓にする。 */}
      <IconButton
        aria-label={t.searchAria}
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        size="small"
        sx={{ display: { md: "none" }, color: "primary.main" }}
      >
        <SearchIcon fontSize="small" />
      </IconButton>

      {/* PCの平常時は素のinput。フォーカスした時点でAutocomplete(StockSearchPanel)を
          読み込んで差し替えるので、全ページの初期JSにMUIのAutocompleteを積まずに済む
          （検索窓を常時表示しても初期JSは今までと同じ）。 */}
      {!(open && panelReady) && (
        <Box
          sx={{
            display: { xs: "none", md: "flex" },
            alignItems: "center",
            gap: 0.75,
            width: 220,
            px: 1,
            py: 0.5,
            border: 1,
            borderColor: "divider",
            borderRadius: 1,
            bgcolor: "background.paper",
          }}
        >
          <SearchIcon fontSize="small" sx={{ color: "text.secondary" }} />
          <Box
            component="input"
            type="search"
            aria-label={t.searchAria}
            placeholder={t.searchPlaceholder}
            value={query}
            onFocus={() => setOpen(true)}
            onChange={(event: ChangeEvent<HTMLInputElement>) => {
              setQuery(event.target.value);
              setOpen(true);
            }}
            sx={{
              width: "100%",
              border: 0,
              outline: 0,
              bgcolor: "transparent",
              color: "text.primary",
              fontFamily: "inherit",
              fontSize: "0.8125rem",
            }}
          />
        </Box>
      )}

      {open && (
        <Box
          sx={{
            position: "absolute",
            right: 0,
            // モバイルは虫眼鏡の下に落とす。PCは平常時の検索窓と同じ位置に重ねる。
            top: { xs: 44, md: 0 },
            zIndex: (theme) => theme.zIndex.appBar + 1,
            width: { xs: 288, sm: 320 },
          }}
        >
          <StockSearchPanel
            query={query}
            results={results}
            loading={loading}
            locale={locale}
            onQueryChange={setQuery}
            onSelect={goTo}
            onReady={handlePanelReady}
          />
        </Box>
      )}
    </Box>
  );
}
