"use client";

import Box from "@mui/material/Box";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import CircularProgress from "@mui/material/CircularProgress";
import type { SearchOption } from "./StockSearch";
import { UI, type Locale } from "@/lib/i18n";

// 検索パネルの中身。虫眼鏡をタップするまで表示されないので、StockSearch側から
// next/dynamicで遅延読み込みする。MUIのAutocomplete/TextFieldはMUIの中でも重い部類で、
// 「閉じているのが既定」なのに全ページの初期JSに積まれていた。
// 見た目・挙動は据え置きで、読み込みのタイミングだけ後ろにずらす。
export default function StockSearchPanel({
  query,
  results,
  loading,
  locale,
  onQueryChange,
  onSelect,
}: {
  query: string;
  results: SearchOption[];
  loading: boolean;
  locale: Locale;
  onQueryChange: (value: string) => void;
  onSelect: (option: SearchOption) => void;
}) {
  const t = UI[locale];
  const trimmedQuery = query.trim();

  return (
    <Autocomplete
      open
      disablePortal
      disableCloseOnSelect={false}
      filterOptions={(x) => x}
      options={trimmedQuery ? results : []}
      loading={loading}
      groupBy={(option) =>
        option.type === "stock" ? t.searchGroupStocks : t.searchGroupInvestors
      }
      getOptionLabel={(option) =>
        option.type === "stock" ? option.stockName : option.filerName
      }
      noOptionsText={trimmedQuery ? t.searchNoResults(query) : t.searchPlaceholder}
      loadingText={t.searchLoading}
      onInputChange={(_, value) => onQueryChange(value)}
      onChange={(_, value) => {
        if (value) onSelect(value);
      }}
      onClose={(_, reason) => {
        if (reason === "selectOption") return;
      }}
      renderOption={(props, option) => {
        const { key, ...rest } = props;
        return (
          <Box
            component="li"
            key={key ?? (option.type === "stock" ? option.stockCode : option.filerName)}
            {...rest}
            sx={{ display: "flex", justifyContent: "space-between" }}
          >
            <span>{option.type === "stock" ? option.stockName : option.filerName}</span>
            {option.type === "stock" && (
              <Box
                component="span"
                sx={{
                  ml: 1,
                  flexShrink: 0,
                  fontFamily: "var(--font-geist-mono)",
                  fontSize: 12,
                  color: "text.secondary",
                }}
              >
                {option.stockCode}
              </Box>
            )}
          </Box>
        );
      }}
      renderInput={(params) => (
        <TextField
          {...params}
          autoFocus
          placeholder={t.searchPlaceholder}
          size="small"
          sx={{ bgcolor: "background.paper" }}
          slotProps={{
            ...params.slotProps,
            input: {
              ...params.slotProps.input,
              endAdornment: (
                <>
                  {loading ? <CircularProgress color="inherit" size={16} /> : null}
                  {params.slotProps.input.endAdornment}
                </>
              ),
            },
          }}
        />
      )}
    />
  );
}
