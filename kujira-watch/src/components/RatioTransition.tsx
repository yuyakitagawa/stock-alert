import Box from "@mui/material/Box";

// 「前回% → 今回%」に増減の向き（▲緑/▼赤）を添える保有比率表示。
// /stocks/[code]の推移テーブルと/investors/[filer]の取引テーブルで共用する。
export default function RatioTransition({
  ratio,
  prior,
}: {
  ratio: number | null;
  prior: number | null;
}) {
  if (ratio === null) return <span>-</span>;
  const diff = prior !== null ? Math.round((ratio - prior) * 100) / 100 : null;
  return (
    <span>
      {prior !== null && `${prior}% → `}
      {ratio}%
      {diff !== null && diff !== 0 && (
        <Box
          component="span"
          sx={{ ml: 0.5, fontSize: "0.75rem", color: diff > 0 ? "success.main" : "error.main" }}
        >
          {diff > 0 ? "▲" : "▼"}{Math.abs(diff)}pt
        </Box>
      )}
    </span>
  );
}
