import Box from "@mui/material/Box";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import { formatAmountOku, type BuybackRow } from "@/lib/buybacks";
import { formatDate } from "@/lib/format";

// /stocks/[code] の「自社株買い（TDnet適時開示）」セクション。
// 決定開示には取得枠（上限金額・比率・期間）を、進捗報告は開示タイトルのみを出す。
// 決定の上限金額が取れている行は太字にして、一覧の中で「新しい取得枠」が目に入るようにする。
export default function BuybackHistory({ rows }: { rows: BuybackRow[] }) {
  if (rows.length === 0) return null;
  const decisions = rows.filter((r) => r.kind === "決定");
  const totalYen = decisions.reduce((sum, r) => sum + (r.maxAmountYen ?? 0), 0);
  const summary = [
    `開示${rows.length}件`,
    decisions.length > 0 ? `取得枠の決定${decisions.length}件` : null,
    totalYen > 0 ? `上限合計${formatAmountOku(totalYen)}` : null,
  ]
    .filter(Boolean)
    .join("・");

  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 700, color: "primary.main" }}>
        自社株買い（TDnet適時開示）
      </Typography>
      <Typography variant="caption" sx={{ display: "block", mb: 1, color: "text.secondary" }}>
        {summary}。上限は取締役会で決議した取得枠で、実際の取得額はこれを下回ることがあります。
      </Typography>
      <TableContainer sx={{ borderTop: 1, borderColor: "divider" }}>
        <Table size="small" sx={{ minWidth: 560, "& .MuiTableCell-root": { borderColor: "divider" } }}>
          <TableHead>
            <TableRow>
              <TableCell sx={{ color: "text.disabled" }}>開示日</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>種別</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>取得枠（上限）</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>内容</TableCell>
              <TableCell sx={{ color: "text.disabled" }}>原文</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((r) => {
              const amount = formatAmountOku(r.maxAmountYen);
              const frame = [
                amount,
                r.ratioPct !== null ? `発行済の${r.ratioPct}%` : null,
                r.maxShares !== null ? `${r.maxShares.toLocaleString("ja-JP")}株` : null,
              ]
                .filter(Boolean)
                .join(" / ");
              const period =
                r.periodFrom && r.periodTo
                  ? r.periodFrom === r.periodTo
                    ? formatDate(r.periodFrom)
                    : `${formatDate(r.periodFrom)}〜${formatDate(r.periodTo)}`
                  : null;
              const detail = [r.method, period, r.willCancel ? "消却予定" : null]
                .filter(Boolean)
                .join("・");
              const isDecision = r.kind === "決定";
              return (
                <TableRow key={`${r.disclosedAt}-${r.title}`}>
                  <TableCell sx={{ whiteSpace: "nowrap", color: "text.secondary" }}>
                    {formatDate(r.disclosedAt.slice(0, 10))}
                  </TableCell>
                  <TableCell
                    sx={{
                      whiteSpace: "nowrap",
                      color: isDecision ? "primary.main" : "text.secondary",
                      fontWeight: isDecision ? 700 : 400,
                    }}
                  >
                    {r.kind}
                  </TableCell>
                  <TableCell
                    sx={{
                      whiteSpace: "nowrap",
                      fontWeight: amount ? 700 : 400,
                      color: amount ? "text.primary" : "text.disabled",
                    }}
                  >
                    {frame || "-"}
                  </TableCell>
                  <TableCell sx={{ color: "text.secondary" }}>
                    <span className="block text-xs">{r.title}</span>
                    {detail && <span className="block text-xs text-ink-tertiary">{detail}</span>}
                  </TableCell>
                  <TableCell sx={{ whiteSpace: "nowrap" }}>
                    {r.docUrl && (
                      <a
                        href={r.docUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-brand-blue hover:underline"
                      >
                        PDF ↗
                      </a>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
