import Link from "next/link";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import { displayFilerName } from "@/lib/format";
import type { TrendingEntry } from "@/lib/trendingStats";

// 期間比較（直近N日 vs 前N日）の増加件数ランキング表。
// /trending（銘柄）と/ranking/trending（投資家）で共用する。
export default function TrendingTable({
  entries,
  headLabel,
  windowDays,
  hrefOf,
}: {
  entries: TrendingEntry[];
  headLabel: string;
  windowDays: number;
  hrefOf: (entry: TrendingEntry) => string | null;
}) {
  return (
    <TableContainer>
      <Table size="small" sx={{ minWidth: 420, "& .MuiTableCell-root": { borderColor: "divider" } }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ color: "text.secondary" }}>{headLabel}</TableCell>
            <TableCell align="right" sx={{ color: "text.secondary" }}>直近{windowDays}日</TableCell>
            <TableCell align="right" sx={{ color: "text.secondary" }}>前{windowDays}日</TableCell>
            <TableCell align="right" sx={{ color: "text.secondary" }}>増加</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {entries.map((entry) => {
            const href = hrefOf(entry);
            return (
              <TableRow key={entry.key}>
                <TableCell>
                  {href ? (
                    <Link href={href} className="text-brand-blue hover:underline">
                      {displayFilerName(entry.label)}
                    </Link>
                  ) : (
                    <span className="text-foreground/80">{displayFilerName(entry.label)}</span>
                  )}
                  {entry.isNew && (
                    <span className="kicker ml-2 whitespace-nowrap text-brand-gold">NEW</span>
                  )}
                </TableCell>
                <TableCell align="right" sx={{ fontWeight: 700, color: "primary.main" }}>
                  {entry.count}件
                </TableCell>
                <TableCell align="right" sx={{ color: "text.disabled" }}>
                  {entry.prevCount}件
                </TableCell>
                <TableCell align="right" sx={{ color: "text.secondary" }}>+{entry.delta}件</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
