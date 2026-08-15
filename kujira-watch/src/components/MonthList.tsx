"use client";

import Link from "next/link";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { formatDealAmount, formatMonth } from "@/lib/format";
import type { MonthSummary } from "@/lib/microcms";

export default function MonthList({ months }: { months: MonthSummary[] }) {
  return (
    <List disablePadding sx={{ borderTop: 1, borderBottom: 1, borderColor: "divider" }}>
      {months.map((m) => (
        <ListItemButton
          key={m.month}
          component={Link}
          href={`/monthly/${m.month}`}
          divider
          sx={{ py: 2, px: 0, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 2 }}
        >
          <Box>
            <Typography sx={{ fontWeight: 700, color: "primary.main" }}>{formatMonth(m.month)}</Typography>
            <Typography variant="body2" sx={{ mt: 0.25, color: "text.secondary" }}>
              {m.count}件・{formatDealAmount(m.amount)}
            </Typography>
          </Box>
          <Typography variant="overline" sx={{ flexShrink: 0, color: "brand.blue" }}>
            この月のまとめを見る ›
          </Typography>
        </ListItemButton>
      ))}
    </List>
  );
}
