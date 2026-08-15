"use client";

import Link from "next/link";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { formatDealAmount } from "@/lib/format";

type DayEntry = {
  date: string;
  label: string;
  count: number;
  amount: number;
};

// /weekly下部の「日別の記事一覧」。開示が無い日はリンクにせずグレーアウトで示す。
export default function DailyDigestList({ days }: { days: DayEntry[] }) {
  return (
    <List disablePadding sx={{ borderTop: 1, borderBottom: 1, borderColor: "divider" }}>
      {days.map((day) =>
        day.count > 0 ? (
          <ListItemButton
            key={day.date}
            component={Link}
            href={`/date/${day.date}`}
            divider
            sx={{ py: 2, px: 0, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 2 }}
          >
            <Box>
              <Typography sx={{ fontWeight: 700, color: "primary.main" }}>{day.label}</Typography>
              <Typography variant="body2" sx={{ mt: 0.25, color: "text.secondary" }}>
                {day.count}件・{formatDealAmount(day.amount)}
              </Typography>
            </Box>
            <Typography variant="overline" sx={{ flexShrink: 0, color: "brand.blue" }}>
              この日の記事を見る ›
            </Typography>
          </ListItemButton>
        ) : (
          <Box key={day.date} sx={{ py: 2, display: "flex", justifyContent: "space-between", borderBottom: 1, borderColor: "divider" }}>
            <Box>
              <Typography sx={{ fontWeight: 700, color: "text.disabled" }}>{day.label}</Typography>
              <Typography variant="body2" sx={{ mt: 0.25, color: "text.disabled" }}>開示なし</Typography>
            </Box>
          </Box>
        )
      )}
    </List>
  );
}
