"use client";

import Link from "next/link";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { CATEGORIES } from "@/types/article";

// TOPページの見出し直下に置くカテゴリ絞り込み。以前はHeader(全ページ共通)に
// 常設していたが、日付一覧の直前に移した方が「見出し→フィルター→一覧」の
// 導線として分かりやすいため専用コンポーネントに切り出した。
export default function CategoryFilterDetails() {
  return (
    <Accordion
      disableGutters
      elevation={0}
      square
      sx={{
        mb: 4,
        bgcolor: "transparent",
        borderBottom: 1,
        borderColor: "divider",
        "&::before": { display: "none" },
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon fontSize="small" />}
        sx={{ px: 0, minHeight: "auto", "& .MuiAccordionSummary-content": { my: 1 } }}
      >
        <Typography variant="overline" sx={{ color: "text.secondary" }}>
          カテゴリで絞り込む
        </Typography>
      </AccordionSummary>
      <AccordionDetails sx={{ px: 0, pb: 2 }}>
        <Stack direction="row" component="nav" aria-label="カテゴリ" sx={{ flexWrap: "wrap", columnGap: 2, rowGap: 0.75 }}>
          {CATEGORIES.map((category) => (
            <Typography
              key={category}
              component={Link}
              href={`/category/${encodeURIComponent(category)}`}
              variant="overline"
              sx={{ color: "text.secondary", textDecoration: "none", "&:hover": { color: "primary.main" } }}
            >
              {category}
            </Typography>
          ))}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
}
