"use client";

import { useState } from "react";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

export type FaqCategory = { id: string; label: string };

export type FaqItem = {
  question: string;
  answer: string;
  category: string;
  render?: React.ReactNode;
};

// カテゴリごとに色分けした質問バッジ用。9カテゴリ固定のため、増減時はここも見直す。
const CATEGORY_COLORS: Record<string, string> = {
  basics: "brand.blue",
  terms: "#7c3aed",
  "term-supplement": "#4f46e5",
  "regulation-deep-dive": "#e11d48",
  "investor-players": "#0891b2",
  "investing-basics-extra": "#d97706",
  usage: "#059669",
  howto: "brand.gold",
  "about-site": "rgba(32, 29, 26, 0.5)",
};

// カテゴリタブは見た目上の絞り込みのみで、初期状態は必ず「すべて」（全件表示）。
// クローラー・AIOがJS実行前に読むSSR済みHTMLには常に全問が含まれるようにするため。
export default function FaqList({
  faqs,
  categories,
}: {
  faqs: FaqItem[];
  categories: FaqCategory[];
}) {
  const [active, setActive] = useState<string>("all");
  const filtered = active === "all" ? faqs : faqs.filter((faq) => faq.category === active);
  const categoryLabel = (id: string) =>
    categories.find((category) => category.id === id)?.label ?? id;

  return (
    <Box>
      <Tabs
        value={active}
        onChange={(_, value) => setActive(value)}
        variant="scrollable"
        scrollButtons="auto"
        allowScrollButtonsMobile
        aria-label="カテゴリで絞り込み"
        sx={{
          mb: 3,
          borderBottom: 1,
          borderColor: "divider",
          minHeight: 40,
          "& .MuiTab-root": {
            minHeight: 40,
            minWidth: "auto",
            px: 0,
            mr: 3,
            fontSize: "0.6875rem",
            fontWeight: 700,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "text.secondary",
          },
          "& .Mui-selected": { color: "primary.main" },
          "& .MuiTabs-indicator": { bgcolor: "brand.gold" },
        }}
      >
        <Tab value="all" label={`すべて（${faqs.length}）`} />
        {categories.map((category) => {
          const count = faqs.filter((faq) => faq.category === category.id).length;
          return <Tab key={category.id} value={category.id} label={`${category.label}（${count}）`} />;
        })}
      </Tabs>
      <Box>
        {filtered.map((faq) => (
          <Accordion key={faq.question} disableGutters elevation={0} square sx={{ bgcolor: "transparent", "&::before": { display: "none" } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon fontSize="small" />} sx={{ px: 0 }}>
              <Box>
                <Typography
                  variant="overline"
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 0.75,
                    color: CATEGORY_COLORS[faq.category] ?? "text.secondary",
                  }}
                >
                  <Box component="span" aria-hidden sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: "currentColor" }} />
                  {categoryLabel(faq.category)}
                </Typography>
                <Typography sx={{ fontWeight: 600, color: "primary.main" }}>{faq.question}</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 0 }}>
              <Typography variant="body2" sx={{ lineHeight: 1.7, color: "text.secondary" }}>
                {faq.render ?? faq.answer}
              </Typography>
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>
    </Box>
  );
}
