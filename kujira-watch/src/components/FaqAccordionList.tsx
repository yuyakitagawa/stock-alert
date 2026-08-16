"use client";

import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import type { FaqItem } from "@/lib/faqData";

// 1カテゴリ分のQ&Aをアコーディオンで並べる。
//
// 以前は全502件を1ページに置き、タブ(MUI Tabs)でカテゴリ絞り込みをしていたが、
// HTMLが1.57MB(gzip 241KB)に膨らんでいた。ページをカテゴリ別に分けたことで
// 「絞り込み」自体がURLの役割になり、タブとクライアント状態が不要になった。
export default function FaqAccordionList({ faqs }: { faqs: FaqItem[] }) {
  return (
    <Box>
      {faqs.map((faq) => (
        <Accordion
          key={faq.question}
          disableGutters
          elevation={0}
          square
          sx={{ bgcolor: "transparent", "&::before": { display: "none" } }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon fontSize="small" />} sx={{ px: 0 }}>
            <Typography component="h3" sx={{ fontWeight: 600, color: "primary.main" }}>
              {faq.question}
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ px: 0 }}>
            <Typography variant="body2" sx={{ lineHeight: 1.7, color: "text.secondary" }}>
              {faq.render ?? faq.answer}
            </Typography>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
}
