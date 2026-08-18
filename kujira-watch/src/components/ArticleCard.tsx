"use client";

import Image from "next/image";
import Link from "next/link";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import type { ArticleContent } from "@/types/article";
import { formatDate, formatDealAmountOrCorrection } from "@/lib/format";
import type { Locale } from "@/lib/i18n";
import DealDirectionBadge from "./DealDirectionBadge";
import DealTypeBadge from "./DealTypeBadge";

export default function ArticleCard({
  article,
  locale = "ja",
}: {
  article: ArticleContent;
  locale?: Locale;
}) {
  const href = locale === "en" ? `/en/articles/${article.id}` : `/articles/${article.id}`;
  const title = locale === "en" ? article.titleEn ?? article.title : article.title;
  return (
    <Card
      variant="outlined"
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        borderColor: "divider",
        transition: "border-color 0.15s ease-out",
        "&:hover": { borderColor: "brand.gold" },
      }}
    >
      <CardActionArea
        component={Link}
        href={href}
        sx={{ display: "flex", flexDirection: "column", alignItems: "stretch", height: "100%", flexGrow: 1 }}
      >
        {article.eyecatch && (
          <CardMedia sx={{ position: "relative", aspectRatio: "16 / 9", bgcolor: "action.hover" }}>
            <Image
              src={article.eyecatch.url}
              alt={article.eyecatch.alt || title}
              fill
              style={{ objectFit: "cover" }}
              sizes="(min-width: 640px) 50vw, 100vw"
            />
          </CardMedia>
        )}
        <CardContent sx={{ flexGrow: 1, width: "100%" }}>
          <Stack direction="row" sx={{ mb: 1, flexWrap: "wrap", alignItems: "center", columnGap: 1.5, rowGap: 0.5 }}>
            <DealTypeBadge dealType={article.dealType} locale={locale} />
            <DealDirectionBadge tags={article.tags} locale={locale} />
            <Typography variant="overline" sx={{ color: "text.disabled" }}>
              {formatDate(article.dealDate, locale)}
            </Typography>
          </Stack>
          <Typography variant="h6" component="h2" sx={{ fontWeight: 700, lineHeight: 1.35, color: "primary.main" }}>
            {title}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1, color: "text.secondary" }}>
            {locale === "en"
              ? `${article.stockName} (${article.stockCode}) · ${formatDealAmountOrCorrection(article, locale)}`
              : `${article.stockName}（${article.stockCode}） ・ ${formatDealAmountOrCorrection(article, locale)}`}
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}
