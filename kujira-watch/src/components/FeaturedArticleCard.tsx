"use client";

import Image from "next/image";
import Link from "next/link";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import type { ArticleContent } from "@/types/article";
import { excerptFromHtml, formatDate, formatDealAmount } from "@/lib/format";
import { UI, type Locale } from "@/lib/i18n";
import DealDirectionBadge from "./DealDirectionBadge";
import DealTypeBadge from "./DealTypeBadge";

export default function FeaturedArticleCard({
  article,
  rank,
  locale = "ja",
}: {
  article: ArticleContent;
  rank: number;
  locale?: Locale;
}) {
  const t = UI[locale];
  const hasImage = Boolean(article.eyecatch);
  const href = locale === "en" ? `/en/articles/${article.id}` : `/articles/${article.id}`;
  const title = locale === "en" ? article.titleEn ?? article.title : article.title;
  const body = locale === "en" ? article.bodyEn ?? article.body : article.body;
  return (
    <Card
      elevation={4}
      sx={{
        position: "relative",
        mb: 5,
        minHeight: hasImage ? { xs: 320, sm: 352 } : "auto",
        bgcolor: "primary.main",
        color: "common.white",
        overflow: "hidden",
      }}
    >
      <CardActionArea
        component={Link}
        href={href}
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: hasImage ? "flex-end" : "flex-start",
          alignItems: "stretch",
          height: "100%",
          minHeight: "inherit",
          "&:hover .featured-media": { transform: "scale(1.05)" },
        }}
      >
        {article.eyecatch && (
          <Box sx={{ position: "absolute", inset: 0 }}>
            <Image
              src={article.eyecatch.url}
              alt={article.eyecatch.alt || title}
              fill
              priority
              className="featured-media"
              style={{ objectFit: "cover", opacity: 0.5, transition: "transform 0.3s" }}
              sizes="100vw"
            />
            <Box
              sx={{
                position: "absolute",
                inset: 0,
                // primary.main(#16213a)に合わせた固定値。sxでのtheme callbackはRSC境界を越えて
                // 関数を渡せないため使えない（Server ComponentからClient Componentへのprops）。
                // アイキャッチには白太字の事実テキストが焼き込まれているため、カード側の
                // タイトル・抜粋が重なる下半分は不透明に近いスクリムで覆い、二重表示を防ぐ。
                background:
                  "linear-gradient(to top, #16213a 30%, rgba(22, 33, 58, 0.88) 62%, rgba(22, 33, 58, 0.35))",
              }}
            />
          </Box>
        )}
        <Box sx={{ position: "relative", p: { xs: 3, sm: 5 } }}>
          <Stack direction="row" sx={{ mb: 2, flexWrap: "wrap", alignItems: "center", columnGap: 2, rowGap: 0.5 }}>
            <Typography variant="overline" sx={{ color: "brand.goldBright" }}>
              {t.featuredRankLabels[rank] ?? t.featuredFallback}
            </Typography>
            <DealTypeBadge dealType={article.dealType} locale={locale} onDark />
            <DealDirectionBadge tags={article.tags} locale={locale} onDark />
            <Typography variant="overline" sx={{ color: "rgba(255,255,255,0.6)" }}>
              {formatDate(article.dealDate, locale)}
            </Typography>
          </Stack>
          <Typography variant="h4" component="h2" sx={{ fontWeight: 700, lineHeight: 1.3, fontSize: { xs: "1.25rem", sm: "1.5rem" } }}>
            {title}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1.5, color: "rgba(255,255,255,0.8)" }}>
            {locale === "en"
              ? `${article.stockName} (${article.stockCode}) · ${formatDealAmount(article.dealAmount, locale)}`
              : `${article.stockName}（${article.stockCode}） ・ ${formatDealAmount(article.dealAmount, locale)}`}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1.5, maxWidth: 640, color: "rgba(255,255,255,0.7)" }}>
            {excerptFromHtml(body, 90)}
          </Typography>
        </Box>
      </CardActionArea>
    </Card>
  );
}
