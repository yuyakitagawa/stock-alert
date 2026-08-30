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
import { formatDate, formatDealAmountOrCorrection, toDateAttr } from "@/lib/format";
import DealDirectionBadge from "./DealDirectionBadge";
import DealTypeBadge from "./DealTypeBadge";

// 見出しレベルは置かれる文脈で変わる（セクション見出しの下ならh3、取引日グループの
// 下ならh4）。既定は取引日ページのようにh1直下へ並ぶ場合のh2。
export default function ArticleCard({
  article,
  headingLevel = "h2",
}: {
  article: ArticleContent;
  headingLevel?: "h2" | "h3" | "h4";
}) {
  const href = `/articles/${article.id}`;
  const title = article.title;
  return (
    // カード1枚＝それ単体で意味が通る記事の要約なのでarticle要素で出す。
    <Card component="article" sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
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
            <DealTypeBadge dealType={article.dealType} />
            <DealDirectionBadge tags={article.tags} />
            <Typography
              variant="overline"
              component="time"
              dateTime={toDateAttr(article.dealDate)}
              sx={{ color: "text.disabled" }}
            >
              {formatDate(article.dealDate)}
            </Typography>
          </Stack>
          <Typography variant="h6" component={headingLevel} sx={{ color: "primary.main" }}>
            {title}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1, color: "text.secondary" }}>
            {`${article.stockName}（${article.stockCode}） ・ ${formatDealAmountOrCorrection(article)}`}
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}
