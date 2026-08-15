"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import MenuIcon from "@mui/icons-material/Menu";
import CloseIcon from "@mui/icons-material/Close";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { SITE_NAME, SITE_NAME_EN } from "@/lib/site";
import { DEAL_TYPE_EN, EN_SLUG_TO_DEAL_TYPE } from "@/lib/dealTypeInfo";
import type { DealType } from "@/types/article";
import { UI, type Locale } from "@/lib/i18n";

type MenuLink = {
  href: string;
  label: string;
};

// 現在のパスから、もう一方のロケールの等価URLを計算する。
// 記事(id)・銘柄(code)はロケール間で共通のキーなのでprefixの付け替えだけで対応できるが、
// カテゴリだけは日本語文字列⇄英語slugの変換が要る。/weekly, /investors, /date, /faq は
// 英語版が無いためホームにフォールバックする。
function alternatePath(pathname: string): string {
  const enArticle = pathname.match(/^\/en\/articles\/(.+)$/);
  if (enArticle) return `/articles/${enArticle[1]}`;
  const jaArticle = pathname.match(/^\/articles\/(.+)$/);
  if (jaArticle) return `/en/articles/${jaArticle[1]}`;

  const enStock = pathname.match(/^\/en\/stocks\/(.+)$/);
  if (enStock) return `/stocks/${enStock[1]}`;
  const jaStock = pathname.match(/^\/stocks\/(.+)$/);
  if (jaStock) return `/en/stocks/${jaStock[1]}`;

  const enCategory = pathname.match(/^\/en\/category\/([^/]+)$/);
  if (enCategory) {
    const dealType = EN_SLUG_TO_DEAL_TYPE[enCategory[1]];
    return dealType ? `/category/${encodeURIComponent(dealType)}` : "/";
  }
  const jaCategory = pathname.match(/^\/category\/([^/]+)$/);
  if (jaCategory) {
    const dealType = decodeURIComponent(jaCategory[1]) as DealType;
    const info = DEAL_TYPE_EN[dealType];
    return info ? `/en/category/${info.slug}` : "/en";
  }

  if (pathname === "/en/about") return "/about";
  if (pathname === "/about") return "/en/about";

  return pathname.startsWith("/en") ? "/" : "/en";
}

// オートスクロールで記事一覧が際限なく伸びるため、ページ最下部のフッターまで
// スクロールしてたどり着くのが実質困難になった。運営者情報・免責事項等は
// ヘッダー右上のハンバーガーメニューに集約し、常にアクセスできるようにする。
export default function HeaderMenu({ locale = "ja" }: { locale?: Locale }) {
  const [open, setOpen] = useState(false);
  const year = new Date().getFullYear();
  const pathname = usePathname();
  const t = UI[locale];

  const close = () => setOpen(false);

  const siteLinks: MenuLink[] =
    locale === "en"
      ? [
          { href: "/en", label: "Top" },
          { href: "/en/about", label: t.aboutMenuLabel },
        ]
      : [
          { href: "/", label: "TOP" },
          { href: "/weekly", label: "今週のまとめ" },
          { href: "/ranking", label: "投資家勝率ランキング" },
          { href: "/investors", label: "投資家一覧" },
          { href: "/stocks", label: "銘柄一覧" },
          { href: "/about", label: "運営者情報・免責事項" },
          { href: "/faq", label: "よくある質問" },
          { href: "/feed.xml", label: "RSSフィード" },
        ];

  const currentPath = pathname ?? (locale === "en" ? "/en" : "/");
  const otherHref = alternatePath(currentPath);
  const jaHref = locale === "ja" ? currentPath : otherHref;
  const enHref = locale === "en" ? currentPath : otherHref;

  return (
    // PCでは中央寄せの本文カラムではなく、画面の本当の右端にハンバーガーを固定する
    // (ヘッダーはposition:stickyでabsolute配置の基準になる)
    <Box sx={{ flexShrink: 0, position: { sm: "absolute" }, right: { sm: 16 }, top: { sm: 16 } }}>
      <IconButton
        aria-label={t.menuLabel}
        aria-expanded={open}
        onClick={() => setOpen(true)}
        size="small"
        sx={{ color: "primary.main" }}
      >
        <MenuIcon fontSize="small" />
      </IconButton>

      <Drawer anchor="right" open={open} onClose={close}>
        <Box sx={{ width: "86vw", maxWidth: 320, height: "100%", overflowY: "auto" }} role="presentation">
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: 1, borderColor: "divider", px: 2, py: 1.5 }}>
            <Typography variant="subtitle2" sx={{ display: "flex", alignItems: "center", gap: 1, fontWeight: 700, color: "primary.main" }}>
              <Box component="span" aria-hidden>
                🐋
              </Box>
              {locale === "en" ? SITE_NAME_EN : SITE_NAME}
            </Typography>
            <IconButton
              aria-label={locale === "en" ? "Close menu" : "メニューを閉じる"}
              onClick={close}
              size="small"
              sx={{ color: "primary.main" }}
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>

          <Typography variant="overline" sx={{ display: "block", bgcolor: "action.hover", px: 2, py: 0.75, color: "text.secondary" }}>
            {t.langSectionLabel}
          </Typography>
          <List component="nav" aria-label={t.langSectionLabel} disablePadding>
            <ListItemButton component={Link} href={jaHref} onClick={close} divider selected={locale === "ja"}>
              <ListItemText
                primary="日本語"
                slotProps={{ primary: { sx: { fontWeight: locale === "ja" ? 700 : 400, color: locale === "ja" ? "brand.blue" : "primary.main" } } }}
              />
              <ChevronRightIcon fontSize="small" sx={{ color: "action.disabled" }} />
            </ListItemButton>
            <ListItemButton component={Link} href={enHref} onClick={close} divider selected={locale === "en"}>
              <ListItemText
                primary="English"
                slotProps={{ primary: { sx: { fontWeight: locale === "en" ? 700 : 400, color: locale === "en" ? "brand.blue" : "primary.main" } } }}
              />
              <ChevronRightIcon fontSize="small" sx={{ color: "action.disabled" }} />
            </ListItemButton>
          </List>

          <Typography variant="overline" sx={{ display: "block", bgcolor: "action.hover", px: 2, py: 0.75, color: "text.secondary" }}>
            {locale === "en" ? "Menu" : "メニュー"}
          </Typography>
          <List component="nav" aria-label={locale === "en" ? "Menu" : "メニュー"} disablePadding>
            {siteLinks.map((link) => (
              <ListItemButton key={link.href} component={Link} href={link.href} onClick={close} divider>
                <ListItemText primary={link.label} slotProps={{ primary: { sx: { color: "primary.main" } } }} />
                <ChevronRightIcon fontSize="small" sx={{ color: "action.disabled" }} />
              </ListItemButton>
            ))}
          </List>

          <Typography variant="caption" sx={{ display: "block", mt: 1.5, px: 2, lineHeight: 1.6, color: "text.secondary" }}>
            {locale === "en"
              ? "This site is commentary based on public EDINET large-shareholding filings and is not investment advice."
              : "本サイトはEDINET大量保有報告書等の公開情報をもとにした解説であり、投資助言ではありません。"}
          </Typography>
          <Divider sx={{ my: 1 }} />
          <Typography variant="caption" sx={{ display: "block", px: 2, pb: 2, color: "text.disabled" }}>
            © {year} {locale === "en" ? SITE_NAME_EN : SITE_NAME}
          </Typography>
        </Box>
      </Drawer>
    </Box>
  );
}
