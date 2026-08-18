"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import dynamic from "next/dynamic";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import type { MenuLink } from "./HeaderMenuDrawer";
import { DEAL_TYPE_EN, EN_SLUG_TO_DEAL_TYPE } from "@/lib/dealTypeInfo";
import type { DealType } from "@/types/article";
import { UI, type Locale } from "@/lib/i18n";
import { X_PROFILE_URL, X_SCREEN_NAME, YOUTUBE_CHANNEL_URL } from "@/lib/site";

// ドロワー本体(Drawer/List/Divider/…)は開くまで要らない。MUI DrawerはModal/Portal/
// Backdrop/Slide一式を引き連れており、閉じているのが既定なのに全ページの初期JSへ
// 積まれていた。開いたときに読み込む。
const HeaderMenuDrawer = dynamic(() => import("./HeaderMenuDrawer"), { ssr: false });

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

  if (pathname === "/en/privacy") return "/privacy";
  if (pathname === "/privacy") return "/en/privacy";

  return pathname.startsWith("/en") ? "/" : "/en";
}

// オートスクロールで記事一覧が際限なく伸びるページでは最下部のフッターまで
// スクロールしてたどり着けないため、主要リンクはヘッダー右上のハンバーガーメニュー
// からも常にアクセスできるようにする（フッター(Footer.tsx)とは併存させる）。
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
          { href: "/en/privacy", label: t.privacyMenuLabel },
          { href: X_PROFILE_URL, label: `Official X (@${X_SCREEN_NAME})`, external: true },
        ]
      : [
          { href: "/", label: "TOP" },
          { href: "/weekly", label: "週次トレンド" },
          { href: "/ranking", label: "投資家ランキング" },
          { href: "/investors", label: "投資家一覧" },
          { href: "/stocks", label: "銘柄一覧" },
          { href: "/about", label: "このサイトについて" },
          { href: "/privacy", label: t.privacyMenuLabel },
          { href: "/faq", label: "よくある質問" },
          { href: "/feed.xml", label: "RSSフィード" },
          { href: X_PROFILE_URL, label: `公式X（@${X_SCREEN_NAME}）`, external: true },
          { href: YOUTUBE_CHANNEL_URL, label: "公式YouTube（1分ショート解説）", external: true },
        ];

  const currentPath = pathname ?? (locale === "en" ? "/en" : "/");
  const otherHref = alternatePath(currentPath);
  const jaHref = locale === "ja" ? currentPath : otherHref;
  const enHref = locale === "en" ? currentPath : otherHref;

  return (
    // ヘッダー右端グループ(検索・訪問者数・メニュー)の一員としてフロー配置する。
    // かつてはPCで画面右端にabsolute固定していたが、AppBarのbackdrop-filterが
    // 包含ブロックを作るため実際にはカラム右端に落ち、訪問者数の上に重なるだけだった。
    <Box sx={{ flexShrink: 0 }}>
      <IconButton
        aria-label={t.menuLabel}
        aria-expanded={open}
        onClick={() => setOpen(true)}
        size="small"
        sx={{ color: "primary.main" }}
      >
        <MenuIcon fontSize="small" />
      </IconButton>

      {open && (
        <HeaderMenuDrawer
          open={open}
          onClose={close}
          locale={locale}
          siteLinks={siteLinks}
          jaHref={jaHref}
          enHref={enHref}
          year={year}
        />
      )}
    </Box>
  );
}
