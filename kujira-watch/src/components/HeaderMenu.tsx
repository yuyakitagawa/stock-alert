"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import type { MenuGroup } from "./HeaderMenuDrawer";
import { mainNavLinks } from "@/lib/nav";
import { UI } from "@/lib/i18n";
import { X_PROFILE_URL, X_SCREEN_NAME, YOUTUBE_CHANNEL_URL } from "@/lib/site";

// ドロワー本体(Drawer/List/Divider/…)は開くまで要らない。MUI DrawerはModal/Portal/
// Backdrop/Slide一式を引き連れており、閉じているのが既定なのに全ページの初期JSへ
// 積まれていた。開いたときに読み込む。
const HeaderMenuDrawer = dynamic(() => import("./HeaderMenuDrawer"), { ssr: false });

// オートスクロールで記事一覧が際限なく伸びるページでは最下部のフッターまで
// スクロールしてたどり着けないため、主要リンクはヘッダー右上のハンバーガーメニュー
// からも常にアクセスできるようにする（フッター(Footer.tsx)とは併存させる）。
export default function HeaderMenu() {
  const [open, setOpen] = useState(false);
  const year = new Date().getFullYear();
  const t = UI;

  const close = () => setOpen(false);

  // 先頭グループは上部タブと並び・ラベルを完全一致させる（lib/nav.tsで一元管理。
  // 別々に定義するとページ改名時にメニューだけ取り残されるため）。タブに無い
  // ページはその下に見出し付きグループで置く（フッター(Footer.tsx)と同じ分類名）。
  const menuGroups: MenuGroup[] = [
    { heading: "主要ページ", links: mainNavLinks() },
    {
      heading: "サイト情報",
      links: [
        { href: "/about", label: "このサイトについて" },
        { href: "/faq", label: "よくある質問" },
        { href: "/privacy", label: t.privacyMenuLabel },
        { href: "/terms", label: "利用規約" },
      ],
    },
    {
      heading: "フォロー",
      links: [
        { href: X_PROFILE_URL, label: `公式X（@${X_SCREEN_NAME}）`, external: true },
        { href: YOUTUBE_CHANNEL_URL, label: "公式YouTube（1分ショート解説）", external: true },
        { href: "/feed.xml", label: "RSSフィード" },
      ],
    },
  ];

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
          menuGroups={menuGroups}
          year={year}
        />
      )}
    </Box>
  );
}
