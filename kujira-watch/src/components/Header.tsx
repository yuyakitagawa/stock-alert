"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Box from "@mui/material/Box";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import HeaderMenu from "./HeaderMenu";
import StockSearch from "./StockSearch";
import VisitCounter from "./VisitCounter";
import { SITE_NAME, SITE_NAME_EN } from "@/lib/site";
import { DEAL_TYPES } from "@/types/article";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import type { Locale } from "@/lib/i18n";

// タブが「今どのページを開いているか」を示せるよう、現在のパスと一致する
// (または配下にある)リンクだけをアクティブ表示する。
function isActiveTab(pathname: string, href: string): boolean {
  if (href === "/" || href === "/en") return pathname === href;
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function Header({ locale = "ja" }: { locale?: Locale }) {
  const homeHref = locale === "en" ? "/en" : "/";
  const pathname = usePathname() ?? homeHref;

  const navLinks =
    locale === "en"
      ? [
          { href: "/en", label: "Top" },
          ...DEAL_TYPES.map((dealType) => ({
            href: `/en/category/${DEAL_TYPE_EN[dealType].slug}`,
            label: DEAL_TYPE_EN[dealType].label,
          })),
        ]
      : [
          { href: "/", label: "TOP" },
          { href: "/disclosures", label: "開示速報" },
          { href: "/weekly", label: "今週のまとめ" },
          { href: "/trending", label: "急増ランキング" },
          { href: "/ranking", label: "投資家勝率ランキング" },
          { href: "/activists", label: "アクティビスト保有" },
          { href: "/investors", label: "投資家一覧" },
          { href: "/stocks", label: "銘柄一覧" },
          // 月別アーカイブは回遊の起点というより過去分の入口なので一番右に置く。
          { href: "/monthly", label: "月別アーカイブ" },
        ];

  // MUI TabsのvalueはchildのTabのvalueと厳密一致する必要があるため、
  // サブページ(配下URL)では一致する先頭リンクのhrefを採用する。どれにも
  // 一致しない場合はfalseにして「選択中タブなし」の状態にする。
  const activeHref = navLinks.find((link) => isActiveTab(pathname, link.href))?.href ?? false;

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        bgcolor: "rgba(255, 253, 248, 0.95)",
        backdropFilter: "blur(6px)",
        borderBottom: 1,
        borderColor: "primary.main",
        color: "primary.main",
      }}
    >
      <Toolbar
        disableGutters
        sx={{ mx: "auto", width: "100%", maxWidth: "48rem", px: 2, pt: 2, pb: 1, alignItems: "flex-start" }}
      >
        <Box sx={{ width: "100%" }}>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}>
            <Box
              component={Link}
              href={homeHref}
              sx={{ display: "flex", alignItems: "center", gap: 1.25, textDecoration: "none" }}
            >
              <Box component="span" aria-hidden sx={{ fontSize: "1.25rem", lineHeight: 1 }}>
                🐋
              </Box>
              <Box component="span" sx={{ lineHeight: 1.2 }}>
                <Box
                  component="span"
                  sx={{
                    display: "block",
                    fontSize: { xs: "1.25rem", sm: "1.5rem" },
                    fontWeight: 700,
                    letterSpacing: "-0.01em",
                    color: "primary.main",
                    // サイト名はブランドの顔なので絶対に折り返させない（375pxで3行に折れていた）。
                    whiteSpace: "nowrap",
                  }}
                >
                  {locale === "en" ? SITE_NAME_EN : SITE_NAME}
                </Box>
                <Box
                  component="span"
                  className="kicker"
                  sx={{ display: { xs: "none", sm: "block" }, mt: 0.25, color: "brand.blue" }}
                >
                  {locale === "en"
                    ? "Tracking Japan's Market \"Whales\" from Large-Holding Filings"
                    : "EDINET大量保有報告書から読む「クジラ」の動き"}
                </Box>
              </Box>
            </Box>
            <Box sx={{ display: "flex", flexShrink: 0, alignItems: "center", gap: 1 }}>
              <StockSearch locale={locale} />
              {/* モバイルではロゴの横幅を優先し、訪問者数はsm以上でのみ表示する。 */}
              <Box sx={{ display: { xs: "none", sm: "flex" }, alignItems: "center" }}>
                <VisitCounter locale={locale} />
              </Box>
              <HeaderMenu locale={locale} />
            </Box>
          </Box>
          <Tabs
            value={activeHref}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            aria-label={locale === "en" ? "Categories" : "主要ページ"}
            sx={{
              mt: 1,
              borderTop: 1,
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
              "& .Mui-selected": { color: "brand.blue" },
              "& .MuiTabs-indicator": { bgcolor: "brand.blue" },
            }}
          >
            {navLinks.map((link) => (
              <Tab key={link.href} value={link.href} label={link.label} component={Link} href={link.href} />
            ))}
          </Tabs>
        </Box>
      </Toolbar>
    </AppBar>
  );
}
