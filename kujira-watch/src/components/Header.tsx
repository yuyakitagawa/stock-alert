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
import { SITE_NAME } from "@/lib/site";
import { mainNavLinks } from "@/lib/nav";

// タブが「今どのページを開いているか」を示せるよう、現在のパスと一致する
// (または配下にある)リンクだけをアクティブ表示する。
function isActiveTab(pathname: string, href: string): boolean {
  if (href === "/") return pathname === href;
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function Header() {
  const homeHref = "/";
  const pathname = usePathname() ?? homeHref;

  const navLinks = mainNavLinks();

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
              {/* ファビコン(icon.tsx)と同じ「紺丸＋クジラ」のロゴマークに揃える。 */}
              <Box
                component="span"
                aria-hidden
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 32,
                  height: 32,
                  flexShrink: 0,
                  borderRadius: "50%",
                  bgcolor: "primary.main",
                  fontSize: "var(--text-xl)",
                  lineHeight: 1,
                }}
              >
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
                  {SITE_NAME}
                </Box>
                <Box
                  component="span"
                  className="kicker"
                  sx={{ display: { xs: "none", sm: "block" }, mt: 0.25, color: "brand.blue" }}
                >
                  EDINET大量保有報告書から読む大口投資家の動き
                </Box>
              </Box>
            </Box>
            <Box sx={{ display: "flex", flexShrink: 0, alignItems: "center", gap: 1 }}>
              <StockSearch />
              {/* モバイルではロゴの横幅を優先し、訪問者数はsm以上でのみ表示する。 */}
              <Box sx={{ display: { xs: "none", sm: "flex" }, alignItems: "center" }}>
                <VisitCounter />
              </Box>
              <HeaderMenu />
            </Box>
          </Box>
          <Tabs
            value={activeHref}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            aria-label="主要ページ"
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
                fontSize: "var(--text-2xs)",
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
