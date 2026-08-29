"use client";

import Link from "next/link";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import CloseIcon from "@mui/icons-material/Close";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { SITE_NAME } from "@/lib/site";
import VisitCounter from "./VisitCounter";

// external: 外部サイト（公式Xなど）へのリンク。next/linkではなく素のaで別タブに開く。
export type MenuLink = { href: string; label: string; external?: boolean };
// 見出し付きのリンクグループ。先頭グループは上部タブと同一内容（HeaderMenu側で構築）。
export type MenuGroup = { heading: string; links: MenuLink[] };

// ハンバーガーメニューの中身。閉じているのが既定なので、HeaderMenu側から
// next/dynamicで遅延読み込みする。MUI Drawerは Modal/Portal/Backdrop/Slide 一式を
// 引き連れており、それが全ページの初期JSに積まれていた。
// 見た目・挙動は据え置きで、読み込みのタイミングだけ後ろにずらす。
export default function HeaderMenuDrawer({
  open,
  onClose,
  menuGroups,
  year,
}: {
  open: boolean;
  onClose: () => void;
  menuGroups: MenuGroup[];
  year: number;
}) {
  const close = onClose;

  return (
      <Drawer anchor="right" open={open} onClose={close}>
        <Box sx={{ width: "86vw", maxWidth: 320, height: "100%", overflowY: "auto" }} role="presentation">
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: 1, borderColor: "divider", px: 2, py: 1.5 }}>
            <Typography variant="subtitle2" sx={{ display: "flex", alignItems: "center", gap: 1, fontWeight: 700, color: "primary.main" }}>
              <Box component="span" aria-hidden>
                🐋
              </Box>
              {SITE_NAME}
            </Typography>
            <IconButton
              aria-label="メニューを閉じる"
              onClick={close}
              size="small"
              sx={{ color: "primary.main" }}
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>


          {menuGroups.map((group) => (
            <Box key={group.heading}>
              <Typography variant="overline" sx={{ display: "block", bgcolor: "action.hover", px: 2, py: 0.75, color: "text.secondary" }}>
                {group.heading}
              </Typography>
              <List component="nav" aria-label={group.heading} disablePadding>
                {group.links.map((link) =>
                  link.external ? (
                    <ListItemButton
                      key={link.href}
                      component="a"
                      href={link.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={close}
                      divider
                    >
                      <ListItemText primary={link.label} slotProps={{ primary: { sx: { color: "primary.main" } } }} />
                      <ChevronRightIcon fontSize="small" sx={{ color: "action.disabled" }} />
                    </ListItemButton>
                  ) : (
                    <ListItemButton key={link.href} component={Link} href={link.href} onClick={close} divider>
                      <ListItemText primary={link.label} slotProps={{ primary: { sx: { color: "primary.main" } } }} />
                      <ChevronRightIcon fontSize="small" sx={{ color: "action.disabled" }} />
                    </ListItemButton>
                  ),
                )}
              </List>
            </Box>
          ))}

          <Typography variant="caption" sx={{ display: "block", mt: 1.5, px: 2, lineHeight: 1.6, color: "text.secondary" }}>
            本サイトはEDINET大量保有報告書等の公開情報をもとにした解説であり、投資助言ではありません。
          </Typography>
          <Divider sx={{ my: 1 }} />
          <Typography variant="caption" sx={{ display: "block", px: 2, color: "text.disabled" }}>
            © {year} {SITE_NAME}
          </Typography>
          {/* ヘッダーの訪問者数はスマホでは非表示のため、メニュー最下部にも置く（加算はしない）。 */}
          <Box sx={{ px: 2, pb: 2, pt: 0.5 }}>
            <VisitCounter increment={false} />
          </Box>
        </Box>
      </Drawer>
  );
}
