"use client";

import { useMemo } from "react";
import { AppRouterCacheProvider } from "@mui/material-nextjs/v13-appRouter";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { PALETTES, type PaletteName } from "@/lib/siteTheme";
import { createSiteTheme } from "@/theme";

export default function ThemeRegistry({
  palette = "classic",
  children,
}: {
  palette?: PaletteName;
  children: React.ReactNode;
}) {
  const theme = useMemo(() => createSiteTheme(PALETTES[palette]), [palette]);
  return (
    <AppRouterCacheProvider>
      <ThemeProvider theme={theme}>
        <CssBaseline enableColorScheme={false} />
        {children}
      </ThemeProvider>
    </AppRouterCacheProvider>
  );
}
