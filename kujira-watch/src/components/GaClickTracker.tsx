"use client";

import { useEffect } from "react";
import { sendGAEvent } from "@next/third-parties/google";

function labelFor(el: Element): string {
  const aria = el.getAttribute("aria-label");
  if (aria) return aria.trim();
  const text = el.textContent?.trim();
  if (text) return text.slice(0, 100);
  if (el instanceof HTMLAnchorElement) return el.href;
  return el.tagName.toLowerCase();
}

// ページ内の全ボタン・リンククリックをGA4に送るためのグローバルリスナー。
// 各ボタンに個別実装せず、ここ1箇所でサイト全体をカバーする。
export default function GaClickTracker() {
  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      const target = (event.target as Element | null)?.closest(
        "a, button, [role='button']",
      );
      if (!target) return;
      sendGAEvent("event", "click", {
        label: labelFor(target),
        tag: target.tagName.toLowerCase(),
        path: window.location.pathname,
      });
    };
    document.addEventListener("click", handleClick, true);
    return () => document.removeEventListener("click", handleClick, true);
  }, []);

  return null;
}
