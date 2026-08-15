"use client";

import { useEffect } from "react";

// MUIコンポーネント(.MuiButtonBase-root)は独自のTouchRippleを持つため対象から除外する
// (二重にリップルが発火してしまうため)。
const RIPPLE_SELECTOR =
  "a:not(.MuiButtonBase-root), button:not(.MuiButtonBase-root), [role='button']:not(.MuiButtonBase-root), summary:not(.MuiButtonBase-root)";
const RIPPLE_DURATION_MS = 600;

// タップ位置から波紋が広がる視覚フィードバック。各要素に個別実装せず、
// ここ1箇所のグローバルリスナーでサイト全体をカバーする。
export default function RippleEffect() {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const originalStyles = new WeakMap<HTMLElement, { position: string; overflow: string }>();

    const handlePointerDown = (event: PointerEvent) => {
      if (event.button !== 0) return;
      const el = (event.target as Element | null)?.closest<HTMLElement>(RIPPLE_SELECTOR);
      if (!el || el.hasAttribute("disabled") || el.getAttribute("aria-disabled") === "true") return;

      const rect = el.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const radius = Math.hypot(Math.max(x, rect.width - x), Math.max(y, rect.height - y));
      const diameter = radius * 2;

      if (!originalStyles.has(el)) {
        originalStyles.set(el, { position: el.style.position, overflow: el.style.overflow });
      }
      if (getComputedStyle(el).position === "static") el.style.position = "relative";
      el.style.overflow = "hidden";

      const span = document.createElement("span");
      span.className = "ripple-span";
      span.style.width = `${diameter}px`;
      span.style.height = `${diameter}px`;
      span.style.left = `${x - radius}px`;
      span.style.top = `${y - radius}px`;
      el.appendChild(span);

      window.setTimeout(() => {
        span.remove();
        if (!el.querySelector(".ripple-span")) {
          const saved = originalStyles.get(el);
          if (saved) {
            el.style.position = saved.position;
            el.style.overflow = saved.overflow;
          }
        }
      }, RIPPLE_DURATION_MS);
    };

    // passive指定。このハンドラはpreventDefaultを呼ばないので、ブラウザに
    // 「スクロール開始前にこの処理の完了を待つ必要はない」と伝えられる。
    // 指定が無いとdocument上のpointerdownがスクロール開始をブロックしうる。
    document.addEventListener("pointerdown", handlePointerDown, { passive: true });
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, []);

  return null;
}
