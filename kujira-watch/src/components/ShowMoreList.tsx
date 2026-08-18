"use client";

import { useState, type ReactNode } from "react";

// 続きを「もっと見る」で開く一覧。押すとボタンが消え、残りが同じグリッドの一番下に
// そのまま続く。<details>だと開いた続きが別リストになり、ボタンが一覧の途中に
// 挟まったまま残って落ち着かないため置き換えた。
// 閉じている間も残りはHTMLに含めてCSS(.show-more-collapsed)で隠すだけなので、
// <details>のときと同じくクローラーからは全件見える。
// 隠す対象のliには呼び出し側でshow-more-extraを付けること。
export default function ShowMoreList({
  className,
  restCount,
  unit,
  children,
}: {
  className: string;
  restCount: number;
  unit: string;
  children: ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  const collapsed = !expanded && restCount > 0;

  return (
    <>
      <ul className={collapsed ? `${className} show-more-collapsed` : className}>{children}</ul>
      {collapsed && (
        <div className="mt-3 text-center">
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="text-sm text-brand-blue hover:underline"
          >
            もっと見る（残り{restCount}{unit}）
          </button>
        </div>
      )}
    </>
  );
}
