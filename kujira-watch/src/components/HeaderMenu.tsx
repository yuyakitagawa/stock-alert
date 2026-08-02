"use client";

import { useState } from "react";
import Link from "next/link";
import { SITE_NAME } from "@/lib/site";
import VisitCounter from "./VisitCounter";

// オートスクロールで記事一覧が際限なく伸びるため、ページ最下部のフッターまで
// スクロールしてたどり着くのが実質困難になった。運営者情報・免責事項等は
// ヘッダー右上のハンバーガーメニューに集約し、常にアクセスできるようにする。
export default function HeaderMenu() {
  const [open, setOpen] = useState(false);
  const year = new Date().getFullYear();

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        aria-label="メニュー"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="flex h-9 w-9 items-center justify-center rounded-full text-brand-navy hover:bg-section-tint"
      >
        <span aria-hidden className="text-xl leading-none">
          {open ? "✕" : "☰"}
        </span>
      </button>
      {open && (
        <>
          <button
            type="button"
            aria-label="メニューを閉じる"
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-10 cursor-default"
          />
          <div className="absolute right-0 top-full z-20 mt-2 w-72 rounded-lg bg-white p-4 text-xs text-gray-600 shadow-lg ring-1 ring-gray-200">
            <p className="font-semibold text-brand-navy">🐋 {SITE_NAME}へようこそ</p>
            <p className="mt-1 leading-relaxed">
              {SITE_NAME}は、EDINETの大量保有報告書（5%ルール）などの公開情報をもとに、機関投資家・
              アクティビストファンド・インサイダー・自社株買いといった「クジラ」（相場を動かすほどの
              資金力を持つ大口投資家の俗称）が、どの銘柄をいつ・どれくらいの規模で動かしたかを
              日次でまとめて解説するブログです。個人投資家では追いきれない大口投資家の動きを、
              取引日ごとに一覧できます。
            </p>
            <p className="mt-3 leading-relaxed">
              本サイトはEDINET大量保有報告書等の公開情報をもとにした解説であり、投資助言ではありません。
            </p>
            <div className="mt-3 flex flex-col gap-2 text-sm text-brand-blue">
              <Link href="/about" onClick={() => setOpen(false)} className="hover:underline">
                運営者情報・免責事項
              </Link>
              <Link href="/feed.xml" onClick={() => setOpen(false)} className="hover:underline">
                RSSフィード
              </Link>
            </div>
            <p className="mt-3 text-gray-400">
              © {year} {SITE_NAME}
            </p>
            <VisitCounter />
          </div>
        </>
      )}
    </div>
  );
}
