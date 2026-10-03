"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

type Bridge = { on: (events: string[], cb: () => void) => void };
declare global {
  interface Window {
    StoryblokBridge?: new () => Bridge;
  }
}

const BRIDGE_SRC = "https://app.storyblok.com/f/storyblok-v2-latest.js";

// ビジュアルエディタ内（Draft Mode時）だけ読み込む。保存・公開のたびにサーバー描画を
// 取り直して画面へ反映する。一般閲覧者には出力しない（layout.tsx で分岐）。
export default function StoryblokBridge() {
  const router = useRouter();
  useEffect(() => {
    const start = () => {
      if (!window.StoryblokBridge) return;
      new window.StoryblokBridge().on(["change", "published"], () => router.refresh());
    };
    if (window.StoryblokBridge) return start();
    const script = document.createElement("script");
    script.src = BRIDGE_SRC;
    script.async = true;
    script.onload = start;
    document.body.appendChild(script);
  }, [router]);
  return null;
}
