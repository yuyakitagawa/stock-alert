"use client";
import { useCallback, useEffect, useState } from "react";

/**
 * ブックマーク（マイ・ウォッチリスト）を localStorage に保存する。
 * ログイン不要・端末/ブラウザごと。複数コンポーネント間とタブ間で同期する。
 */
const KEY = "stocksignal:bookmarks";
const EVENT = "stocksignal:bookmarks-changed";

function read(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function write(codes: string[]) {
  localStorage.setItem(KEY, JSON.stringify(codes));
  // 同一タブ内の他コンポーネントへ通知（storageイベントは他タブにしか飛ばないため）
  window.dispatchEvent(new CustomEvent(EVENT));
}

export function useBookmarks() {
  // SSR とのハイドレーション不一致を避けるため、初期値は空 → マウント後に読み込む
  const [codes, setCodes] = useState<string[]>([]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setCodes(read());
    setMounted(true);
    const sync = () => setCodes(read());
    window.addEventListener(EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const toggle = useCallback((code: string) => {
    const cur = read();
    write(cur.includes(code) ? cur.filter((c) => c !== code) : [code, ...cur]);
  }, []);

  const remove = useCallback((code: string) => {
    write(read().filter((c) => c !== code));
  }, []);

  return {
    codes,
    mounted,
    count: codes.length,
    isBookmarked: (code: string) => codes.includes(code),
    toggle,
    remove,
  };
}
