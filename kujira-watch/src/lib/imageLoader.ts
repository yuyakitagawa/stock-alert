"use client";

/**
 * next/image のカスタムローダー。Vercelの画像最適化を経由せず、microCMSの画像API
 * （imgixベース）にリサイズ・WebP変換をやらせる。
 *
 * 2026-08-27にVercelの画像最適化の無料枠を使い切り、本番の全アイキャッチが
 * `OPTIMIZED_IMAGE_REQUEST_PAYMENT_REQUIRED`（HTTP 402）で表示できなくなった
 * （next/imageは読み込みに失敗するとalt文字列を出すため、カード上でタイトルが
 * 二重に見える状態になっていた）。記事は毎日増え、画像を差し替えるたびにURLが変わって
 * 最適化がやり直しになるので、枠のある仕組みに載せ続けること自体が持たない。
 * microCMS側の変換は転送量（Hobbyで20GB/月）にしか効かないので、そちらへ寄せる。
 */
export default function microcmsImageLoader({
  src,
  width,
  quality,
}: {
  src: string;
  width: number;
  quality?: number;
}): string {
  // microCMS以外（/public配下の画像など）は変換パラメータを付けずそのまま返す
  if (!src.startsWith("https://images.microcms-assets.io/")) return src;
  const sep = src.includes("?") ? "&" : "?";
  return `${src}${sep}w=${width}&q=${quality ?? 75}&fm=webp`;
}
