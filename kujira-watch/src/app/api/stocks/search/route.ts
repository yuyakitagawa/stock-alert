import { NextRequest, NextResponse } from "next/server";
import { searchStocks } from "@/lib/microcms";
import { searchStockMaster } from "@/lib/companyInfo";
import { getAllFilers } from "@/lib/investors";

// ヘッダーの検索ボックス(StockSearch)から叩かれる。銘柄は企業名 or 証券コードの部分一致、
// 投資家はEDINET提出者名の部分一致で検索する。
// 銘柄はmicroCMS記事(searchStocks)とjpx_stock_list全上場銘柄(searchStockMaster)の
// 両方から引く。記事側だけだと記事化されていない銘柄が検索から消えるため
// （実例: 6929 日本セラミック、2026-08-18）。記事のある銘柄の方が読者に見せる中身が
// 多いので、記事側のヒットを先に並べる。
// 投資家はgetAllFilers()のキャッシュ済み一覧をメモリ上で絞り込むため、Supabaseへの
// 追加往復は発生しない。
const MAX_INVESTOR_RESULTS = 10;
const MAX_STOCK_RESULTS = 20;

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = (searchParams.get("q") ?? "").trim();
  if (!q) {
    return NextResponse.json({ results: [], investors: [] });
  }

  const [articleHits, masterHits, filers] = await Promise.all([
    searchStocks(q),
    // マスター検索が落ちても記事側の結果は返す（逆も同様）。
    searchStockMaster(q).catch(() => []),
    // 投資家検索はSupabase障害時も銘柄検索を止めないよう、失敗は空配列に落とす。
    getAllFilers().catch(() => []),
  ]);

  const byCode = new Map<string, { stockCode: string; stockName: string }>();
  for (const hit of [...articleHits, ...masterHits]) {
    if (!byCode.has(hit.stockCode)) byCode.set(hit.stockCode, hit);
  }
  const results = Array.from(byCode.values()).slice(0, MAX_STOCK_RESULTS);

  const investors = filers
    .filter((f) => f.filerName.includes(q))
    .slice(0, MAX_INVESTOR_RESULTS)
    .map(({ filerName, category }) => ({ filerName, category }));

  return NextResponse.json({ results, investors });
}
