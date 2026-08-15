import { NextRequest, NextResponse } from "next/server";
import { searchStocks } from "@/lib/microcms";
import { getAllFilers } from "@/lib/investors";

// ヘッダーの検索ボックス(StockSearch)から叩かれる。銘柄は企業名 or 証券コードの部分一致、
// 投資家はEDINET提出者名の部分一致で検索する。
// 投資家はgetAllFilers()のキャッシュ済み一覧をメモリ上で絞り込むため、Supabaseへの
// 追加往復は発生しない。
const MAX_INVESTOR_RESULTS = 10;

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = (searchParams.get("q") ?? "").trim();
  if (!q) {
    return NextResponse.json({ results: [], investors: [] });
  }

  const [results, filers] = await Promise.all([
    searchStocks(q),
    // 投資家検索はSupabase障害時も銘柄検索を止めないよう、失敗は空配列に落とす。
    getAllFilers().catch(() => []),
  ]);

  const investors = filers
    .filter((f) => f.filerName.includes(q))
    .slice(0, MAX_INVESTOR_RESULTS)
    .map(({ filerName, category }) => ({ filerName, category }));

  return NextResponse.json({ results, investors });
}
