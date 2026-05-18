const UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

async function getAuth(): Promise<{ cookie: string; crumb: string } | null> {
  try {
    const r1 = await fetch("https://fc.yahoo.com", {
      headers: { "User-Agent": UA },
      redirect: "follow",
      cache: "no-store",
    });

    const raw = r1.headers.get("set-cookie") ?? "";
    const cookies = raw
      .split(/,(?=[^ ][^,]+=)/)
      .map(c => c.trim().split(";")[0])
      .filter(c => c.includes("="));
    const cookieStr = cookies.join("; ");

    const r2 = await fetch("https://query1.finance.yahoo.com/v1/test/getcrumb", {
      headers: { "User-Agent": UA, "Cookie": cookieStr },
      cache: "no-store",
    });
    const crumb = (await r2.text()).trim();
    if (!crumb || crumb.startsWith("{") || crumb.length > 60) return null;

    return { cookie: cookieStr, crumb };
  } catch {
    return null;
  }
}

export async function yfQuoteSummary(code: string, modules: string): Promise<Record<string, unknown> | null> {
  const auth = await getAuth();
  if (!auth) return null;

  try {
    const url = `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${code}.T?modules=${modules}&crumb=${encodeURIComponent(auth.crumb)}`;
    const res = await fetch(url, {
      headers: { "User-Agent": UA, "Cookie": auth.cookie },
      cache: "no-store",
    });
    if (!res.ok) return null;
    const data = await res.json();
    return (data?.quoteSummary?.result?.[0] as Record<string, unknown>) ?? null;
  } catch {
    return null;
  }
}
