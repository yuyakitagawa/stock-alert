// ウォッチリストのlocalStorage永続化ヘルパー。アカウント機能を持たないサイトのため、
// 「気になる投資家・銘柄を保存して再訪する」導線は端末ローカルにのみ保存する
// （サーバーには一切送らない。クライアントコンポーネントからのみ呼ぶこと）。
export type WatchItemType = "investor" | "stock";

export type WatchItem = {
  type: WatchItemType;
  // investor: 投資家名（デコード済み）/ stock: 証券コード
  id: string;
  // 一覧表示用のラベル（investorは投資家名と同じ、stockは「社名（コード）」）
  label: string;
  addedAt: number;
};

const STORAGE_KEY = "kujira-watchlist";

// 同一タブ内の別コンポーネント（保存ボタン⇔一覧）へ変更を伝えるためのイベント。
// storageイベントは他タブにしか飛ばないため自前で発火する。
export const WATCHLIST_CHANGE_EVENT = "kujira-watchlist-change";

export function getWatchlist(): WatchItem[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is WatchItem =>
        typeof item === "object" &&
        item !== null &&
        ((item as WatchItem).type === "investor" || (item as WatchItem).type === "stock") &&
        typeof (item as WatchItem).id === "string" &&
        typeof (item as WatchItem).label === "string"
    );
  } catch {
    // 壊れたJSON等は「未保存」として扱う（閲覧を妨げない）
    return [];
  }
}

export function isWatched(type: WatchItemType, id: string): boolean {
  return getWatchlist().some((item) => item.type === type && item.id === id);
}

function save(items: WatchItem[]) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    window.dispatchEvent(new Event(WATCHLIST_CHANGE_EVENT));
  } catch {
    // プライベートブラウジング等で保存できない場合は黙って諦める
  }
}

// 保存⇔解除をトグルし、トグル後の保存状態を返す。
export function toggleWatch(item: Omit<WatchItem, "addedAt">): boolean {
  const items = getWatchlist();
  const exists = items.some((i) => i.type === item.type && i.id === item.id);
  if (exists) {
    save(items.filter((i) => !(i.type === item.type && i.id === item.id)));
    return false;
  }
  save([...items, { ...item, addedAt: Date.now() }]);
  return true;
}

export function removeWatch(type: WatchItemType, id: string) {
  save(getWatchlist().filter((i) => !(i.type === type && i.id === id)));
}
