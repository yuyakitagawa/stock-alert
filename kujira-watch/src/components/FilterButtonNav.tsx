import Stack from "@mui/material/Stack";
import ActionButton from "./ActionButton";

// 一覧ページ（ランキング・銘柄・投資家）の絞り込みナビ。以前はテキストリンクの
// 太字/薄字で選択状態を表していたが、押せる要素だと分かりづらかったため
// ボタン化し、選択中のみ塗りつぶし(contained)にする。
export default function FilterButtonNav({
  ariaLabel,
  items,
}: {
  ariaLabel: string;
  items: { href: string; label: string; selected: boolean }[];
}) {
  return (
    <Stack
      component="nav"
      aria-label={ariaLabel}
      direction="row"
      className="no-scrollbar"
      sx={{
        gap: 1,
        mb: 3,
        // スマホでは折り返すと絞り込みだけで画面が埋まってしまうため横スクロールにする。
        flexWrap: { xs: "nowrap", sm: "wrap" },
        overflowX: { xs: "auto", sm: "visible" },
        "& .MuiButton-root": { flexShrink: 0 },
      }}
    >
      {items.map((item) => (
        <ActionButton key={item.href} href={item.href} selected={item.selected}>
          {item.label}
        </ActionButton>
      ))}
    </Stack>
  );
}
