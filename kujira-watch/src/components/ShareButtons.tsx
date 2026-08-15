import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import ActionButton from "./ActionButton";
import { SITE_NAME } from "@/lib/site";

// 記事詳細の共有ボタン。JSを持ち込まないよう、公式のWeb Intent URLへの
// 素のリンク（MUI Buttonのcomponent="a"）だけで実装する（SDKを読み込むと
// ページ表示が目に見えて遅くなるため。/aboutのX埋め込みタイムラインを撤去したのと同じ理由）。
// 流入をGA4で識別できるようUTMを付ける（自動投稿のutm_source=xとは別値にして、
// 読者による手動シェア経由の流入と区別する）。
export default function ShareButtons({ url, title }: { url: string; title: string }) {
  const shareUrl = `${url}?utm_source=share&utm_medium=social`;
  const encodedUrl = encodeURIComponent(shareUrl);
  const xHref = `https://x.com/intent/post?text=${encodeURIComponent(`${title}｜${SITE_NAME}`)}&url=${encodedUrl}`;
  const lineHref = `https://social-plugins.line.me/lineit/share?url=${encodedUrl}`;

  return (
    <Stack
      direction="row"
      sx={{
        mt: 4,
        pt: 2.5,
        flexWrap: "wrap",
        alignItems: "center",
        gap: 1,
        borderTop: "1px solid var(--rule)",
      }}
    >
      <Typography variant="overline" sx={{ mr: 0.5, color: "text.disabled" }}>
        この記事を共有
      </Typography>
      <ActionButton href={xHref} external>
        Xでポスト
      </ActionButton>
      <ActionButton href={lineHref} external>
        LINEで送る
      </ActionButton>
    </Stack>
  );
}
