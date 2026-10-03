import { draftMode } from "next/headers";
import { redirect } from "next/navigation";

// プレビュー（Draft Mode）を解除して通常表示に戻す。編集後に同じブラウザで
// サイトを見ると下書きのまま表示されるため、その時に開く。
export async function GET() {
  (await draftMode()).disable();
  redirect("/");
}
