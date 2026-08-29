import type { Metadata } from "next";
import Link from "next/link";
import { SITE_NAME, SITE_URL, X_HANDLE, X_PROFILE_URL } from "@/lib/site";

const title = "お問い合わせ";
const description = `${SITE_NAME}への連絡窓口です。掲載内容の誤りのご指摘、訂正・削除のご依頼、引用・転載のお問い合わせを公式Xで受け付けています。`;

export const metadata: Metadata = {
  title,
  description,
  alternates: {
    canonical: `${SITE_URL}/contact`,
  },
};

// 運営者は実名・メールアドレスを公開しない方針のため、窓口は公式XのDM/リプライ1本に絞る
// （lib/site.ts の ORGANIZATION_CONTACT_POINT と /about・/privacy の記載もこの窓口を指す）。
// 窓口ページが無く /contact が404だったため、連絡手段がプライバシーポリシー本文の中にしか
// 書かれていない状態だった（2026-08-25の監査で検出）。用件ごとの扱いもここに集約する。
export default function ContactPage() {
  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-foreground/70">
          {SITE_NAME}へのご連絡は、公式X（旧Twitter）
          <a
            href={X_PROFILE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            {X_HANDLE}
          </a>
          のダイレクトメッセージまたはリプライで受け付けています。運営者は実名・メールアドレスを公開していないため、窓口はこの1か所です。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">掲載内容の誤りのご指摘</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          記事の数値・社名・保有比率などに誤りを見つけられた場合は、対象記事のURLを添えてお知らせください。EDINETの大量保有報告書、TDnetの適時開示といった開示原本と照合したうえで確認し、誤りが確認された記事は修正または削除します。是正した内容は
          <Link href="/about" className="text-brand-blue hover:underline">
            このサイトについて
          </Link>
          に記載の方針に沿って扱います。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">訂正・削除のご依頼</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          記事で言及されている企業・投資家ご本人からの訂正・削除のご依頼も、同じ窓口で承ります。本サイトが扱うのは金融商品取引法に基づいて公開された開示書類の内容ですが、記載の誤りや、開示の趣旨と異なる書き方があればお知らせください。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">引用・転載、取材のお問い合わせ</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          記事・図表の引用や転載、取材のご依頼についてもこの窓口へお寄せください。引用の範囲や条件は
          <Link href="/terms" className="text-brand-blue hover:underline">
            利用規約
          </Link>
          に定めています。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">お答えできないこと</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          本サイトは金融商品取引法上の投資助言・代理業者ではありません。個別銘柄の売買や投資判断に関するご相談にはお答えできません。特定の銘柄を取り上げるよう求めるご依頼、開示に基づかない内容の掲載のご依頼にも応じられません。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">個人情報の取り扱い</h2>
        <p className="text-sm leading-relaxed text-foreground/70">
          本サイトはお問い合わせフォームを設けておらず、閲覧者から氏名・メールアドレス等をサイト上で直接取得することはありません。Xを通じていただいたご連絡の内容は、ご依頼への対応の目的にのみ利用します。詳細は
          <Link href="/privacy" className="text-brand-blue hover:underline">
            プライバシーポリシー
          </Link>
          をご確認ください。
        </p>
      </section>

      <p className="text-sm leading-relaxed text-foreground/70">
        <a
          href={X_PROFILE_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="text-brand-blue hover:underline"
        >
          公式X（{X_HANDLE}）を開く
        </a>
      </p>
    </article>
  );
}
