import type { Metadata } from "next";
import Link from "next/link";
import { SITE_NAME, SITE_URL, X_HANDLE, X_PROFILE_URL } from "@/lib/site";

const title = "プライバシーポリシー";
const description =
  "本サイトにおけるアクセス解析ツール・広告配信（Google AdSense）でのCookie利用と、その無効化方法について。";

export const metadata: Metadata = {
  title,
  description,
  alternates: {
    canonical: `${SITE_URL}/privacy`,
  },
};

export default function PrivacyPage() {
  return (
    <article className="border-t border-rule bg-paper p-6 sm:p-10">
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">{title}</h1>

      <section className="mb-6">
        <p className="text-sm leading-relaxed text-ink-secondary">
          {SITE_NAME}（{SITE_URL}
          。以下「本サイト」）における、閲覧者の情報の取り扱いについて定めたものです。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">広告配信について</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          本サイトは、第三者配信の広告サービス「Google
          AdSense」を利用しています。Googleを含む第三者配信事業者は、Cookieを使用して、閲覧者の
          本サイトや他のサイトへの過去のアクセス情報に基づいた広告を配信します。
        </p>
        <p className="mt-3 text-sm leading-relaxed text-ink-secondary">
          パーソナライズ広告（閲覧履歴に基づく広告）は、
          <a
            href="https://myadcenter.google.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            Google 広告設定
          </a>
          からオフにできます。また、第三者配信事業者のCookie利用を無効にする方法については
          <a
            href="https://www.aboutads.info/choices/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            www.aboutads.info
          </a>
          をご覧ください。Googleが広告事業でどのように情報を利用するかについては
          <a
            href="https://policies.google.com/technologies/ads"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            Googleの広告に関するポリシー
          </a>
          に記載されています。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">アクセス解析について</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          本サイトは、閲覧状況の把握のために以下のツール・仕組みを利用しています。いずれも
          氏名・メールアドレス・電話番号といった個人を特定できる情報は取得しません。
        </p>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-ink-secondary">
          <li>
            <strong className="text-brand-navy">Google アナリティクス（GA4）</strong>:
            Cookieを利用してトラフィックデータを収集します。データの収集を無効にしたい場合は
            <a
              href="https://tools.google.com/dlpage/gaoptout?hl=ja"
              target="_blank"
              rel="noopener noreferrer"
              className="text-brand-blue hover:underline"
            >
              Google アナリティクス オプトアウト アドオン
            </a>
            をご利用ください。
          </li>
          <li>
            <strong className="text-brand-navy">Vercel Speed Insights</strong>:
            ページの表示速度を集計します。
          </li>
          <li>
            <strong className="text-brand-navy">運営者独自のアクセスログ</strong>:
            検索エンジン等のクローラーと実際のブラウザからのアクセスを区別するため、
            アクセス日時・ページ・User-Agentを記録しています。ブラウザからのアクセスには、
            訪問者数の重複集計を避ける目的でランダムな識別子（<code>kw_vid</code>
            Cookie）を発行しますが、この識別子から個人を特定することはできません。
          </li>
        </ul>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">Cookieの無効化</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          Cookieはブラウザの設定からいつでも無効化・削除できます。ただし無効化した場合、
          本サイトの一部機能が正しく動作しないことがあります。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">個人情報の取り扱い</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          本サイトは会員登録を必要とせず、閲覧者から氏名・メールアドレス等の個人情報を
          サイト上で直接取得することはありません。上記の解析・広告配信で取得される情報に
          ついても、運営者が個人を特定する目的で利用することはありません。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">制定日・改定日</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          制定日: 2026年8月15日
          <br />
          最終改定日: 2026年8月26日（お問い合わせ窓口の明記）
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">お問い合わせ</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          本サイトへのご連絡は、公式X（旧Twitter）
          <a
            href={X_PROFILE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            {X_HANDLE}
          </a>
          のダイレクトメッセージまたはリプライで受け付けています。掲載内容の誤りのご指摘、
          訂正・削除のご依頼、引用・転載のお問い合わせ、本ポリシーに関するお問い合わせは、
          すべてこの窓口へお寄せください。ご指摘をいただいた場合は、EDINET・TDnetの開示原本と
          照合のうえ確認し、誤りが確認された記事は修正または削除します。
        </p>
        <p className="mt-3 text-sm leading-relaxed text-ink-secondary">
          なお本サイトは金融商品取引法上の投資助言・代理業者ではないため、個別銘柄の売買や
          投資判断に関するご相談にはお答えできません。
        </p>
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-xl font-bold text-brand-navy">免責事項</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          本サイトの内容に関する免責事項、データソース、運営者情報については
          <Link href="/about" className="text-brand-blue hover:underline">
            このサイトについて
          </Link>
          をご覧ください。本サイトからリンクした外部サイト・広告主のサイトで提供される情報・
          サービスについては、運営者は一切の責任を負いません。
        </p>
      </section>

      <section>
        <h2 className="mb-2 text-xl font-bold text-brand-navy">改定について</h2>
        <p className="text-sm leading-relaxed text-ink-secondary">
          本ポリシーの内容は、法令の変更や利用ツールの追加・変更に応じて予告なく改定することが
          あります。ご質問は
          <a
            href="https://x.com/kujira_watch"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-blue hover:underline"
          >
            公式X（@kujira_watch）
          </a>
          までお寄せください。
        </p>
      </section>
    </article>
  );
}
