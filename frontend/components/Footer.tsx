export default function Footer() {
  return (
    <footer className="mt-12">
      <div className="border-t border-gray-800 py-6 px-4">
        <div className="max-w-7xl mx-auto text-center space-y-2">
          <p className="text-xs text-gray-600 leading-relaxed max-w-2xl mx-auto">
            本サービスが提供するシグナルおよびスコアは情報提供のみを目的としており、特定の有価証券の売買を勧誘するものではありません。
            投資判断はご自身の責任において行ってください。過去のシグナル精度は将来の利益を保証するものではありません。
          </p>
          <p className="text-xs text-gray-600">
            &copy; {new Date().getFullYear()} StockSignal — 日本株 AI シグナル
          </p>
        </div>
      </div>
    </footer>
  );
}
