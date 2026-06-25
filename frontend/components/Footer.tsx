"use client";
import { useLang } from "@/contexts/LanguageContext";
import { UI } from "@/lib/i18n";

export default function Footer() {
  const { lang } = useLang();
  const ui = UI[lang];

  return (
    <footer className="mt-12">
      <div className="border-t border-gray-800 py-6 px-4">
        <div className="max-w-7xl mx-auto text-center space-y-2">
          <p className="text-xs text-gray-600 leading-relaxed max-w-2xl mx-auto">
            {ui.footerDisclaimer}
          </p>
          <p className="text-xs text-gray-600">
            &copy; {new Date().getFullYear()} {ui.footerCopy}
          </p>
        </div>
      </div>
    </footer>
  );
}
