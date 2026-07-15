# BGOlearn アルゴリズムマニュアル

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Deutsch](README.de.md)

このリポジトリは、[BGOlearn](https://github.com/BGOlearn) の静的ドキュメントサイトを管理しています。ベイズ大域最適化、最適化戦略、サロゲートモデル、獲得関数、および実践例を掲載しています。

## オンラインドキュメント

ドキュメントは、継続的に維持される以下のサイトから利用できます。

- **GitHub Pages（推奨）：** <https://bgolearn.github.io/docs/>
- **Netlify：** <https://bgolearn.netlify.app/>

両方のサイトは同一のドキュメントを提供します。`netlify.app` へアクセスできない国やネットワーク環境の利用者も BGOlearn アルゴリズムマニュアルを閲覧できるよう、GitHub Pages を代替のアクセス先として維持しています。

## デプロイ

公開される静的サイトは [`html/`](html/) ディレクトリにあります。`main` ブランチへの各プッシュで、GitHub Actions がこのディレクトリを GitHub Pages に自動デプロイします。

初回設定時は、リポジトリの **Settings → Pages → Build and deployment → Source** で **GitHub Actions** を選択してください。
