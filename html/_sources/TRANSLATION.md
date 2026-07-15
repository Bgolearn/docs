# Translation workflow

This book exposes five language versions:

- `index.md` and other root Markdown files: English
- `zh/`: Chinese
- `ja/`: Japanese
- `ko/`: Korean
- `de/`: German

The language switcher is loaded from `_static/language-switcher.js` and maps each page to the same page path in the selected language. For example:

- `getting_started.html`
- `zh/getting_started.html`
- `ja/getting_started.html`
- `ko/getting_started.html`
- `de/getting_started.html`

When adding a new page, add the English Markdown file first, then create matching files in each language directory. Keep filenames and subdirectories identical so the one-click language switcher can find the corresponding page.
