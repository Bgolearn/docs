# BGOlearn Documentation

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Deutsch](README.de.md)

This repository hosts the static documentation website for [BGOlearn](https://github.com/BGOlearn), including guides to Bayesian global optimization, optimization strategies, surrogate models, acquisition functions, and practical examples.

## Documentation

The documentation is available from the following maintained endpoints:

- **GitHub Pages (recommended):** <https://bgolearn.github.io/docs/>
- **Netlify:** <https://bgolearn.netlify.app/>

Both endpoints serve the same documentation. GitHub Pages is maintained as an alternative access route for users in countries or network environments where `netlify.app` may be unavailable.

## Deployment

The published static site is located in the [`html/`](html/) directory. Every push to the `main` branch automatically deploys this directory to GitHub Pages through GitHub Actions.

For the initial setup, configure **Settings → Pages → Build and deployment → Source** as **GitHub Actions** in the repository.
