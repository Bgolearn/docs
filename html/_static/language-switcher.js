(function () {
  "use strict";

  var languages = [
    { code: "en", label: "English" },
    { code: "zh", label: "中文" },
  ];
  var languageCodes = languages.map(function (language) {
    return language.code;
  });
  var uiText = {
    en: { language: "Language", current: "English" },
    zh: { language: "语言", current: "中文" },
  };
  var localizedTitles = {
    "Bgolearn Documentation": {
      zh: "Bgolearn 手册",
    },
    "Getting Started": { zh: "入门" },
    "Installation & Quick Start": { zh: "安装与快速开始" },
    "Basic Concepts": { zh: "基本概念" },
    "Your First Optimization": { zh: "第一次优化" },
    "Single-Objective Optimization": { zh: "单目标优化" },
    "API Reference": { zh: "接口参考" },
    "Acquisition Functions": { zh: "采集函数" },
    "Surrogate Models": { zh: "代理模型" },
    "Optimization Strategies": { zh: "优化策略" },
    "Multi-Objective Optimization": { zh: "多目标优化" },
    "MultiBgolearn Overview": { zh: "MultiBgolearn 概览" },
    "Multi-Objective Concepts": { zh: "多目标概念" },
    "MOBO Algorithms": { zh: "多目标贝叶斯优化算法" },
    "Pareto Optimization": { zh: "帕累托优化" },
    "User Interfaces": { zh: "用户界面" },
    "BgoFace GUI": { zh: "BgoFace 图形界面" },
    "Applications": { zh: "应用" },
    "Materials Discovery": { zh: "材料发现" },
    "Examples & Tutorials": { zh: "示例与教程" },
    "Single-Objective Examples": { zh: "单目标示例" },
    "Multi-Objective Examples": { zh: "多目标示例" },
  };

  function getPageName() {
    if (
      typeof DOCUMENTATION_OPTIONS !== "undefined" &&
      DOCUMENTATION_OPTIONS.pagename
    ) {
      return DOCUMENTATION_OPTIONS.pagename;
    }

    var pathname = window.location.pathname.replace(/\/$/, "/index.html");
    var htmlRoot = "/_build/html/";
    var rootIndex = pathname.indexOf(htmlRoot);

    if (rootIndex !== -1) {
      pathname = pathname.slice(rootIndex + htmlRoot.length);
    }

    return pathname.replace(/^\//, "").replace(/\.html$/, "");
  }

  function getPageParts(pageName) {
    var parts = pageName.split("/").filter(Boolean);
    var currentLanguage = "en";

    if (parts.length && languageCodes.indexOf(parts[0]) !== -1 && parts[0] !== "en") {
      currentLanguage = parts[0];
      parts = parts.slice(1);
    }

    if (!parts.length) {
      parts = ["index"];
    }

    return {
      currentLanguage: currentLanguage,
      pagePath: parts.join("/"),
      depth: pageName.split("/").filter(Boolean).length - 1,
    };
  }

  function buildHref(targetLanguage, pagePath, depth) {
    var rootPrefix = depth > 0 ? "../".repeat(depth) : "";
    var targetPath = targetLanguage === "en" ? pagePath : targetLanguage + "/" + pagePath;
    return rootPrefix + targetPath + ".html";
  }

  function setStoredLanguage(language) {
    try {
      window.localStorage.setItem("bgolearn-language", language);
    } catch (error) {
      return;
    }
  }

  function getStoredLanguage() {
    try {
      return window.localStorage.getItem("bgolearn-language");
    } catch (error) {
      return null;
    }
  }

  function getSiteRootPath(depth) {
    var path = window.location.pathname;
    var directory = path.endsWith("/") ? path : path.slice(0, path.lastIndexOf("/") + 1);

    for (var i = 0; i < depth; i += 1) {
      directory = directory.replace(/\/[^/]+\/$/, "/");
    }

    return directory || "/";
  }

  function pagePathFromUrl(url, rootPath) {
    var path = decodeURI(url.pathname);
    rootPath = decodeURI(rootPath);

    if (path.indexOf(rootPath) !== 0) {
      return null;
    }

    var relative = path.slice(rootPath.length).replace(/^\//, "");
    if (!relative || relative.endsWith("/")) {
      relative += "index.html";
    }

    if (!relative.endsWith(".html")) {
      return null;
    }

    relative = relative.replace(/\.html$/, "");
    var parts = relative.split("/").filter(Boolean);

    if (parts.length && languageCodes.indexOf(parts[0]) !== -1 && parts[0] !== "en") {
      parts = parts.slice(1);
    }

    return parts.length ? parts.join("/") : "index";
  }

  function shouldIgnoreLink(link, url) {
    var href = link.getAttribute("href") || "";
    return (
      !href ||
      href.indexOf("#") === 0 ||
      href.indexOf("mailto:") === 0 ||
      href.indexOf("javascript:") === 0 ||
      link.hasAttribute("data-bgo-language-link") ||
      url.origin !== window.location.origin
    );
  }

  function keepLinksInLanguage(currentLanguage, currentDepth) {
    if (currentLanguage === "en") {
      return;
    }

    var rootPath = getSiteRootPath(currentDepth);

    document.querySelectorAll("a[href]").forEach(function (link) {
      var url;
      try {
        url = new URL(link.getAttribute("href"), window.location.href);
      } catch (error) {
        return;
      }

      if (shouldIgnoreLink(link, url)) {
        return;
      }

      var targetPage = pagePathFromUrl(url, rootPath);
      if (!targetPage) {
        return;
      }

      link.href = buildHref(currentLanguage, targetPage, currentDepth) + url.hash;
    });
  }

  function redirectToStoredLanguage(page) {
    var storedLanguage = getStoredLanguage();

    if (
      storedLanguage &&
      storedLanguage !== "en" &&
      storedLanguage !== page.currentLanguage &&
      languageCodes.indexOf(storedLanguage) !== -1
    ) {
      window.location.href = buildHref(storedLanguage, page.pagePath, page.depth);
      return true;
    }

    return false;
  }

  function createSwitcher() {
    var article = document.querySelector("article.bd-article");
    if (!article || document.querySelector(".bgolearn-language-switcher")) {
      return;
    }

    var page = getPageParts(getPageName());
    if (redirectToStoredLanguage(page)) {
      return;
    }
    setStoredLanguage(page.currentLanguage);

    var switcher = document.createElement("div");
    switcher.className = "dropdown bgolearn-language-switcher";
    switcher.setAttribute("aria-label", "Language switcher");

    var button = document.createElement("button");
    button.className = "btn btn-sm dropdown-toggle bgolearn-language-switcher__button";
    button.type = "button";
    button.setAttribute("data-bs-toggle", "dropdown");
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("title", (uiText[page.currentLanguage] || uiText.en).language);
    button.setAttribute("aria-label", (uiText[page.currentLanguage] || uiText.en).language);
    button.innerHTML =
      '<i class="fas fa-language" aria-hidden="true"></i>' +
      '<span class="bgolearn-language-switcher__current">' +
      ((uiText[page.currentLanguage] || uiText.en).current) +
      "</span>";
    switcher.appendChild(button);

    var menu = document.createElement("ul");
    menu.className = "dropdown-menu dropdown-menu-end";

    languages.forEach(function (language) {
      var item = document.createElement("li");
      var link = document.createElement("a");
      link.className = "dropdown-item bgolearn-language-switcher__link";
      link.href = buildHref(language.code, page.pagePath, page.depth);
      link.setAttribute("data-bgo-language-link", "true");
      link.textContent = language.label;
      link.addEventListener("click", function () {
        setStoredLanguage(language.code);
      });

      if (language.code === page.currentLanguage) {
        link.className += " active";
        link.setAttribute("aria-current", "page");
      }

      item.appendChild(link);
      menu.appendChild(item);
    });
    switcher.appendChild(menu);

    var headerButtons = document.querySelector(".article-header-buttons");
    if (headerButtons) {
      headerButtons.appendChild(switcher);
    } else {
      article.insertBefore(switcher, article.firstElementChild);
    }
    localizeNavigation(page.currentLanguage);
    keepLinksInLanguage(page.currentLanguage, page.depth);
  }

  function localizeNavigation(currentLanguage) {
    if (currentLanguage === "en") {
      return;
    }

    var selector = ".bd-sidebar-primary a, .bd-sidebar-primary .caption-text, .bd-toc a";
    document.querySelectorAll(selector).forEach(function (node) {
      var original = node.textContent.trim().replace(/^\d+\.?\s*/, "");
      var translation = localizedTitles[original] && localizedTitles[original][currentLanguage];

      if (translation) {
        node.textContent = node.textContent.replace(original, translation);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", createSwitcher);
  } else {
    createSwitcher();
  }
})();
