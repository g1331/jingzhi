(() => {
  "use strict";

  const content = document.getElementById("content");
  marked.setOptions({ gfm: true, breaks: true });

  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    node.removeAttribute("href");
    node.removeAttribute("srcset");
    if (node.hasAttribute("src") && !node.getAttribute("src").startsWith("data:image/")) {
      node.removeAttribute("src");
    }
  });

  function normalizeMathDelimiters(source) {
    return source
      .replace(/\\\[/g, () => "$$")
      .replace(/\\\]/g, () => "$$")
      .replace(/\\\(/g, () => "$")
      .replace(/\\\)/g, () => "$");
  }

  function protectMath(source) {
    const expressions = [];
    const protectedSource = source.replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g, (match) => {
      const index = expressions.push(match) - 1;
      return `MATHPLACEHOLDER${index}END`;
    });
    return { protectedSource, expressions };
  }

  function restoreMath(expressions) {
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    const placeholder = /MATHPLACEHOLDER(\d+)END/g;
    textNodes.forEach((node) => {
      if (!placeholder.test(node.nodeValue)) return;
      placeholder.lastIndex = 0;
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      let match;
      while ((match = placeholder.exec(node.nodeValue)) !== null) {
        fragment.append(node.nodeValue.slice(cursor, match.index));
        fragment.append(expressions[Number(match[1])] || "");
        cursor = placeholder.lastIndex;
      }
      fragment.append(node.nodeValue.slice(cursor));
      node.replaceWith(fragment);
      placeholder.lastIndex = 0;
    });
  }

  window.renderMarkdown = (source) => {
    if (!source || !source.trim()) {
      content.className = "empty-state";
      content.textContent = "回答、总结、知识点和错题会显示在这里。";
      return;
    }

    const markdown = normalizeMathDelimiters(source);
    const { protectedSource, expressions } = protectMath(markdown);
    const rendered = marked.parse(protectedSource);
    content.innerHTML = DOMPurify.sanitize(rendered, {
      USE_PROFILES: { html: true },
      FORBID_TAGS: ["audio", "button", "form", "iframe", "input", "object", "script", "style", "textarea", "video"],
      FORBID_ATTR: ["contenteditable", "form", "srcdoc"]
    });
    content.className = "markdown-body";
    restoreMath(expressions);
    renderMathInElement(content, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false }
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      throwOnError: false,
      strict: false,
      trust: false
    });
  };
})();
