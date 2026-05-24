import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Mermaid 客户端渲染模块
// 使用 mermaid v10+ 的 mermaid.run() API

let mermaidModulePromise = null;
let initialized = false;

export function onInitialRouteRender() {
  if (!ExecutionEnvironment.canUseDOM) return;

  renderMermaidBlocks();
}

export function onRouteDidUpdate() {
  if (!ExecutionEnvironment.canUseDOM) return;

  renderMermaidBlocks();
}

function getMermaidModule() {
  if (!mermaidModulePromise) {
    mermaidModulePromise = import('mermaid');
  }
  return mermaidModulePromise;
}

function initializeMermaid(mermaid) {
  if (initialized) return;

  mermaid.default.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: {
      useMaxWidth: true,
      htmlLabels: true,
    },
  });

  initialized = true;
}

function renderMermaidBlocks() {
  getMermaidModule()
    .then((mermaid) => {
      initializeMermaid(mermaid);

      const nodes = collectMermaidNodes();
      if (nodes.length === 0) return;

      window.setTimeout(() => {
        mermaid.default.run({ nodes }).catch((err) => {
          console.error('[Mermaid] Render failed:', err);
        });
      }, 0);
    })
    .catch((err) => {
      console.error('[Mermaid] Failed to load:', err);
    });
}

// 预处理 mermaid 代码块。Docusaurus 默认会把 ```mermaid 输出成普通 Prism
// 代码块；这里在浏览器端替换成 Mermaid 可渲染节点。
function collectMermaidNodes() {
  const codeBlocks = Array.from(
    document.querySelectorAll('pre code.language-mermaid, pre.language-mermaid, code.language-mermaid, .language-mermaid')
  );
  const nodes = [];
  const seenContainers = new Set();

  codeBlocks.forEach((block) => {
    const container =
      block.closest('[data-mermaid-container="true"]') ||
      block.closest('[class*="codeBlockContainer"]') ||
      block.closest('pre') ||
      block;

    if (seenContainers.has(container) || container.hasAttribute('data-mermaid-rendered')) {
      return;
    }
    seenContainers.add(container);

    let code = extractCodeText(block);
    code = decodeHtmlEntities(code);
    if (!code) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'mermaid-diagram';
    wrapper.setAttribute('data-mermaid-container', 'true');
    wrapper.setAttribute('data-mermaid-rendered', 'true');

    const diagram = document.createElement('div');
    diagram.className = 'mermaid';
    diagram.textContent = code;

    wrapper.appendChild(diagram);
    container.replaceWith(wrapper);
    nodes.push(diagram);
  });

  return nodes;
}

function extractCodeText(block) {
  const lineNodes = block.querySelectorAll?.('.token-line');
  if (lineNodes?.length) {
    return Array.from(lineNodes)
      .map((line) => line.textContent || '')
      .join('\n');
  }

  return block.textContent || '';
}

// 解码 HTML 实体
function decodeHtmlEntities(text) {
  const textarea = document.createElement('textarea');
  textarea.innerHTML = text;
  let decoded = textarea.value;

  // 额外处理常见的 HTML 实体
  decoded = decoded
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/<br\s*\/?>/gi, '<br/>')  // 标准化 br 标签
    .replace(/\u00a0/g, ' ')  // 替换 non-breaking space
    .trim();

  return decoded;
}
