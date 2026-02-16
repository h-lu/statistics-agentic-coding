import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import mermaid from 'mermaid';

// Mermaid 渲染模块
let isInitialized = false;
let renderCount = 0;

function initMermaid() {
  if (isInitialized) return;
  isInitialized = true;

  mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: {
      useMaxWidth: true,
      htmlLabels: true,
    },
  });
  console.log('[Mermaid] Initialized');
}

async function renderMermaidBlocks() {
  if (!ExecutionEnvironment.canUseDOM) return;

  initMermaid();

  // 查找所有未渲染的 mermaid 代码块
  const codeBlocks = document.querySelectorAll('pre.language-mermaid:not([data-mermaid-rendered])');
  const divBlocks = document.querySelectorAll('div.language-mermaid:not([data-mermaid-rendered])');

  const allBlocks = [...codeBlocks, ...divBlocks];
  console.log(`[Mermaid] Found ${allBlocks.length} blocks to render`);

  for (let i = 0; i < allBlocks.length; i++) {
    const block = allBlocks[i];
    const code = block.textContent || '';

    if (!code.trim()) continue;

    const id = `mermaid-${Date.now()}-${i}`;

    try {
      const { svg } = await mermaid.render(id, code);
      block.dataset.mermaidRendered = 'true';
      renderCount++;

      // 找到容器并替换
      const container = block.closest('.codeBlockContainer_Ckt0') ||
                        block.closest('.codeBlockContent_biex') ||
                        block.closest('[class*="codeBlockContainer"]') ||
                        block.parentElement?.parentElement ||
                        block.parentElement;

      if (container) {
        const svgContainer = document.createElement('div');
        svgContainer.className = 'mermaid-svg';
        svgContainer.style.cssText = 'overflow-x: auto; padding: 1rem; background: #f6f8fa; border-radius: 8px; margin: 1rem 0;';
        svgContainer.innerHTML = svg;
        container.replaceWith(svgContainer);
        console.log(`[Mermaid] Rendered block ${i}`);
      }
    } catch (err) {
      console.error(`[Mermaid] Error rendering block ${i}:`, err);
    }
  }
}

// 使用 MutationObserver 监听 DOM 变化
function setupObserver() {
  if (!ExecutionEnvironment.canUseDOM) return;

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === 'childList') {
        const hasNewMermaidBlocks = Array.from(mutation.addedNodes).some(node => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            return node.querySelector?.('pre.language-mermaid, div.language-mermaid') ||
                   node.matches?.('pre.language-mermaid, div.language-mermaid');
          }
          return false;
        });

        if (hasNewMermaidBlocks) {
          setTimeout(renderMermaidBlocks, 100);
          break;
        }
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  console.log('[Mermaid] Observer setup complete');
}

// Docusaurus 生命周期钩子
export function onRouteDidUpdate() {
  console.log('[Mermaid] onRouteDidUpdate');
  setTimeout(renderMermaidBlocks, 200);
}

export function onInitialRouteRender() {
  console.log('[Mermaid] onInitialRouteRender');
  setTimeout(() => {
    renderMermaidBlocks();
    setupObserver();
  }, 300);
}
