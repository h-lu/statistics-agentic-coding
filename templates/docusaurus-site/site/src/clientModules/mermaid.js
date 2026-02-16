import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import mermaid from 'mermaid';

// Mermaid 客户端渲染模块
// 注意：这是客户端渲染方案，SSR 时会输出原始代码块，客户端 JS 加载后渲染为 SVG

let isInitialized = false;

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
    sequence: {
      useMaxWidth: true,
    },
    gantt: {
      useMaxWidth: true,
    },
  });
  console.log('[Mermaid] Initialized with config');
}

async function renderMermaidBlocks() {
  if (!ExecutionEnvironment.canUseDOM) return;

  initMermaid();

  // 查找所有 mermaid 代码块
  // Docusaurus/Prism 生成的结构: div.language-mermaid > pre > code
  const mermaidBlocks = document.querySelectorAll('.language-mermaid');

  console.log(`[Mermaid] Found ${mermaidBlocks.length} mermaid blocks`);

  let renderedCount = 0;

  for (let i = 0; i < mermaidBlocks.length; i++) {
    const block = mermaidBlocks[i];

    // 跳过已渲染的
    if (block.hasAttribute('data-mermaid-rendered')) continue;

    // 获取代码内容
    // 可能是 pre.language-mermaid > code 或 div.language-mermaid > pre > code
    let codeElement;
    let container;

    if (block.tagName === 'PRE') {
      // pre.language-mermaid > code
      codeElement = block.querySelector('code') || block;
      container = block.closest('.codeBlockContainer_Ckt0') ||
                  block.closest('.codeBlockContent_biex') ||
                  block.closest('[class*="codeBlockContainer"]') ||
                  block.parentElement;
    } else if (block.tagName === 'DIV') {
      // div.language-mermaid > pre > code
      codeElement = block.querySelector('code') || block.querySelector('pre') || block;
      container = block.closest('.codeBlockContainer_Ckt0') ||
                  block.closest('.codeBlockContent_biex') ||
                  block.closest('[class*="codeBlockContainer"]') ||
                  block;
    } else {
      codeElement = block;
      container = block;
    }

    // 获取纯文本代码（去除 token span）
    const code = codeElement.textContent || '';

    if (!code.trim()) {
      console.log(`[Mermaid] Block ${i} is empty, skipping`);
      continue;
    }

    // 检查是否是有效的 mermaid 关键字开头
    const validStarts = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram',
                         'erDiagram', 'gantt', 'pie', 'journey', 'gitGraph', 'mindmap', 'timeline'];
    const codeStart = code.trim().split('\n')[0].trim();
    const isValidMermaid = validStarts.some(start => codeStart.startsWith(start));

    if (!isValidMermaid) {
      console.log(`[Mermaid] Block ${i} doesn't start with valid mermaid keyword: ${codeStart.substring(0, 50)}...`);
      continue;
    }

    const id = `mermaid-graph-${Date.now()}-${i}`;

    try {
      console.log(`[Mermaid] Rendering block ${i}...`);
      const { svg } = await mermaid.render(id, code);

      // 标记已渲染
      block.setAttribute('data-mermaid-rendered', 'true');
      renderedCount++;

      // 创建 SVG 容器
      const svgContainer = document.createElement('div');
      svgContainer.className = 'mermaid-svg-container';
      svgContainer.style.cssText = 'overflow-x: auto; padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 1rem 0; border: 1px solid #e9ecef;';
      svgContainer.innerHTML = svg;

      // 替换原始代码块容器
      if (container && container !== block) {
        container.replaceWith(svgContainer);
      } else {
        block.replaceWith(svgContainer);
      }

      console.log(`[Mermaid] Successfully rendered block ${i}`);
    } catch (err) {
      console.error(`[Mermaid] Error rendering block ${i}:`, err.message);
      console.error(`[Mermaid] Code was:`, code.substring(0, 100));
      // 不标记为已渲染，允许后续重试
    }
  }

  console.log(`[Mermaid] Total rendered: ${renderedCount}`);
  return renderedCount;
}

// 设置 MutationObserver 监听 DOM 变化（SPA 路由切换）
function setupObserver() {
  if (!ExecutionEnvironment.canUseDOM) return;

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        const hasMermaidBlocks = Array.from(mutation.addedNodes).some(node => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            return node.querySelector?.('.language-mermaid') ||
                   node.matches?.('.language-mermaid');
          }
          return false;
        });

        if (hasMermaidBlocks) {
          console.log('[Mermaid] New mermaid blocks detected via observer');
          setTimeout(renderMermaidBlocks, 150);
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
export function onRouteDidUpdate({ location, previousLocation }) {
  // 仅在路由实际变化时渲染
  if (location.pathname !== previousLocation?.pathname) {
    console.log('[Mermaid] Route changed, triggering render');
    setTimeout(renderMermaidBlocks, 200);
  }
}

export function onInitialRouteRender() {
  console.log('[Mermaid] Initial route render');
  setTimeout(() => {
    renderMermaidBlocks();
    setupObserver();
  }, 300);
}
