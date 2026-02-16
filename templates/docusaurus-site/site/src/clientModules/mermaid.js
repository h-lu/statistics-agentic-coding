import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Mermaid 渲染模块
let mermaid = null;
let mermaidInitialized = false;

async function initMermaid() {
  if (!ExecutionEnvironment.canUseDOM) return;

  if (!mermaid) {
    mermaid = (await import('mermaid')).default;
  }

  if (!mermaidInitialized) {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
      },
    });
    mermaidInitialized = true;
  }

  return mermaid;
}

async function renderMermaid() {
  if (!ExecutionEnvironment.canUseDOM) return;

  const m = await initMermaid();
  if (!m) return;

  // 查找所有 mermaid 代码块
  const codeBlocks = document.querySelectorAll('.language-mermaid');

  for (let i = 0; i < codeBlocks.length; i++) {
    const block = codeBlocks[i];

    // 检查是否已经渲染过
    if (block.dataset.mermaidRendered === 'true') continue;

    // 获取 mermaid 代码
    const code = block.textContent || '';
    if (!code.trim()) continue;

    const id = `mermaid-${Date.now()}-${i}`;

    try {
      // 渲染 SVG
      const { svg } = await m.render(id, code);

      // 标记为已渲染
      block.dataset.mermaidRendered = 'true';

      // 找到代码块容器
      const container = block.closest('.codeBlockContainer_Ckt0') ||
                        block.closest('.codeBlockContent_biex') ||
                        block.parentElement;

      if (container) {
        // 创建 SVG 容器
        const svgContainer = document.createElement('div');
        svgContainer.className = 'mermaid-svg';
        svgContainer.style.cssText = 'overflow-x: auto; padding: 1rem;';
        svgContainer.innerHTML = svg;

        // 替换整个代码块容器
        container.replaceWith(svgContainer);
      }
    } catch (renderErr) {
      console.error('Mermaid render error:', renderErr);
    }
  }
}

// Docusaurus client module 生命周期钩子
export function onRouteDidUpdate({ location }) {
  // 延迟执行确保 DOM 已更新
  setTimeout(renderMermaid, 300);
}

export default function clientModule() {
  // 初始加载时也渲染
  if (ExecutionEnvironment.canUseDOM) {
    setTimeout(renderMermaid, 500);
  }
}
