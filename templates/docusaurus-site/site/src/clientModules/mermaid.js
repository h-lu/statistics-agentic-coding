import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Mermaid 渲染模块
let mermaidInstance = null;
let mermaidInitialized = false;
let renderAttempted = false;

async function loadMermaid() {
  if (!ExecutionEnvironment.canUseDOM) return null;

  if (!mermaidInstance) {
    try {
      const mermaidModule = await import('mermaid');
      mermaidInstance = mermaidModule.default;
      console.log('[Mermaid] Library loaded successfully');
    } catch (err) {
      console.error('[Mermaid] Failed to load library:', err);
      return null;
    }
  }

  return mermaidInstance;
}

async function initMermaid() {
  const mermaid = await loadMermaid();
  if (!mermaid) return null;

  if (!mermaidInitialized) {
    try {
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
      console.log('[Mermaid] Initialized successfully');
    } catch (err) {
      console.error('[Mermaid] Failed to initialize:', err);
      return null;
    }
  }

  return mermaid;
}

async function renderMermaid() {
  if (!ExecutionEnvironment.canUseDOM) return;

  // 防止重复渲染
  if (renderAttempted) {
    console.log('[Mermaid] Already attempted render, skipping');
    return;
  }
  renderAttempted = true;

  console.log('[Mermaid] Starting render...');

  const mermaid = await initMermaid();
  if (!mermaid) {
    console.error('[Mermaid] Cannot render: library not available');
    return;
  }

  // 查找所有 mermaid 代码块
  const codeBlocks = document.querySelectorAll('.language-mermaid');
  console.log(`[Mermaid] Found ${codeBlocks.length} code blocks`);

  if (codeBlocks.length === 0) {
    console.log('[Mermaid] No code blocks found');
    return;
  }

  for (let i = 0; i < codeBlocks.length; i++) {
    const block = codeBlocks[i];

    // 检查是否已经渲染过
    if (block.dataset.mermaidRendered === 'true') {
      console.log(`[Mermaid] Block ${i} already rendered`);
      continue;
    }

    // 获取 mermaid 代码
    const code = block.textContent || '';
    if (!code.trim()) {
      console.log(`[Mermaid] Block ${i} is empty`);
      continue;
    }

    const id = `mermaid-diagram-${Date.now()}-${i}`;
    console.log(`[Mermaid] Rendering block ${i}...`);

    try {
      // 渲染 SVG
      const { svg } = await mermaid.render(id, code);
      console.log(`[Mermaid] Block ${i} rendered successfully`);

      // 标记为已渲染
      block.dataset.mermaidRendered = 'true';

      // 找到代码块容器 - 尝试多种选择器
      const container = block.closest('.codeBlockContainer_Ckt0') ||
                        block.closest('.codeBlockContent_biex') ||
                        block.closest('pre')?.parentElement ||
                        block.parentElement;

      if (container) {
        // 创建 SVG 容器
        const svgContainer = document.createElement('div');
        svgContainer.className = 'mermaid-svg';
        svgContainer.style.cssText = 'overflow-x: auto; padding: 1rem; background: #f6f8fa; border-radius: 8px;';
        svgContainer.innerHTML = svg;

        // 替换整个代码块容器
        container.replaceWith(svgContainer);
        console.log(`[Mermaid] Block ${i} replaced with SVG`);
      } else {
        console.warn(`[Mermaid] Block ${i}: Could not find container`);
      }
    } catch (renderErr) {
      console.error(`[Mermaid] Block ${i} render error:`, renderErr);
      // 不隐藏错误，保留原始代码块
    }
  }
}

// Docusaurus client module 生命周期钩子
export function onRouteDidUpdate() {
  console.log('[Mermaid] Route updated, scheduling render...');
  // 重置渲染标记以允许新页面渲染
  renderAttempted = false;
  // 延迟执行确保 DOM 已更新
  setTimeout(renderMermaid, 500);
}

export function onInitialRouteRender() {
  console.log('[Mermaid] Initial route render, scheduling render...');
  setTimeout(renderMermaid, 800);
}
