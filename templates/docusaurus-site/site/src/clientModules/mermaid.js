import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Mermaid 渲染模块
let mermaidInstance = null;
let mermaidInitialized = false;
let renderScheduled = false;

async function loadMermaid() {
  if (!ExecutionEnvironment.canUseDOM) return null;

  if (!mermaidInstance) {
    try {
      console.log('[Mermaid] Loading library...');
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
      console.log('[Mermaid] Initializing...');
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
  if (!ExecutionEnvironment.canUseDOM) {
    console.log('[Mermaid] Not in browser, skipping');
    return;
  }

  // 防止重复调度
  if (renderScheduled) {
    console.log('[Mermaid] Render already scheduled, skipping');
    return;
  }
  renderScheduled = true;

  console.log('[Mermaid] Starting render...');

  try {
    const mermaid = await initMermaid();
    if (!mermaid) {
      console.error('[Mermaid] Cannot render: library not available');
      renderScheduled = false;
      return;
    }

    // 查找所有 mermaid 代码块 - 使用更具体的选择器
    // Prism 处理后，language-mermaid 在 pre 元素上
    const codeBlocks = document.querySelectorAll('pre.language-mermaid, pre.prism-code.language-mermaid');
    console.log(`[Mermaid] Found ${codeBlocks.length} code blocks with pre.language-mermaid`);

    // 如果没找到，尝试备用选择器
    let blocks = codeBlocks;
    if (blocks.length === 0) {
      blocks = document.querySelectorAll('.language-mermaid');
      console.log(`[Mermaid] Fallback found ${blocks.length} blocks with .language-mermaid`);
    }
    const finalBlocks = Array.from(blocks);

    if (finalBlocks.length === 0) {
      console.log('[Mermaid] No code blocks found');
      renderScheduled = false;
      return;
    }

    console.log(`[Mermaid] Processing ${finalBlocks.length} blocks...`);

    for (let i = 0; i < finalBlocks.length; i++) {
      const block = finalBlocks[i];

      // 检查是否已经渲染过
      if (block.dataset.mermaidRendered === 'true') {
        console.log(`[Mermaid] Block ${i} already rendered`);
        continue;
      }

      // 获取 mermaid 代码
      const code = block.textContent || '';
      console.log(`[Mermaid] Block ${i} code length: ${code.length}, first 50 chars: "${code.substring(0, 50).replace(/\n/g, '\\n')}"`);

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

        // 找到代码块容器 - 从 pre 元素向上找到最外层容器
        // HTML 结构: div.codeBlockContainer > div.codeBlockContent > pre.prism-code
        const container = block.closest('.codeBlockContainer_Ckt0') ||
                          block.closest('.codeBlockContent_biex') ||
                          block.closest('[class*="codeBlockContainer"]') ||
                          block.parentElement?.parentElement;

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

    console.log('[Mermaid] Render complete');
  } catch (err) {
    console.error('[Mermaid] Unexpected error during render:', err);
  } finally {
    renderScheduled = false;
  }
}

// Docusaurus client module 生命周期钩子
export function onRouteDidUpdate() {
  console.log('[Mermaid] onRouteDidUpdate called');
  // 延迟执行确保 DOM 已更新
  setTimeout(renderMermaid, 300);
}

export function onInitialRouteRender() {
  console.log('[Mermaid] onInitialRouteRender called');
  setTimeout(renderMermaid, 500);
}
