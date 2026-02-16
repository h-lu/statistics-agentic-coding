import React, { useEffect } from 'react';
import Layout from '@theme-original/Layout';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Mermaid 渲染函数（仅在客户端执行）
let mermaidInitialized = false;

async function renderMermaid() {
  if (!ExecutionEnvironment.canUseDOM) return;

  // 动态导入 mermaid
  const mermaid = (await import('mermaid')).default;

  // 只初始化一次
  if (!mermaidInitialized) {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
    });
    mermaidInitialized = true;
  }

  // 查找所有 mermaid 代码块并渲染
  const mermaidBlocks = document.querySelectorAll('.language-mermaid');
  for (let i = 0; i < mermaidBlocks.length; i++) {
    const block = mermaidBlocks[i];
    const code = block.textContent;
    const id = `mermaid-${Date.now()}-${i}`;

    try {
      const { svg } = await mermaid.render(id, code);
      // 创建一个容器来显示 SVG
      const container = document.createElement('div');
      container.className = 'mermaid-svg';
      container.innerHTML = svg;
      // 替换代码块为 SVG
      const codeBlock = block.closest('.codeBlockContainer_Ckt0') || block.closest('pre');
      if (codeBlock) {
        codeBlock.replaceWith(container);
      }
    } catch (err) {
      console.error('Mermaid render error:', err);
    }
  }
}

export default function LayoutWrapper(props) {
  useEffect(() => {
    // 在页面加载后渲染 mermaid
    renderMermaid();
  }, []); // 空数组表示只在组件挂载时执行一次

  return <Layout {...props} />;
}
