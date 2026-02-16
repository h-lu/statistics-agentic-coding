import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// 简化的 Mermaid 客户端模块 - 使用 mermaid.run() API

export function onInitialRouteRender() {
  if (!ExecutionEnvironment.canUseDOM) return;
  
  // 动态加载 mermaid
  import('mermaid').then((mermaid) => {
    mermaid.default.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
    });
    
    // 使用 mermaid.run() - 这是 mermaid v10+ 推荐的方式
    setTimeout(() => {
      mermaid.default.run({
        querySelector: '.language-mermaid',
        postRenderCallback: function(id) {
          console.log('[Mermaid] Rendered:', id);
        }
      });
    }, 500);
  }).catch(err => {
    console.error('[Mermaid] Failed to load:', err);
  });
}

export function onRouteDidUpdate() {
  if (!ExecutionEnvironment.canUseDOM) return;
  
  import('mermaid').then((mermaid) => {
    setTimeout(() => {
      mermaid.default.run({
        querySelector: '.language-mermaid',
      });
    }, 300);
  }).catch(err => {
    console.error('[Mermaid] Failed to load:', err);
  });
}
