#!/bin/bash
# MCP 工具安装脚本

echo "=== MCP 工具安装脚本 ==="
echo ""

# 检查是否已有配置
if kimi mcp list | grep -q "context7"; then
    echo "✓ Context7 MCP 已配置"
else
    echo "安装 Context7 MCP..."
    echo "请先访问 https://context7.com 获取 API Key"
    echo ""
    read -p "请输入 Context7 API Key (格式: ctx7sk-...): " CONTEXT7_KEY
    
    if [ -n "$CONTEXT7_KEY" ]; then
        kimi mcp add --transport http context7 https://mcp.context7.com/mcp \
            --header "CONTEXT7_API_KEY: $CONTEXT7_KEY"
        echo "✓ Context7 MCP 安装完成"
    else
        echo "跳过 Context7 安装（未提供 API Key）"
    fi
fi

echo ""

# 检查 Perplexity（可选）
if ! kimi mcp list | grep -q "perplexity"; then
    echo "是否安装 Perplexity MCP? (需要 API Key from https://perplexity.ai)"
    read -p "请输入 Perplexity API Key (或回车跳过): " PERPLEXITY_KEY
    
    if [ -n "$PERPLEXITY_KEY" ]; then
        kimi mcp add --transport http perplexity https://mcp.perplexity.ai/mcp \
            --header "PERPLEXITY_API_KEY: $PERPLEXITY_KEY"
        echo "✓ Perplexity MCP 安装完成"
    fi
fi

echo ""
echo "=== 当前 MCP 配置 ==="
kimi mcp list
echo ""
echo "测试连接:"
kimi mcp test exa
