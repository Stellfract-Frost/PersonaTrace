"""兼容入口：`python mcp_server.py`（MCP 客户端配置里通常写这个路径）。

实现在 `personatrace/mcp_server.py`。装了包之后也可以：
    personatrace-mcp
    python -m personatrace.mcp_server
"""

from personatrace.mcp_server import main

if __name__ == "__main__":
    raise SystemExit(main())
