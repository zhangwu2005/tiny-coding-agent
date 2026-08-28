编程智能体：Tiny Coding Agent

Git 仓库地址：提交前替换为你的公开仓库地址。

运行
1. 安装 Python 3.10 或更高版本。
2. 在 PowerShell 中进入本目录：
   cd D:\桌面\mini-coding-agent\coding_agent
3. 设置模型环境变量（不要写入代码、README 或视频）。使用 DeepSeek 时：
   $env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
   # 可选，默认使用 deepseek-v4-flash：
   $env:DEEPSEEK_MODEL="deepseek-v4-flash"
   # 网络受限时可填 OpenAI 兼容代理地址；官方地址为 https://api.deepseek.com：
   # $env:DEEPSEEK_BASE_URL="https://你的代理地址/v1"
   # 请求超时秒数，默认 45：$env:LLM_TIMEOUT="45"
   # 其他 OpenAI 兼容服务可使用 OPENAI_API_KEY、OPENAI_MODEL 和 LLM_BASE_URL。
4. 运行：
   python -m coding_agent --workspace . "创建 hello.py，运行它，并说明结果"
   执行命令前默认会请求人工确认；录制演示时可加 --auto-approve。

测试
在本目录执行：
   python -m tests.test_agent
该测试不需要 API Key 或第三方依赖。若已安装 pytest，也可以执行 pytest -q。

特色
这是一个不依赖 Agent 框架的 OpenAI 兼容 coding agent。项目自行实现了对话历史、模型 HTTP 请求、tool calling 解析、最大步数终止、错误反馈和本地工具执行。模型可调用 list_files、read_file、write_file 和 run_command；文件路径限制在工作区内，命令默认需要人工批准。DeepSeek Key 只能调用 DeepSeek 账号可用的模型，不代表所有模型厂商均可调用。

提交前检查
确认公开 Git 仓库地址已补上，且仓库历史完整。按题目要求，最终 zip 只放 README.txt 和不超过 2 分钟、200 MB 的 mp4 视频；API Key 等凭据不得提交。
