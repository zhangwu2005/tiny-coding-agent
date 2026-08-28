编程智能体：Tiny Coding Agent

Git 仓库地址：https://github.com/zhangwu2005/tiny-coding-agent

运行
1. 安装 Python 3.10 或更高版本。
2. 在 PowerShell 中进入项目目录并设置凭据（不得写入代码或视频）：
   cd D:\桌面\mini-coding-agent\coding_agent
   $env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
   $env:DEEPSEEK_MODEL="deepseek-v4-flash"
3. 运行：
   python -m coding_agent --workspace . "创建 hello.py，运行它，并说明结果"
   命令默认需人工确认；演示时可加 --auto-approve。

测试
执行 python -m tests.test_agent，无需 API Key 或第三方依赖。

特色
这是一个不依赖 Agent 框架的 OpenAI 兼容 coding agent。项目自行实现对话历史、模型 HTTP 请求、tool calling 解析、循环终止、错误处理和本地工具执行。模型可调用 list_files、read_file、write_file 和 run_command；文件路径限制在工作区内，命令默认需人工批准。

提交前检查
最终 zip 只放 README.txt 和不超过 2 分钟、200 MB 的 mp4 视频；不得提交 API Key。
