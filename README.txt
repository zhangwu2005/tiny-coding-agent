编程智能体：Tiny Coding Agent

Git 仓库：https://github.com/zhangwu2005/tiny-coding-agent

运行
1. 安装 Python 3.10+，进入项目目录。
2. 在 PowerShell 设置凭据（不得写入代码、README 或视频）：
   $env:DEEPSEEK_API_KEY="你的 API Key"
   $env:DEEPSEEK_MODEL="deepseek-v4-flash"
3. 执行：
   python -m coding_agent --workspace . "创建 hello.py，运行并验证"
   命令默认需确认；--auto-approve 批准非高风险命令。--verification-policy none|syntax|test|full 设置验证门槛；--test-provenance-policy allow|warn|independent 控制自改测试能否作为终止证据。证据不足由用户 accept；--context-limit 调整上下文预算。

测试
python -m tests.test_agent
共 35 项离线检查，无需 API Key 或第三方依赖。

特色
项目不依赖 Agent 框架，自行实现模型 HTTP 请求、对话历史、上下文压缩、tool calling 解析、本地工具、循环终止和错误处理。七个工具支持结构化计划、目录浏览、分段读取、跨文件搜索、文件写入、唯一匹配替换和命令执行。

模型只有行动与结束建议权：TaskPlan 的 ID、状态转换和完成证据由控制器校验，未完成计划会阻止自动结束；完整 transcript 保留审计记录，发给模型的副本超出预算时压缩为结构化状态与最近完整工具批次。文件修改采用先读后写和 revision 防护。

控制器按当前代码版本和文件角色记录语法检查、静态分析、定向测试和完整测试集；旧证据不证明新代码，自改测试标记来源风险。控制器判断验证是否充分，证据不足时用户拥有最终验收权。命令有风险分级与人工批准，测试失败会触发反思检查点，重复调用会熔断。

提交 zip 仅包含 README.txt 和不超过 2 分钟、200 MB 的 mp4 视频；不得包含 API Key。
