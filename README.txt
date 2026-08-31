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
   命令默认需人工确认；演示时可加 --auto-approve，高风险命令仍需显式确认。
   控制器默认要求测试级证据；可用 --verification-policy none|syntax|test|full 调整。
   若证据不足，必须由用户输入 accept，或预先显式添加 --accept-incomplete 才能验收。

测试
执行 python -m tests.test_agent，可完成 25 项离线检查，无需 API Key 或第三方依赖。

特色
这是一个不依赖 Agent 框架的 OpenAI 兼容 coding agent。项目自行实现对话历史、模型 HTTP 请求、tool calling 解析、循环终止、错误处理和本地工具执行。六个工具支持目录浏览、分段读取、跨文件搜索、创建文件、唯一匹配替换和命令执行；工具结果有上下文上限。

小型护栏体现“可验证的自主性”：模型只提出工具行动和完成建议；控制器依据当前修改版本上的语法检查、静态分析、定向测试或完整测试记录裁决流程；证据不足时用户拥有最终验收权。每条记录保留命令、类型、文件版本快照、退出码和结果，旧版本证据不会证明新代码，较弱的成功检查也不能掩盖较强测试的失败。

已有文件采用“先读后写”的版本校验，避免盲目覆盖和过期上下文；测试失败会生成带错误证据的反思检查点；重复工具批次会提前熔断；命令执行给出可解释的风险分级。文件路径限制在工作区内，命令默认需人工批准。

提交前检查
最终 zip 只放 README.txt 和不超过 2 分钟、200 MB 的 mp4 视频；不得提交 API Key。
