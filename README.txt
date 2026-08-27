编程智能体：Tiny Coding Agent

Git 仓库地址：提交前填写本项目公开仓库地址。

运行：
1. 安装 Python 3.10+。
2. 设置环境变量 OPENAI_API_KEY；如使用兼容网关，再设置 LLM_BASE_URL 和 OPENAI_MODEL。
3. 在本目录执行：python -m coding_agent "请在当前目录创建 hello.py 并运行它"。
   默认工作区为当前目录，也可用 --workspace 指定；--auto-approve 用于演示时自动批准命令。

特色：
这是一个不依赖 Agent 框架的 OpenAI 兼容 coding agent。核心循环由项目自行实现：保存对话历史、把本地工具描述发送给模型、解析 tool_calls、在受限工作区执行读写文件和命令、把工具结果回传模型，并用最大步数和错误结果保证循环可终止。工具包括 list_files、read_file、write_file、run_command。路径会被限制在工作区内；执行命令默认需要人工确认，凭据只从环境变量读取。项目还包含完全离线的单元测试和一个 FakeClient，可在没有 API key 时验证核心逻辑。

真实任务演示建议：让 agent 为一个空目录创建一个 Python 单元测试、实现函数、运行测试并根据失败信息修复。视频中可展示工具调用、命令输出、文件变化，以及源码中的 Agent.run 和 ToolExecutor。

其它说明：本项目没有使用 LangChain、LlamaIndex、Agents SDK 或服务端托管的代码执行/文件工具；HTTP 请求、工具 schema、工具执行和错误处理均由本项目自行编写。
