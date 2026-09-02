Tiny Coding Agent

一、Git仓库地址
https://github.com/zhangwu2005/tiny-coding-agent

二、如何运行
要求Python 3.10+。进入项目目录，录屏前在本机环境变量中设置DEEPSEEK_API_KEY。
运行真实任务：
python -m coding_agent --workspace generated_workspace --auto-approve --verification-policy test --test-provenance-policy allow "创建calculator.py，实现divide(a,b)，创建unittest测试并运行"
项目自检：
python -m tests.test_agent
命令默认需要人工批准；--auto-approve仅自动批准非高风险命令。

三、特色功能说明
项目不依赖Agent框架，自行实现模型请求、tool calling解析、对话历史、上下文管理、本地工具、错误处理和循环终止。大模型提出计划、工具调用和结束建议；Agent解析并调度：update_plan交给TaskPlan，文件与命令调用交给ToolExecutor；结果写回历史供模型继续决策。模型不再调用工具并提出结束时，Agent调用CompletionPolicy依据计划和验证证据决定是否结束。
ContextManager在上下文超限时保留原始任务、结构化状态和最近完整工具批次，完整transcript仍可审计。TaskPlan校验计划结构、状态转换和完成证据，防止模型口头宣布完成。ToolExecutor限制工作区路径；修改前比较内容修订标识，拒绝过期请求；替换要求目标唯一。
验证采用确定性规则：修改使旧证据失效，零测试不算成功，失败须由相同命令成功重跑清除。CompletionPolicy检查未解决失败、计划完成度和当前有效证据，模型不能降低验证门槛。

四、其他说明
coding_agent/是核心程序包；tests/包含35项离线测试；demo_workspace/提供示例代码和独立验收测试；pyproject.toml配置项目入口。凭据只从环境变量读取。
