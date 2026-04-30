
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage,AssistantMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model = "deepseek/DeepSeek-V3-0324"
# model="openai/gpt-4.1"
token = os.environ["GITHUB_TOKEN"]
# print(token)

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)


messages = [
    SystemMessage("你是一个资深AI agent开发工程师，专业、简洁、有用地回答相关技术问题")
]

# 🔴 新增：总 token 统计
total_tokens = 0

print("==== DeepSeek 多轮对话已启动！输入 exit 退出 ====")

while True:
    # 获取用户输入
    user_input = input("你：")

    # 输入 exit 就退出
    if user_input.lower() == "exit":
        print("对话结束")
        break

    # 把用户说的加入历史
    messages.append(UserMessage(user_input))

    # 调用模型
    response = client.complete(
        messages=messages,
        temperature=1,
        max_tokens=1500,
        model=model
    )

    # AI 回答
    ai_response = response.choices[0].message.content
    print("AI：", ai_response)


    # ===========================
    # 🔴 新增：Token 消耗计算
    # ===========================
    usage = response.usage
    prompt_tokens = usage.prompt_tokens        # 提问消耗
    completion_tokens = usage.completion_tokens  # 回答消耗
    current_total = usage.total_tokens        # 本轮总消耗
    total_tokens += current_total             # 累计总消耗

    # 打印消耗
    print(f"\n📊 Token 消耗：")
    print(f"├─ 提问：{prompt_tokens}")
    print(f"├─ 回答：{completion_tokens}")
    print(f"├─ 本轮总：{current_total}")
    print(f"└─ 累计总：{total_tokens}\n")

    # 把 AI 回答也加入历史 → 下一轮能记住上下文！
    messages.append(AssistantMessage(ai_response))


