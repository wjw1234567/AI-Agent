# 必须最先设置镜像 + 关闭Windows警告
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 只导入真正需要的！Conversation 已经废弃，彻底删掉
from transformers import pipeline,AutoTokenizer, AutoModelForCausalLM

# ===================== 1. 文本生成 =====================
# generator = pipeline("text-generation", model="gpt2")
# result = generator("Once upon a time", max_length=50, num_return_sequences=1)
# print("="*50)
# print("文本生成:\n", result[0]["generated_text"])
#
# ===================== 2. 对话（无限轮方式） =====================

model_name = "Qwen/Qwen2-1.5B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

print("===== Qwen2 纯中文聊天（quit 退出）=====\n")

while True:
    user_input = input("你：")
    if user_input.lower() == "quit":
        print("AI：再见！")
        break

    messages = [{"role": "user", "content": user_input}]

    # ✅ 修复关键：return_dict=False
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=False  # 必须加！！！
    )

    outputs = model.generate(
        inputs,
        max_new_tokens=300,
        temperature=0.6,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print("AI：", response)
    print("-" * 50)



# ===================== 3. 文本摘要 =====================
# summarizer = pipeline("summarization")
# text = """
# Hugging Face is an AI company that provides tools for natural language processing.
# They offer the Transformers library which is the most popular library for NLP tasks.
# Millions of developers use Hugging Face for text classification, translation,
# summarization, and many other NLP applications.
# """
# summary = summarizer(text, max_length=30, min_length=10)
# print("="*50)
# print("摘要:\n", summary[0]["summary_text"])
# print("="*50)