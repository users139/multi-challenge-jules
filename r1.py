from openai import OpenAI, AuthenticationError  # OpenAI库用于API调用, AuthenticationError 用于捕获认证错误
import httpx  # 用于创建HTTP客户端
import os
import time
import traceback  # 用于在测试部分打印完整的错误堆栈


# 警告: 以下代码中的 get_openai_api_response_cot 函数移除了所有 try...except 错误处理。
# 这意味着任何在API调用或客户端设置中发生的错误都将导致该函数直接抛出异常并可能终止程序。
# 这种做法通常不推荐用于生产环境或需要稳定运行的脚本。

def get_openai_api_response_cot(msg, max_retries=3):
    OPENROUTER_API_KEY = "sk-or1"  # <--- 在这里替换您的API Key

    # 初始化 OpenAI 客户端。如果此处发生错误，程序将崩溃。
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(
            verify=False,  # 根据您的要求设置 SSL 验证为 False
        )
    )

    # 模型名称保持为 "deepseek-r1"。请确保此模型ID在OpenRouter上有效。
    model_id = "google/gemini-2.5-flash-preview-05-20"

    retry = 0

    while retry < max_retries:
        # API 调用。如果此调用失败（例如网络错误、API密钥无效、模型未找到、服务器错误等），
        # 程序将因未捕获的异常而终止。
        completion = client.chat.completions.create(
            model=model_id,
            messages=msg,
            temperature=0.95,
            max_tokens=1024,
            top_p=0.7,
        )

        res = ""  # 为本次尝试初始化res
        cot = ""  # 模型思考部分始终返回空字符串

        if completion.choices and len(completion.choices) > 0:
            message_object = completion.choices[0].message
            res = message_object.content.strip() if message_object.content else ""
            # cot 已被设为 "", 不再从响应中提取

            # 检查响应长度是否符合要求
            if len(res) < 3:
                print(f"响应过短 (长度 {len(res)}): '{res}'. 第 {retry + 1}/{max_retries} 次尝试后重试...", flush=True)
                retry += 1
                if retry < max_retries:
                    time.sleep(1)
                continue

            return res, cot  # cot 固定为空字符串 ""
        else:
            # API调用技术上成功，但响应中没有 'choices' 或 'choices' 列表为空
            print(f"API响应中未找到 'choices' 或 'choices' 为空. 第 {retry + 1}/{max_retries} 次尝试后重试...",
                  flush=True)
            retry += 1
            if retry < max_retries:
                time.sleep(1)
            continue

    # 如果循环因达到 max_retries 而结束
    print(f"已达到最大重试次数 ({max_retries})，且未能获得符合要求的响应。", flush=True)
    print(f"最后一次尝试发送的消息: {msg}", flush=True)
    raise Exception("超出最大的评测次数！ (因多次尝试后响应仍不符合要求)")


# --- 测试部分 ---
if __name__ == "__main__":
    print("=" * 50)
    print("正在测试 API 调用 (get_openai_api_response_cot)...")
    print("=" * 50)

    # 提示: 请确保您已在上面的 OPENROUTER_API_KEY 变量中设置了有效的 API 密钥。
    # 如果 API 密钥不正确或无效，测试将失败并提示认证错误。

    # 用于测试的示例消息 (OpenAI格式)
    sample_test_messages = [
        {"role": "user", "content": "你好！请用中文简单介绍一下你自己，说明你是什么模型。"}
    ]

    print(f"\n[测试] 发送消息到 API: {sample_test_messages}")

    # 对测试调用使用 try-except 以便优雅地处理错误并提供反馈
    # 注意：get_openai_api_response_cot 函数本身没有内部的 try-except
    try:
        # 为了快速测试，可以将 max_retries 设置为较小的值，例如 1
        model_reply, thinking_part = get_openai_api_response_cot(sample_test_messages, max_retries=1)

        print("\n[测试结果] API 调用成功!")
        print("-" * 30)
        print(f"模型回复 (res):")
        print(model_reply)
        print("-" * 30)
        print(f"模型思考部分 (cot): '{thinking_part}' (此部分应始终为空字符串)")

        if thinking_part == "":
            print("\n[测试验证] 'cot' 部分为空字符串，符合预期。")
        else:
            print(f"\n[测试警告] 'cot' 部分为 '{thinking_part}'，未按预期返回空字符串。")

    except AuthenticationError as auth_err:
        print("\n[测试结果] API 调用失败: 认证错误")
        print("-" * 30)
        print(f"错误详情: {auth_err}")
        print("\n请务必检查以下几点：")
        print("1. `OPENROUTER_API_KEY` 是否已在 `r1.py` 文件顶部正确设置？")
        print("2. API 密钥是否有效且具有访问目标模型的权限？")
        print("3. OpenRouter 服务是否正常？")

    except Exception as e:
        print("\n[测试结果] API 调用失败，发生未预料的错误")
        print("-" * 30)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {e}")
        print("\n完整错误追溯信息:")
        traceback.print_exc()  # 打印完整的Python错误堆栈信息

    print("\n" + "=" * 50)
    print("测试结束。")
    print("=" * 50)
