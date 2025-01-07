from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.shortcuts import render
from openai import OpenAI
import logging

# 配置日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# 初始化大模型 API
api_kwargs = {
    "base_url": "http://172.26.192.23:8001/v1",
    "api_key": "gkcs",
}
client = OpenAI(**api_kwargs)


# 通用模型调用函数
def call_model(client, model, prompt, temperature=0.5, seed=None):
    messages = [
        {"content": "You are a helpful assistant!", "role": "system"},
        {"content": prompt, "role": "user"},
    ]
    generate_cfg = {"temperature": temperature}
    if seed is not None:
        generate_cfg["seed"] = seed
    response = client.chat.completions.create(
        model=model, messages=messages, stream=False, **generate_cfg
    )
    return response.choices[0].message.content


# 定义测试大纲各部分生成函数
# 1.引言
def generate_intro(input_text):
    prompt = f"""
    根据以下需求说明：
    {input_text}

    按照以下模板生成【1.引言】部分的内容：
    ## 1. 引言
    ### 1.1 测试目标
    说明测试大纲的目标、范围及其作用。
    ### 1.2 定义和缩写
    以markdown表格形式列出缩写及其定义。
    ### 1.3 系统概述
    描述系统的主要功能、架构及关键模块。
    ### 1.4 文档概述
    对文档的整体结构进行描述。
    """
    return call_model(client, model="Qwen2-5-Coder-32B-Instruct", prompt=prompt)

# 2.测试准备
def generate_test_preparation(input_text):
    prompt = f"""
    根据以下需求说明：
    {input_text}

    按照以下模板生成【2.测试准备】部分的内容：
    ## 2. 测试准备
    ### 2.1 测试环境
    描述测试所需的硬件、软件和网络环境。
    ### 2.2 硬件准备
    列出测试需要的硬件设备及其配置。
    ### 2.3 软件准备
    列出测试需要的软件工具及其版本。
    ### 2.4 其他测试前准备
    包括测试数据准备、权限配置和环境验证等。
    """
    return call_model(client, model="Qwen2-5-Coder-32B-Instruct", prompt=prompt)

# 3.需求可追踪性
def generate_test_traceability(input_text):
    prompt = f"""
    根据以下需求说明：
    {input_text}

    按照以下模板生成【3.需求可追踪性】部分的内容：
    ## 3. 需求可追踪性
    需求条目 | 测试用例编号 | 覆盖情况 | 备注
    --- | --- | --- | ---
    REQ-001 | TC-001 | 覆盖 | 功能正常
    REQ-002 | TC-002, TC-003 | 部分覆盖 | 需要补充测试用例
    """
    return call_model(client, model="Qwen2-5-Coder-32B-Instruct", prompt=prompt)

# 4.测试说明
def generate_test_description(input_text):
    prompt = f"""
    根据以下需求说明：
    {input_text}

    按照以下模板生成【4.测试说明】部分的内容：
    ## 4. 测试说明
    ### 4.1 功能测试
    测试用例名称 | 前提和约束 | 用例初始化 | 测试步骤 | 终止条件 | 通过准则
    --- | --- | --- | --- | --- | ---
    登录功能测试 | 用户已注册 | 打开登录页面 | 步骤1: 输入用户名和密码<br>步骤2: 点击登录按钮 | 登录成功或失败 | 正确跳转或提示错误
    """
    return call_model(client, model="Qwen2-5-Coder-32B-Instruct", prompt=prompt)

# 测试大纲生成
def generate_test_outline(request):
    input_text = request.POST.get("input_text", "")

    if not input_text:
        logger.error("No input text provided")
        return JsonResponse({"error": "No input text provided"}, status=400)

    logger.info(f"Received input text: {input_text}")
    # 生成各部分内容
    intro = generate_intro(input_text)
    logger.info(f"Generated intro: {intro}")  # 打印引言部分
    preparation = generate_test_preparation(input_text)
    logger.info(f"Generated preparation: {preparation}")  # 打印测试准备部分
    traceability = generate_test_traceability(input_text)
    logger.info(f"Generated traceability: {traceability}")  # 打印需求可追踪性部分
    description = generate_test_description(input_text)
    logger.info(f"Generated description: {description}")  # 打印测试说明部分

    # 合并内容
    outline = f"{intro}\n\n{preparation}\n\n{traceability}\n\n{description}"
    logger.info(f"Final generated outline: {outline}")  # 打印生成的完整大纲

    return JsonResponse({"outline": outline})

# 需求分析生成
# todo：在这里写需求部分代码，暂时先写一个空的函数，仿照前面测试的生成，通用的可以直接调。
# todo：分模块生成，可以参考前面标号1、2、3、4的函数
def generate_requirement(request):
    input_text = request.POST.get("input_text", "")
    if not input_text:
        return JsonResponse({"error": "No input text provided"}, status=400)

    # 生成各部分内容
    # todo 在这里分模块生成

    # todo 合并
    requirement = "这是一个暂时的合并结果"
    return JsonResponse({"requirement": requirement})