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
    #础石大模型
    #"base_url": "http://172.26.192.23:8001/v1",
    # 础石api key
    # "api_key": "gkcs",

    "base_url": "https://api.lingyiwanwu.com/v1",
    "api_key": "44a7df12b4594691bfd886a30fc55870",
}
client = OpenAI(**api_kwargs)


# 通用模型调用函数
def call_model(client, model, prompt, temperature=0.3, seed=None):
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
    根据以下需求说明详细描述【1. 引言】部分。请依据需求背景、系统概述等内容生成符合实际应用场景的引言内容。确保以下信息得到明确阐述：
    1. 测试目标：测试大纲的目标、范围及其作用；
    2. 定义和缩写：以markdown表格形式列出所有相关缩写及其定义；
    3. 系统概述：详细描述系统的主要功能、架构及关键模块；
    4. 文档概述：简要说明文档的整体结构。
    需求说明：{input_text}
    """
    # return call_model(client, model="Qwen2-5-Coder-32B-Instruct", prompt=prompt)
    #return call_model(client, model="yi-lightning", prompt=prompt)
    return call_model(client, model="yi-lightning", prompt=prompt)

# 2.测试准备
def generate_test_preparation(input_text):
    prompt = f"""
    根据以下需求说明:{input_text}，生成【2. 测试准备】部分的详细内容。请按照以下子章节的要求生成每一部分，并确保描述与需求说明中涉及的准备工作相关：

    ## 2. 测试准备
    ### 2.1 测试环境
    描述测试所需的硬件、软件和网络环境。请特别关注以下内容：
    - 测试需要的操作系统版本（如 Windows 10、Ubuntu 20.04）
    - 网络配置要求（如是否需要内网环境或公网访问等）
    - 特殊硬件要求（如服务器、测试设备等）

    ### 2.2 硬件准备
    列出测试需要的硬件设备及其具体配置，包括：
    - CPU、内存、硬盘、显卡等硬件规格
    - 测试环境所需的设备（如传感器、接口卡等）

    ### 2.3 软件准备
    列出测试需要的软件工具及其版本。请包括以下内容：
    - 操作系统版本及配置要求
    - 所需测试工具（如 Selenium、JMeter）
    - 相关库或依赖项（如 Python 版本、驱动程序等）

    ### 2.4 其他测试前准备
    包括以下内容：
    - 测试数据准备（是否需要模拟数据，数据格式要求等）
    - 权限配置（是否需要特定的权限或角色来运行测试）
    - 环境验证（如测试环境是否部署完毕，服务是否启动等）
    """
    #return call_model(client, model="Qwen2-5-Coder-32B-Instruct", prompt=prompt)
    return call_model(client, model="yi-lightning", prompt=prompt)

# 3.需求可追踪性
def generate_test_traceability(input_text):
    prompt = f"""
    根据以下需求说明:
    {input_text}，
    生成【3. 需求可追踪性】部分的详细内容。请依据需求文档中的条目，生成每个需求的测试用例，并确保测试覆盖情况清晰，并提供必要的备注。模板如下：

    ## 3. 需求可追踪性
    
     格式：
    - 需求条目：对应的需求名目
    - 测试用例：每个需求所对应的测试用例
    - 覆盖情况：描述需求的测试覆盖情况，具体说明哪些测试用例覆盖了该需求，部分覆盖也要列出未覆盖的部分
    - 备注：对于每个需求条目，提供补充说明，说明是否需要添加测试用例或修改需求

     示例格式：
    需求条目 | 测试用例 | 覆盖情况 | 备注
    --- | --- | --- | ---
    检测分析 | 新建检测任务 | 完全覆盖 | 功能正常
    检测分析 | 删除检测任务 | 部分覆盖 | 需要补充测试用例以验证性能
    """
    # return call_model(client, model="Qwen2-5-Coder-32B-Instruct", prompt=prompt)
    return call_model(client, model="yi-vision-v2", prompt=prompt)

# 4.测试说明
def generate_test_description(input_text ,traceability_output):
    prompt = f"""
        根据以下需求说明和第三部分输出生成【4. 测试说明】部分。生成的内容应遵循以下模板，保持结构清晰：

        ## 4. 测试说明
        ### 4.1 功能测试
        每个测试用例需要包括以下字段，并以表格形式输出：
        - 测试用例名称，应与第三部分的输出中的测试用例名称逐条相同
        - 前提和约束：明确每个测试用例需要满足的前提条件，例如用户登录、特定数据存在等。
        - 用例初始化：测试用例开始前的必要初始化步骤，例如数据准备、环境配置等。
        - 测试步骤：详细描述每一个步骤，尽量细化操作，例如“点击按钮”、“输入文本”等。
        - 终止条件：测试用例的结束条件，例如“任务完成”或“结果验证通过”。
        - 通过准则：测试成功的标准，例如“结果与预期一致”或“没有错误提示”。

        例如：
        | 测试用例名称 | 前提和约束 | 用例初始化 | 测试步骤 | 终止条件 | 通过准则 |
        | --- | --- | --- | --- | --- | --- |
        | 测试用例：新建检测任务 | 用户已登录，系统中存在可用的代码仓库和分支 | 进入代码检测-检测任务页面 | 1. 点击“新建检测任务”按钮<br>2. 填写任务名称、分支等信息<br>3. 点击保存 | 任务创建成功，页面展示新任务 | 任务卡片显示任务名称，且任务信息准确 |

        请根据第三部分输出中的测试用例名称逐条生成详细的测试步骤，确保每个步骤描述清晰、详细。

        ### 4.2 非功能测试
        #### 4.2.1 性能测试
        1. 测试内容：任务执行性能测试
        2. 测试步骤：使用负载工具模拟高并发请求，记录响应时间和系统资源使用情况
        3. 期望结果：响应时间在规定范围内，系统资源使用合理

        #### 4.2.2 可靠性测试
        1. 测试内容：系统可用性测试
        2. 测试步骤：模拟不同负载下系统运行情况，测试其是否能够长时间稳定运行
        3. 期望结果：系统应稳定运行，不出现崩溃或错误

        #### 4.2.3 可移植性测试
        1. 测试内容：跨平台支持测试
        2. 测试步骤：在不同操作系统和硬件环境下测试系统功能
        3. 期望结果：系统在不同平台上均能稳定运行，且功能正常

        需求说明：{input_text}
        第三部分输出：{traceability_output}
        """
    # return call_model(client, model="Qwen2-5-Coder-32B-Instruct", prompt=prompt)
    return call_model(client, model="yi-vision-v2", prompt=prompt)

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
    description = generate_test_description(input_text, traceability)
    logger.info(f"Generated description: {description}")  # 打印测试说明部分

    # 合并内容
    outline = f"{intro}\n\n{preparation}\n\n{traceability}\n\n{description}"
    logger.info(f"Final generated outline: {outline}")  # 打印生成的完整大纲

    return JsonResponse({"outline": outline})

