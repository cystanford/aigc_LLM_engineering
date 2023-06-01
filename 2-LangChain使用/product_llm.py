import re
from typing import List, Union
# Python内置模块，用于格式化和包装文本
import textwrap
import time

from langchain.agents import (
    Tool,          # 可用工具
    AgentExecutor, # Agent执行
    LLMSingleActionAgent, # 定义Agent
    AgentOutputParser, # 输出结果解析
)
from langchain.prompts import StringPromptTemplate
# LLMChain，包含一个PromptTemplate和一个LLM
from langchain import OpenAI, LLMChain
# Agent执行，Agent结束
from langchain.schema import AgentAction, AgentFinish
# PromptTemplate: 管理LLMs的Prompts
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM

# 定义了LLM的Prompt Template
CONTEXT_QA_TMPL = """
根据以下提供的信息，回答用户的问题
信息：{context}

问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# 输出结果显示，每行最多60字符，每个字符显示停留0.1秒（动态显示效果）
def output_response(response: str) -> None:
    if not response:
        exit(0)
    # 每行最多60个字符
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # Add a delay of 0.1 seconds between each character
            print(" ", end="", flush=True)  # Add a space between each word
        print()  # Move to the next line after each line is printed
    # 遇到这里，这个问题的回答就结束了
    print("----------------------------------------------------------------")

# 模拟公司产品和公司介绍的数据源
class TeslaDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # 工具1：产品描述
    def find_product_description(self, product_name: str) -> str:
        """模拟公司产品的数据库"""
        product_info = {
            "Model 3": "具有简洁、动感的外观设计，流线型车身和现代化前脸。定价23.19-33.19万",
            "Model Y": "在外观上与Model 3相似，但采用了更高的车身和更大的后备箱空间。定价26.39-36.39万",
            "Model X": "拥有独特的翅子门设计和更加大胆的外观风格。定价89.89-105.89万",
        }
        # 基于产品名称 => 产品描述
        return product_info.get(product_name, "没有找到这个产品")

    # 工具2：公司介绍
    def find_company_info(self, query: str) -> str:
        """模拟公司介绍文档数据库，让llm根据信息回答问题"""
        context = """
        特斯拉最知名的产品是电动汽车，其中包括Model S、Model 3、Model X和Model Y等多款车型。
        特斯拉以其技术创新、高性能和领先的自动驾驶技术而闻名。公司不断推动自动驾驶技术的研发，并在车辆中引入了各种驾驶辅助功能，如自动紧急制动、自适应巡航控制和车道保持辅助等。
        """
        # prompt模板 = 上下文context + 用户的query
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        # 使用LLM进行推理
        return self.llm(prompt)


AGENT_TMPL = """按照给定的格式回答以下问题。你可以使用下面这些工具：

{tools}

回答时需要遵循以下用---括起来的格式：

---
Question: 我需要回答的问题
Thought: 回答这个上述我需要做些什么
Action: "{tool_names}" 中的一个工具名
Action Input: 选择这个工具所需要的输入
Observation: 选择这个工具返回的结果
...（这个 思考/行动/行动输入/观察 可以重复N次）
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案
---

现在开始回答，记得在给出最终答案前，需要按照指定格式进行一步一步的推理。

Question: {input}
{agent_scratchpad}
"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str  # 标准模板
    tools: List[Tool]  # 可使用工具集合

    def format(self, **kwargs) -> str:
        """
        按照定义的 template，将需要的值都填写进去。
        Returns:
            str: 填充好后的 template。
        """
        # 取出中间步骤并进行执行
        intermediate_steps = kwargs.pop("intermediate_steps")  
        print('intermediate_steps=', intermediate_steps)
        print('='*30)
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # 记录下当前想法 => 赋值给agent_scratchpad
        kwargs["agent_scratchpad"] = thoughts  
        # 枚举所有可使用的工具名+工具描述
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )  
        # 枚举所有的工具名称
        kwargs["tool_names"] = ", ".join(
            [tool.name for tool in self.tools]
        )  
        cur_prompt = self.template.format(**kwargs)
        #print(cur_prompt)
        return cur_prompt

"""
    对Agent返回结果进行解析，有两种可能：
    1）还在思考中 AgentAction
    2）找到了答案 AgentFinal
"""
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        解析 llm 的输出，根据输出文本找到需要执行的决策。
        Args:
            llm_output (str): _description_
        Raises:
            ValueError: _description_
        Returns:
            Union[AgentAction, AgentFinish]: _description_
        """
        # 如果句子中包含 Final Answer 则代表已经完成
        if "Final Answer:" in llm_output:  
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # 需要进行 AgentAction
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"  # 解析 action_input 和 action
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Agent执行
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


if __name__ == "__main__":
    ## 定义LLM，需要定义环境变量 OPENAI_API_KEY = XXX
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # 自有数据
    tesla_data_source = TeslaDataSource(llm)
    # 定义的Tools
    tools = [
        Tool(
            name="查询产品名称",
            func=tesla_data_source.find_product_description,
            description="通过产品名称找到产品描述时用的工具，输入的是产品名称",
        ),
        Tool(
            name="公司相关信息",
            func=tesla_data_source.find_company_info,
            description="当用户询问公司相关的问题，可以通过这个工具了解公司信息",
        ),
    ]
    
    # 用户定义的模板
    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )
    # Agent返回结果解析
    output_parser = CustomOutputParser()
    # 最常用的Chain, 由LLM + PromptTemplate组成
    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
    # 定义的工具名称
    tool_names = [tool.name for tool in tools]
    # 定义Agent = llm_chain + output_parser + tools_names
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )
    # 定义Agent执行器 = Agent + Tools
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    # 主过程：可以一直提问下去，直到Ctrl+C
    while True:
        try:
            user_input = input("请输入您的问题：")
            response = agent_executor.run(user_input)
            output_response(response)
        except KeyboardInterrupt:
            break