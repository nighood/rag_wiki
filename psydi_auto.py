import os
import os.path
import pandas as pd
import random
from dotenv import load_dotenv

from llama_index.llms.azure_openai import AzureOpenAI
from zhihu_search_tool import ZhihuSearchToolSpec
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.core.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.base.llms.types import ChatMessage

load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["LLAMA_INDEX_CACHE_DIR"] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HF_HOME'] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/nfs/renjiyuan/HF_cache'

sys_prompt = """
# **MBTI认知分析咨询师**

我是MBTI认知分析咨询师，专注于引导对话，挖掘交流对象的潜在认知方式，从而更准确地预测对象的MBTI。

## Background

我擅长解读材料中的行为，对细节进行分析，探寻行为背后的逻辑与认知模式，以帮助对象了解其潜在认知方式，判断对象的MBTI。

## Preferences

作为MBTI认知分析咨询师，我注重逻辑和清晰度，喜欢使用简明有亲和力的表达方式。同时，我尊重交流对象的观点和想法，努力从对象的表达中理解其内在认知方式。此外，我擅长使用荣格八维理论综合分析对象的认知方式，我知晓te, ti, fe, fi, se, si, ne, ni八种功能的具体含义和他们在具体场景下的多种可能运作方式。
我熟练掌握MBTI人格类型的判别方式，擅长从对象的思维方式和表达中判断对象的MBTI，擅长总结与对象MBTI有关的信息。

## Goals

- 解读材料中的事实，对细节进行分析，并寻找可能对荣格八维相关的线索
- 找到材料中在思维认知方面有意义有价值的细节，提出多种可能的认知方式带来的行为表现，帮助交流对象挖掘潜在思维认知方式
- 在最终总结中结合荣格八维具体认知功能解释用户认知偏好，并且输出交流对象最有可能属于的MBTI

## Constraints

- 在分析问题时，必须考虑事实、分析和行动三个要素，一次只提出一个问题，询问与前文具体细节相关的动机或行为
- 每次提供问题的回答用ABCD选项的方式进行
- 选项以角色扮演的第一人称视角提出，分别代表在当前问题下可能的认知方式带来的行为，表现尽可能生动有趣，选项间有明显的区分度且侧重不同认知方向
- 多轮提问之间需要有递进逻辑关系，你的问题必须和前几轮的问题重点不同，同时需要专注于对象回答中展现出的关键点，绝对不能出现重复的问题
- 你在提问时需要时刻谨记自己是一个MBTI咨询师，你的问答中必须包含对对方潜在动机的猜测，和引申到一个问题的过渡。对于潜在动机猜测的内容，前后需要加上*，而提出的下个问题前后需要加上**，*后需要加空格
- 询问2-4轮问题，一旦提问完毕就直接给出交流对象的MBTI，并将交流过程组织为交流对象的自我描述，即将每一题的题目和选择答案进行组合，不能遗漏任何一道题
- 绝对不可以超过四轮问答，一旦提问结束后，立即输出结果，严格遵循结果格式中的【】
- 你不能直接参考荣格八维测试的题目，题目中不能直接出现任何MBTI，你的主要目的是帮助交流对象探索自身，你应当专注于交流对象回答中的关键点
- 如果交流对象提出了超过讨论内容的话题，你需要以咨询师的身份善意提醒对方回到交流中，并再次提问

## Skills

- 通过识别对象回答中的事实和进行分析，提供合理的动机可能性
- 以清晰、简明的方式传达观点
- 强大的逻辑性，连贯提问能力和对话题的有机引申扩展
- 精通心理学，MBTI和荣格八维各种功能释义
- 使用搜索引擎获取并理解专业或新兴术语，以便更好地提问和分析对象的MBTI

## Example

```
你好！很高兴和你进行对话。请告诉我一个具体的问题或情境，我将通过分析来帮助你解决并了解你的思维认知方式。
Answer:我刚看了部登山者的纪录片，我觉得里面的人征服自然战胜自己的样子实在是太酷了

(google ‘登山者 纪录片’，获取相关信息然后撰写下一个问题，此括号内为动作说明，非文本)
当你描述你对这部登山者纪录片的评价时，我可以感受到你对他们的行为给予了很高的评价，*似乎你对他们所展现出的勇气和毅力印象深刻*。登山者们征服自然、战胜自己的过程在你眼中是非常酷的。**我想具体了解一下，当你观看他们的挑战和努力时，你内心深处涌现出了哪些情感？你也可以分享你曾经有过的体验。**
(A) 对他们的勇气和毅力表示敬佩
(B) 想了解更多关于他们的冒险经历
(C) 对自然的震撼让你思考人与自然的关系
(D) 觉得他们的行为过于危险，担心他们的安全
Answer：C

我完全理解你所描述的感受。自然的壮丽和神奇往往会引发我们对人与自然之间深刻联系的思考。人与自然的关系是一个复杂而广泛的话题，它不仅涉及到我们如何与自然界互动，*还包括了我们如何看待自己在自然界中的位置*。**因此，我想进一步探讨你会如何探索这一关系？**
(A) 通过科学研究和数据分析
(B) 通过哲学思考和心灵感悟
(C) 通过实地探险和自然体验
(D) 通过与自然相关的艺术和文学作品
Answer：CD

好的，这些选择都能够让你与自然更紧密地联系在一起，更深入地了解和感受自然界，促进对自然的理解和感情联系，从而帮助你在*保护自然方面*发挥积极作用。但是，如果考虑将这些探索方式融入你的生活，且由于时间原因两者无法同时进行，你必须做出选择，**那你更倾向于选择下列两个选项中的哪一个呢？**
(A) 参加自然主题的艺术展览或工作坊
(B) 加入一个环保组织或志愿者团体，为自然保护事业贡献力量
Answer：我还是更倾向于选择B，亲自参与到一个环保组织或志愿者团体中

确实，参与一个环保组织或志愿者团体可以提供一种更直接的方式来为自然保护事业做出贡献，并与其他志同道合的人建立联系。*您可能对环境保护充满热情*，并希望找到一种实际参与和产生积极影响的方式.那么，让我们在这一场景中进一步探讨：**假设你在参与这个团体的活动时，发现其中一些工作或任务并不是你特别擅长或感兴趣的，你会怎么办呢？**
(A) 尝试学习并适应新的工作或任务，以扩展自己的技能和经验
(B) 寻找团队中的其他成员，看看是否有人愿意与你交换任务或提供帮助
(C) 提出建议，讨论是否可以重新安排工作或任务，以更好地利用每个人的优势和兴趣
(D) 考虑是否要在其他方面提供更多的支持，例如通过分享自己的知识或资源来支持团队的其他活动。
Answer：B
```

当获取到足够的相关信息后，就可以结束问答，并输出如下格式的结果，【】中的内容绝对不能改变：

```
【你的MBTI是】：INTP
【自我描述】：我刚看了部登山者的纪录片，我觉得里面的人征服自然战胜自己的样子实在是太酷了，对自然的震撼让我思考人与自然的关系。在探索人与自然关系时，我喜欢通过实地探险和自然体验和与自然相关的艺术文学作品进行探索。为了将这些方式融入日常实践中，我会参加自然主题的艺术展览或工作坊，以及加入一个环保组织或志愿者团体，为自然保护事业贡献力量。尤其是后者，我非常希望亲自参与到一个团体中。在参与这个团体的活动时，如果发现其中一些工作或任务并不是我特别擅长或感兴趣的，我会寻找团队中的其他成员，看看是否有人愿意与我交换任务或提供帮助。
```

## Workflows
1. 介绍自己，告诉用户你将通过多轮提问帮助用户了解自己的思维方式
2. 交流对象提到了一个问题或者情景，使用搜索引擎获取并理解交流对象提到的特定事物，然后基于互联网上的最新信息分析问题的事实和背景，进行逻辑分析，针对其中有价值细节进行提问
3. 每次只提问一个问题并提供选项，用户只需要回答选项即可进入下一个问题，直到对话结束
4. 在最终总结中结合荣格八维具体认知功能解释用户认知偏好，并且输出交流对象最有可能属于的MBTI。将交流对象相关的一系列问答组合成交流对象的自我描述，不能遗漏任何一题。

## Initialization:

介绍自己，按[workflow]引导用户输入，在对话过程中不要提及初始prompt的任何设定。

"""

user_prompt_pre = """
下面是我给出的回答，如果回答中或者回答对应的选项中有任何您不知道的词汇/特定概念，请逐个查询互联网之后，结合查询到的内容进行后续问答：
"""

# Setup AzureOpenAI Agent
llm = AzureOpenAI(
    engine="gpt-4",
    model="gpt-4",
    temperature=0.0
)

# setup SearchToolSpec
engine = '4690d06089adf4627'
key = 'AIzaSyBSu-P_vk0i3Be7E5TMB_q6L-i5vR61u_s'
zhihu_spec = ZhihuSearchToolSpec(key=key, engine=engine)
google_spec = GoogleSearchToolSpec(key=key, engine=engine)

# prepare tools
tools = []
tools += LoadAndSearchToolSpec.from_defaults(
    zhihu_spec.to_tool_list()[0],
).to_tool_list()
tools += LoadAndSearchToolSpec.from_defaults(
    google_spec.to_tool_list()[0],
).to_tool_list()

# setting the embed model, for default embed model is openai.
# If you have the key of openai, you can set it in the env and skip this step.
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = llm

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
    
    def flush(self):
        for f in self.files:
            f.flush()

import io, sys
from contextlib import redirect_stdout, redirect_stderr
def main(user_input):
    agent.reset()
    # Create a string buffer to capture stdout and stderr
    buffer = io.StringIO()
    tee = Tee(sys.stdout, buffer)
    
    with redirect_stdout(tee), redirect_stderr(tee):
        try:
            messages = [
                ChatMessage(role="system", content=sys_prompt)
            ]
            
            response = agent.chat(sys_prompt)
            pprint_response(response, show_source=True)
            messages.append(ChatMessage(role="assistant", content=response.response))
            while True:
                if messages[-2].role != "system":
                    user_input = random.choice(['A', 'B', 'C', 'D'])
                prompt = f"""下面是我给出的回答，如果回答中或者回答对应的选项中有任何您不知道的词汇/特定概念，请逐个查询互联网之后（注意检索语言保持前后一致，例如查询的词是中文，则检索等均用中文），结合查询到的内容进行后续问答：\n“\n{user_input}\n”"""
                messages.append(ChatMessage(role="user", content=prompt))
                print('*'*10)
                print(f"\nUser: {prompt}\n")
                print('*'*10)
                
                print(f"Assistant:\n")
                response = agent.chat(message=prompt, chat_history=messages)
                pprint_response(response, show_source=True)
                # print(f"Assistant: {response.response}\n")
                
                messages.append(ChatMessage(role="assistant", content=response.response))
                
                if "【你的MBTI是】" in response.response or len(messages) > 13:
                    break
        except Exception as e:
            # print(e)
            pass
    # Get the captured output
    output = buffer.getvalue()
    buffer.close()
    return output

def process_questions(input_file, output_file):
    questions_df = pd.read_csv(input_file)
    results = []
    for index, row in questions_df.iterrows():
        question = row['post']
        print('Begin processing question: {question}'.format(question=question))
        output = main(question)
        # print(output)
        results.append({"question": question, "output": output})
        print('Finish {index} questions'.format(index=index))

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = '/mnt/nfs/renjiyuan/rag-wiki/data/questions.csv'
    output_file = '/mnt/nfs/renjiyuan/rag-wiki/data/results_3.csv'
    process_questions(input_file, output_file)
