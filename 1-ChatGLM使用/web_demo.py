from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

"""Override Chatbot.postprocess"""
"""
将消息和回复中的 Markdown 格式文本转换为 HTML 格式。
使用了mdtex2html.convert函数执行转换操作
"""

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess

"""
    函数的主要目的是将文本中的特定格式进行转换，以便在 HTML 环境中显示。
"""
def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

"""
    预测聊天机器人的回复。它接收以下参数：
    input：用户的输入文本。
    chatbot：聊天机器人的对话历史记录，用列表表示，每个元素是一个包含消息和回复的元组。
    max_length：生成的回复的最大长度。
    top_p：top-p（nucleus）采样的概率阈值。
    temperature：用于控制生成文本的多样性的温度参数。
    history：聊天机器人的历史记录。
"""
def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))       

        yield chatbot, history

# 将用户输入的内容，重置为空字符串
def reset_user_input():
    return gr.update(value='')

# 重置聊天机器人的状态
def reset_state():
    return [], []

with gr.Blocks() as demo:
    # 创建一个 ChatGLM 的 HTML 标题，显示在页面的中央
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")
    # 创建一个 chatbot 对象，用于处理聊天机器人的逻辑
    chatbot = gr.Chatbot()
    """
        创建一个界面布局，包括：
        一个文本框用于用户输入。
        一个提交按钮，用于触发聊天机器人的回复生成。
        一个清空历史记录的按钮。
        三个滑动条，用于调整最大回复长度、top-p 参数和温度参数。
    """
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    # 创建history变量，用于存储聊天机器人的历史记录
    history = gr.State([])
    """
      设置提交按钮的点击事件，当点击按钮时，调用 predict 函数来生成聊天机器人的回复
      参数列表包括用户输入、聊天机器人对象、最大回复长度、top-p 参数、温度参数和历史记录
      同时，将聊天机器人对象和历史记录作为输出，以便在页面上显示生成的回复
      还设置了进度条来显示生成回复的进度
    """
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    # 点击提交按钮还会调用 reset_user_input 函数来清空用户输入的文本。
    submitBtn.click(reset_user_input, [], [user_input])
    """
        设置清空历史记录按钮的点击事件
        当点击按钮时，调用 reset_state 函数来重置聊天机器人的状态和历史记录
        同样，通过设置进度条来显示重置状态的进度。
    """
    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

"""
    通过 demo.queue().launch() 将界面布局和相关的交互组件启动，以创建一个可交互的聊天界面
    share=True 参数表示可以共享该界面，inbrowser=True 参数表示在浏览器中打开该界面。
"""
demo.queue().launch(share=True, inbrowser=True)