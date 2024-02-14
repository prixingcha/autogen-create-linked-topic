import autogen
import chromadb
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import gradio as gr


def autogen_chat(pdf_path, query):
    if not pdf_path:
        return "pdf path is required"
    config_list = [
         {
             "model": "codellama:7b-instruct",
             "base_url": "http://localhost:11434/v1",
             "api_key": "ollama",
             "seed": None
         }
     ]

    llm_config_proxy = {
        "temperature": 0,
        "config_list": config_list,
    }

    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config_proxy,
        system_message="""You are a helpful assistant. Provide accurate answers based on the context. Respond "Unsure about answer" if uncertain.""",
    )

    user = RetrieveUserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        system_message="Assistant who has extra content retrieval power for solving difficult problems.",
        max_consecutive_auto_reply=10,
        retrieve_config={
            "task": "code",
            "docs_path": ['autogen.pdf'],
            "chunk_token_size": 1000,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path='/tmp/chromadb'),
            "collection_name": "pdfreader",
            "get_or_create": True,
        },
        code_execution_config={"work_dir": "coding"},
    )

    user_question = """
    Compose a short LinkedIn post showcasing how AutoGen is revolutionizing the future of Generative AI 
    through the collaboration of various agents. Craft an introduction, main body, and a compelling 
    conclusion. Encourage readers to share the post. Keep the post under 500 words.
    """

    user.initiate_chat(
        assistant,
        problem=user_question,
    )
    
    messages = user.chat_messages[assistant]
    last_message = messages[-1]["content"]
    return last_message

#create Gradio interface
iface = gr.Interface(
    fn = autogen_chat,
    inputs=[
        gr.Textbox(label="path to pdf", placeholder = "enter path"),
        gr.Textbox(label="Topic", placeholder = "enter topic")
    ],
    outputs = gr.Textbox(label="Assistant's response "),
    title = "autogen assistant chat",
    description="enter a pdf path to get an answer from the ato gen assistant."
)

iface.launch()