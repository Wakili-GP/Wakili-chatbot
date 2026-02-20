import gradio as gr

from app.deps import get_chain

def chat_fn(message, history):
    """
    history comes as: [(user_msg, bot_msg), ...]
    We'll convert it to your chain's expected format:
    [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]
    """
    history_dicts = []
    for u, a in history:
        if u:
            history_dicts.append({"role": "user", "content": u})
        if a:
            history_dicts.append({"role": "assistant", "content": a})

    chain = get_chain(conversation_history=history_dicts)

    # Your chain.invoke is sync
    result = chain.invoke(message)
    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    return answer

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Legal RAG Chatbot",
    description="Ask questions and get answers with retrieved context."
)

if __name__ == "__main__":
    demo.launch()