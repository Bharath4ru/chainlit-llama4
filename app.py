import chainlit as cl
from groq import Groq
import time, base64
client = Groq()
# Utility to encode image
def encode_image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_history", [{"role": "system", "content": "You are a helpful assistant."}])
    msg = cl.Message(content="")
    intro = "Hi! I'm the Meta LLaMA-4 (Scout-17B-16E-Instruct) model, via Groq. Ask me anything!"
    for token in intro:
        await msg.stream_token(token)
        time.sleep(0.005)
    await msg.send()
@cl.on_message
async def on_message(message: cl.Message):
    images = [file for file in message.elements if "image" in file.mime]
    chat_history = cl.user_session.get("chat_history")
    # Prepare user content
    if images:
        image_data = encode_image_to_base64(images[0].path)
        user_content = [
            {"type": "text", "text": message.content},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
    else:
        user_content = message.content

    chat_history.append({"role": "user", "content": user_content})

    # Create streaming message
    msg = cl.Message(content="")

    # Stream response from Groq
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=chat_history,
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=0.8,
        stream=True,
    )

    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            await msg.stream_token(token)

    chat_history.append({"role": "assistant", "content": full_response})
    await msg.send()
