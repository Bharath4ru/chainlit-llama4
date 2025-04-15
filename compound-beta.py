import chainlit as cl
from groq import Groq
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# Initialize Groq client
client = Groq()

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_history", [{"role": "system", "content": "You are a helpful assistant with access to search."}])
    msg = cl.Message(content="")
    intro = "Hi! I'm the Groq Compound-Beta model with built-in search!"
    for token in intro:
        await msg.stream_token(token)
        time.sleep(0.005)
    await msg.send()

@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    chat_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

    # Use the compound-beta model with built-in search
    response = client.chat.completions.create(
        model="compound-beta",
        messages=chat_history,
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
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
