import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import wave

load_dotenv()

async def main():
    client = AsyncOpenAI(
        api_key=os.environ.get("QWEN_API_KEY", "sk-xxx"),
        base_url=os.environ.get("QWEN_BASE_URL")
    )
    
    # create dummy wav
    with wave.open("test.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b'\x00\x00' * 16000)
        
    print("Testing /v1/audio/transcriptions")
    try:
        with open("test.wav", "rb") as f:
            res = await client.audio.transcriptions.create(
                model=os.environ.get("QWEN_MODEL", "qwen3-omni"),
                file=f
            )
        print("Transcription:", res.text)
    except Exception as e:
        print("Failed transcriptions:", e)

    print("Testing /v1/chat/completions with audio base64")
    try:
        import base64
        with open("test.wav", "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        res = await client.chat.completions.create(
            model=os.environ.get("QWEN_MODEL", "qwen3-omni"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe"},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{encoded}"}
                        }
                    ]
                }
            ]
        )
        print("Chat completion:", res.choices[0].message.content)
    except Exception as e:
        print("Failed chat completions:", e)

if __name__ == "__main__":
    asyncio.run(main())
