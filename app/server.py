import base64
import json
import tempfile
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

import whisper
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from googletrans import Translator
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosedError

from Chatbot.app_old import NLUProcessor, get_processor, process_command

warnings.filterwarnings("ignore")

translator = Translator()

# Global variables for models
processor: NLUProcessor | None = None
whisper_model: whisper.Whisper | None = None
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    global processor, whisper_model

    print("🚀 Starting application...")

    # Load Rasa NLU processor
    print("📚 Loading Rasa NLU processor...")
    processor = await get_processor()
    print("✅ Rasa NLU processor loaded!")

    # Load Whisper model
    print("🎤 Loading Whisper model...")
    # Download and cache the model locally
    whisper_model = whisper.load_model("base", download_root="./models")
    print("✅ Whisper model loaded!")

    print("🎉 All models loaded successfully!")
    yield

    # Cleanup
    processor = None
    whisper_model = None
    print("🧹 Models cleaned up")


class Command(BaseModel):
    command: str


class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_chunks = []

    def start_recording(self):
        self.is_recording = True
        self.audio_chunks = []

    def add_audio_chunk(self, chunk):
        if self.is_recording:
            self.audio_chunks.append(chunk)

    async def stop_recording(self):
        self.is_recording = False
        return await self.save_and_transcribe()

    async def save_and_transcribe(self):
        if not self.audio_chunks:
            return None, None

        try:
            # Combine all audio chunks
            combined_audio = b"".join(self.audio_chunks)

            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_file.write(combined_audio)
                temp_audio_path = temp_file.name

            # Transcribe using Whisper
            print("🎤 Transcribing audio...")
            result = whisper_model.transcribe(temp_audio_path)  # type: ignore
            transcription = result["text"].strip()  # type: ignore
            language = result.get("language", "unknown")
            # if language != "en":
            #     # Translate to English if not already
            #     print(f"🌐 Translating transcription from {language} to English...")
            #     print(f"Original transcription: '{transcription}'")
            #     translation = await translator.translate(
            #         transcription, src=language, dest="en"
            #     )
            #     transcription = translation.text
            #     language = "en"

            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)

            print(f"📝 Transcription: '{transcription}' (Language: {language})")
            return transcription, language

        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return None, None


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/sendmoney")
async def send_money_form(request: Request, amount: str = "", receiver: str = ""):
    return templates.TemplateResponse(
        "send_money.html", {"request": request, "amount": amount, "receiver": receiver}
    )


@app.post("/process_command")
async def classify(command: Command):
    print(f"🔍 Processing command: {command.command}")
    return await process_command(command=command.command, processor=processor)


@app.websocket("/ws/voice-nlu")
async def voice_nlu_websocket(websocket: WebSocket):
    await websocket.accept()
    recorder = AudioRecorder()

    print("🔗 WebSocket connected for voice + NLU processing")

    try:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connected",
                    "message": "Voice + NLU processor ready",
                    "models_loaded": {
                        "whisper": whisper_model is not None,
                        "rasa_nlu": processor is not None,
                    },
                }
            )
        )
        print("✅ Initial status sent to client")
    except Exception as e:
        print(f"❌ Error sending initial status: {e}")
        return

    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                print(f"📨 Received message: {message['type']}")

                if message["type"] == "start_recording":
                    recorder.start_recording()
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "recording_started",
                                "message": "🎤 Recording started - speak now!",
                            }
                        )
                    )

                elif message["type"] == "audio_chunk":
                    # Decode base64 audio data
                    audio_chunk = base64.b64decode(message["data"])
                    recorder.add_audio_chunk(audio_chunk)

                elif message["type"] == "stop_recording":
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "processing",
                                "message": "🔄 Processing audio and analyzing intent...",
                            }
                        )
                    )

                    # Transcribe audio
                    transcription, language = await recorder.stop_recording()

                    if transcription:
                        try:
                            # Process with Rasa NLU
                            print(f"🧠 Processing intent for: '{transcription}'")
                            nlu_result = await process_command(
                                command=transcription, processor=processor
                            )

                            # Send complete response
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "processing_complete",
                                        "transcription": {
                                            "text": transcription,
                                            "language": language,
                                            "timestamp": datetime.now().isoformat(),
                                        },
                                        "nlu_result": nlu_result,
                                        "message": "✅ Audio transcribed and intent classified!",
                                    }
                                )
                            )

                            print(f"✅ Complete processing done for: '{transcription}'")

                        except Exception as e:
                            print(f"❌ Error in NLU processing: {e}")
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "processing_complete",
                                        "transcription": {
                                            "text": transcription,
                                            "language": language,
                                            "timestamp": datetime.now().isoformat(),
                                        },
                                        "nlu_result": None,
                                        "error": f"NLU processing failed: {e}",
                                        "message": "⚠️ Audio transcribed but intent classification failed",
                                    }
                                )
                            )

                    else:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": "❌ Failed to transcribe audio",
                                    "error": "No transcription available",
                                }
                            )
                        )

                elif message["type"] == "ping":
                    # Health check
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "pong",
                                "message": "WebSocket connection active",
                                "models_status": {
                                    "whisper": "loaded"
                                    if whisper_model
                                    else "not loaded",
                                    "rasa_nlu": "loaded" if processor else "not loaded",
                                },
                            }
                        )
                    )

            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid message format"})
                )
                continue

            except Exception as e:
                print(f"❌ Error processing message: {e}")
                # Don't send error response here as WebSocket might be closed
                break

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected cleanly")
    except ConnectionClosedError:
        print("🔌 WebSocket connection closed")
    except Exception as e:
        print(f"❌ Unexpected WebSocket error: {e}")
    finally:
        print("🧹 WebSocket connection cleanup completed")


if __name__ == "__main__":
    import uvicorn

    # Create models directory if it doesn't exist
    Path("models").mkdir(parents=True, exist_ok=True)

    print("🚀 Starting Voice + NLU FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
