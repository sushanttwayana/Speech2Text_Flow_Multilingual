import asyncio
import warnings
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import json
import base64
import logging
from datetime import datetime
import os
from contextlib import asynccontextmanager
from rasa.core.agent import Agent

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Global variables for Whisper
MODEL_LOADED = False
PIPE = None  # ASR pipeline
DEVICE = None
SAMPLE_RATE = 16000  # Whisper model sample rate
client_languages: Dict[str, str] = {}

# Global variable for Rasa
NLU_PROCESSOR = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED, PIPE, DEVICE, NLU_PROCESSOR
    try:
        # Load Whisper model
        logger.info("Loading speech recognition model...")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {DEVICE}")

        local_dir = "../whisper_models/whisper-large-v3-turbo/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
        torch_dtype = torch.float16 if DEVICE.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            local_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(local_dir)

        PIPE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=1,
            torch_dtype=torch_dtype,
            device=DEVICE,
        )
        MODEL_LOADED = True
        logger.info("Whisper model loaded and pipeline ready.")

        # Load Rasa model
        logger.info("Loading Rasa NLU model...")
        NLU_MODEL_PATH = "./models/ `   "
        NLU_PROCESSOR = NLUProcessor.create(NLU_MODEL_PATH)
        if NLU_PROCESSOR:
            logger.info("✅ Rasa NLUProcessor initialized successfully.")
        else:
            logger.error("Failed to initialize Rasa NLUProcessor.")

        yield

    except Exception as e:
        logger.exception("Failed to load models:")
        MODEL_LOADED = False
        NLU_PROCESSOR = None
        yield
    finally:
        logger.info("Shutting down...")

app = FastAPI(title="Speech-to-Text and Intent Classification API", version="3.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# --- RASA NLU PROCESSOR CLASS ---
class NLUProcessor:
    """
    A class to load a Rasa model and use it for NLU-only tasks.
    """
    def __init__(self, agent: Agent):
        self.agent = agent
        if self.agent:
            logger.info("✅ Rasa NLUProcessor initialized successfully.")

    @classmethod
    def create(cls, model_path: str) -> Optional["NLUProcessor"]:
        """
        Loads the Rasa model and returns a class instance.

        Args:
            model_path: Path to the trained Rasa model (.tar.gz).

        Returns:
            An instance of NLUProcessor, or None if loading fails.
        """
        try:
            agent = Agent.load(model_path=model_path)
            return cls(agent)
        except Exception as e:
            logger.error(f"❌ Error loading Rasa model: {e}")
            return None

    async def classify_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses the loaded agent to classify the intent of the given text.

        Args:
            text: The user input text.

        Returns:
            The classification result dictionary from Rasa, or None.
        """
        if not self.agent:
            logger.error("Agent not available. Cannot classify intent.")
            return None

        result = await self.agent.parse_message(text)
        return result or None

    async def process_command(self, text: str, confidence_threshold: float = 0.80):
        """
        Processes a text command, classifies it, and prepares form data.

        Args:
            text: The user input text.
            confidence_threshold: The minimum confidence to consider an intent valid.

        Returns:
            Dictionary with action, message, and form data (if applicable).
        """
        classification_result = await self.classify_intent(text)

        if classification_result:
            intent = classification_result.get("intent", {})
            intent_name = intent.get("name")
            confidence = intent.get("confidence")

            logger.info(f"--- Rasa NLU Analysis --- Intent: {intent_name}, Confidence: {confidence:.2f}")

            # Extract entities for form filling
            entities = classification_result.get("entities", [])
            form_data = {}
            for entity in entities:
                if entity["entity"] == "amount":
                    form_data["amount"] = entity["value"]
                elif entity["entity"] == "person":
                    form_data["name"] = entity["value"]

            if confidence and confidence > confidence_threshold:
                if intent_name == "top_up":
                    logger.info("Action: Initiating 'mobile top-up' flow...")
                    return {
                        "action": "top_up",
                        "message": "Initiating 'mobile top-up' flow...",
                        "form_data": form_data
                    }
                elif intent_name == "topup_wallet":
                    logger.info("Action: Initiating 'wallet top-up' flow...")
                    return {
                        "action": "topup_wallet",
                        "message": "Initiating 'wallet top-up' flow...",
                        "form_data": form_data
                    }
                elif intent_name == "send_money":
                    logger.info("Action: Initiating 'send money' flow...")
                    return {
                        "action": "send_money",
                        "message": "Initiating 'send money' flow...",
                        "form_data": form_data,
                        "entities": entities
                    }
                elif intent_name == "card_to_card_transfer":
                    logger.info("Action:  Initiating 'card to card transfer' flow...")
                    return {
                        "action": "card_to_card_transfer",
                        "message": "Initiating 'card to card transfer' flow...",
                        "form_data": form_data
                    }
                else:
                    logger.info(f"Action:  Intent recognized, but no specific action defined: {intent_name}")
                    return {
                        "action": "unknown",
                        "message": "Intent recognized, but no specific action defined.",
                        "form_data": form_data
                    }
            else:
                logger.info(f"Action:  Confidence ({confidence:.2f}) below threshold ({confidence_threshold}).")
                return {
                    "action": "unknown",
                    "message": "Could not determine action. Confidence is below threshold.",
                    "form_data": form_data
                }
        else:
            logger.error("No classification result received.")
            return {
                "action": "unknown",
                "message": "No classification result received.",
                "form_data": {}
            }
# --- CONNECTION MANAGER FOR WEBSOCKET ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, List[np.ndarray]] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.is_recording: Dict[str, bool] = {}
        self.transcribed_segments: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id in self.active_connections:
            logger.warning(f"Client {client_id} reconnected, closing old connection.")
            await self.active_connections[client_id].close()

        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.is_recording[client_id] = False
        self.transcribed_segments[client_id] = []
        logger.info(f"Client {client_id} connected.")
        await self.send_personal_message(json.dumps({"type": "status", "message": "Connected"}), client_id)

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.processing_tasks and not self.processing_tasks[client_id].done():
            self.processing_tasks[client_id].cancel()
            del self.processing_tasks[client_id]
        if client_id in self.is_recording:
            del self.is_recording[client_id]
        if client_id in self.transcribed_segments:
            del self.transcribed_segments[client_id]
        if client_id in client_languages:
            del client_languages[client_id]
        logger.info(f"Client {client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
                logger.info(f"Sent message to {client_id}: {message}")
            except Exception as e:
                logger.error(f"Failed sending to {client_id}: {e} — disconnecting.")
                self.disconnect(client_id)

manager = ConnectionManager()

async def transcribe_segment(audio_np_segment: np.ndarray, language: str) -> str:
    if not MODEL_LOADED:
        logger.error("Transcription requested but model not loaded.")
        raise RuntimeError("Model not loaded.")
    try:
        result = PIPE(audio_np_segment, generate_kwargs={"language": language})
        return result["text"].strip()
    except Exception:
        logger.exception("Transcription error:")
        return ""

async def transcribe_full_audio(client_id: str, full_audio: np.ndarray, language: str) -> str:
    """Final pass over full audio with max 30s chunks."""
    MAX_CHUNK = SAMPLE_RATE * 30
    segments = []
    start = 0
    while start < len(full_audio):
        segment = full_audio[start : start + MAX_CHUNK]
        text = await transcribe_segment(segment, language)
        if text:
            segments.append(text)
        start += MAX_CHUNK
    return ' '.join(segments).strip()

async def process_audio_for_client(client_id: str, language: str):
    CHUNK_SECONDS = 15
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
    audio_buffer = np.array([], dtype=np.float32)

    while manager.is_recording.get(client_id, False):
        try:
            await asyncio.sleep(0.2)
            while manager.audio_buffers[client_id]:
                audio_buffer = np.concatenate([audio_buffer, manager.audio_buffers[client_id].pop(0)])
            while len(audio_buffer) >= CHUNK_SAMPLES:
                chunk = audio_buffer[:CHUNK_SAMPLES]
                audio_buffer = audio_buffer[CHUNK_SAMPLES:]
                transcription = await transcribe_segment(chunk, language)
                if transcription:
                    await manager.send_personal_message(
                        json.dumps({"type": "transcription", "text": transcription, "is_final": False}),
                        client_id
                    )
                    manager.transcribed_segments[client_id].append(transcription)
        except asyncio.CancelledError:
            logger.info(f"Interim task for {client_id} cancelled.")
            break
        except Exception:
            logger.exception(f"Interim processing error for {client_id}")
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": "Interim processing error"}),
                client_id
            )
            break

    if len(audio_buffer) > 0:
        try:
            transcription = await transcribe_segment(audio_buffer, language)
            if transcription:
                await manager.send_personal_message(
                    json.dumps({"type": "transcription", "text": transcription, "is_final": False}),
                    client_id
                )
                manager.transcribed_segments[client_id].append(transcription)
        except Exception:
            logger.exception(f"Error processing final chunk for {client_id}")
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": "Final chunk processing error"}),
                client_id
            )

def merge_transcriptions(segments: List[str]) -> str:
    return ' '.join(segments).strip() if segments else ""

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    interim_task: asyncio.Task = None

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["type"] == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    client_id
                )
                continue

            if data["type"] == "start_recording":
                lang = data.get("language", "en")
                supported = ["en", "ms", "my", "zh", "id", "ne", "ta", "bn"]
                if lang not in supported:
                    logger.warning(f"Unsupported language {lang} from {client_id}, defaulting to en")
                    lang = "en"

                client_languages[client_id] = lang
                logger.info(f"[{client_id}] Started recording; language={lang}")

                manager.audio_buffers[client_id] = []
                manager.is_recording[client_id] = True

                if interim_task and not interim_task.done():
                    interim_task.cancel()
                    try:
                        await interim_task
                    except asyncio.CancelledError:
                        pass

                interim_task = asyncio.create_task(process_audio_for_client(client_id, lang))

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording started.", "language": lang}),
                    client_id
                )
                continue

            if data["type"] == "stop_recording":
                manager.is_recording[client_id] = False

                if interim_task and not interim_task.done():
                    interim_task.cancel()
                    try:
                        await interim_task
                    except asyncio.CancelledError:
                        pass
                    interim_task = None

                # Process any remaining audio in buffer
                final_audio = np.array([], dtype=np.float32)
                if manager.audio_buffers[client_id]:
                    final_audio = np.concatenate(manager.audio_buffers[client_id])
                    logger.info(f"Processing remaining audio buffer for {client_id}, size: {len(final_audio)} samples")
                    manager.audio_buffers[client_id] = []

                final_segments = manager.transcribed_segments.get(client_id, [])
                if final_audio.size > 0:
                    transcription = await transcribe_segment(final_audio, client_languages.get(client_id, "en"))
                    if transcription:
                        final_segments.append(transcription)
                        logger.info(f"Appended final transcription for {client_id}: {transcription}")

                final_text = merge_transcriptions(final_segments) or "No speech detected."
                logger.info(f"Final transcription for {client_id}: {final_text}")

                # Initialize result to avoid UnboundLocalError
                result = {"action": "error", "message": "No transcription or Rasa processor not initialized.", "form_data": {}}

                # Process with Rasa
                if NLU_PROCESSOR and final_text != "No speech detected.":
                    try:
                        result = await NLU_PROCESSOR.process_command(final_text)
                        logger.info(f"Rasa result for {client_id}: {result}")
                    except Exception as e:
                        logger.exception(f"Rasa processing error for {client_id}: {e}")
                        result = {"action": "error", "message": f"Rasa processing failed: {str(e)}", "form_data": {}}

                await manager.send_personal_message(
                    json.dumps({
                        "type": "transcription",
                        "text": final_text,
                        "is_final": True,
                        "intent": result
                    }),
                    client_id
                )

                # Navigate and prefill topup page for topup_wallet intent
                if result.get("action") == "topup_wallet" and result.get("form_data", {}).get("amount"):
                    amount = result["form_data"]["amount"]
                    name = result["form_data"].get("name", "")
                    navigate_message = {
                        "type": "navigate",
                        "route": "/topup.html",  # Changed from /static/topup.html
                        "prefill": {
                            "amount": amount,
                            # "name": name
                        }
                    }
                    logger.info(f"Sending navigate message to {client_id}: {navigate_message}")
                    await manager.send_personal_message(
                        json.dumps(navigate_message),
                        client_id
                    )

                manager.audio_buffers[client_id] = []
                manager.transcribed_segments[client_id] = []

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording stopped."}),
                    client_id
                )
                continue

            if data["type"] == "audio_chunk":
                if not manager.is_recording.get(client_id, False):
                    logger.info(f"Received audio_chunk for {client_id}, but not recording, ignoring.")
                    continue

                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    logger.info(f"Received audio chunk for {client_id}, size: {len(audio_np)} samples")
                    manager.audio_buffers[client_id].append(audio_np)
                except Exception:
                    logger.exception(f"Audio chunk processing error for {client_id}")
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": "Audio chunk processing error"}),
                        client_id
                    )

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected.")
        manager.is_recording[client_id] = False
        if interim_task and not interim_task.done():
            interim_task.cancel()
            try:
                await interim_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)
    except Exception as e:
        logger.exception(f"Unhandled server error for {client_id}: {e}")
        manager.is_recording[client_id] = False
        if interim_task and not interim_task.done():
            interim_task.cancel()
            try:
                await interim_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)
        
        
    except Exception:
        logger.exception(f"Unhandled server error for {client_id}")
        manager.is_recording[client_id] = False
        if interim_task and not interim_task.done():
            interim_task.cancel()
            try:
                await interim_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)

# Add this route to serve topup.html at /topup.html
@app.get("/topup.html", response_class=HTMLResponse)
async def get_topup():
    try:
        with open("static/topup.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("topup.html not found in static directory")
        return HTMLResponse(content="Error: topup.html not found", status_code=404)

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return HTMLResponse(content=open("static/index.html").read())

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_model_loaded": MODEL_LOADED,
        "rasa_model_loaded": NLU_PROCESSOR is not None,
        "device": str(DEVICE) if DEVICE else "not set"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# -----------------------------------------------------------------------------------------\