import asyncio
import warnings
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import json
import base64
import logging
from datetime import datetime
import os
from contextlib import asynccontextmanager
import re
from word2number import w2n

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

# Global variable for NLU
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

        # Load GPT NLU model
        logger.info("Loading GPT NLU model...")
        NLU_PROCESSOR = GPTNLUProcessor(model_name="../models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a")
        if NLU_PROCESSOR:
            logger.info("✅ Mistral GPT NLUProcessor initialized successfully.")
        else:
            logger.error("Failed to initialize GPT NLUProcessor.")

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

# GPT processor 
class GPTNLUProcessor:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading GPT NLU model {model_name} on device {self.device} with 4-bit quantization...")
        torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map=self.device,
        )
        # Set pad token to suppress warning
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        logger.info(f"GPT NLU model loaded: {model_name}")

    def _words_to_num(self, text):
        """Convert number words to numeric string. Fallback to digits extraction."""
        try:
            # word_to_num requires lowercased words only
            return str(w2n.word_to_num(text.lower()))
        except Exception:
            digits = re.findall(r"\d+", text)
            return digits[0] if digits else None

    def _detect_currency(self, text):
        """Basic keyword-based currency detection."""
        currency_map = {
            "rupee": "INR", "rs": "INR", "rs.": "INR", "ringgit": "RM", "rm": "RM",
            "dollar": "USD", "$": "USD",
            "yuan": "CNY", "yen": "JPY", "lkr": "LKR", "rupiah": "IDR"
        }
        text_lower = text.lower()
        for k, v in currency_map.items():
            if k in text_lower:
                return v
        return None

    def classify_and_extract(self, user_text: str):
        """Prompt the model and parse output JSON for intent and entities."""
        logger.info(f"Starting NLU classification for text: {user_text}")
        prompt = f"""
You are an assistant that extracts structured JSON from user commands about money transactions.

Rules:
- action: One of ["topup_wallet", "send_money", "top_up", "card_to_card_transfer","multi_currency_wallet", "unknown"]
- amount: numeric only, convert words to digits ("thousand" → "1000", "two hundred fifty" -> "250")
- currency: 3-letter or symbol ("INR", "USD", "RM", "NPR", "SGD", "MYR", "IDR" etc), guess if possible
- Ignore extraneous text like greetings or thanks.
- Handle minor phrasing variations (e.g., "top of" might mean "top up").
- Output JSON only, no explanations.

Example:
Input: "I want to top up thousand rupees into my wallet"
Output: {{"action": "topup_wallet", "form_data": {{"amount": "1000", "currency": "INR"}}}}

Example:
Input: "Load 350 ringgit into wallet"
Output: {{"action": "topup_wallet", "form_data": {{"amount": "350", "currency": "MYR"}}}}

Example:
Input: "Transfer 100 Indonesian Rupiah into my wallet"
Output: {{"action": "topup_wallet", "form_data": {{"amount": "100", "currency": "IDR"}}}}

Now process:
"{user_text}"
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
        raw_output = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        logger.info(f"Model generation completed. Raw output: {raw_output}")

        # Extract JSON part safely
        try:
            json_str = re.search(r"\{[\s\S]*\}", raw_output).group(0)
            parsed = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse JSON from model output: {e}")
            parsed = {"action": "unknown", "form_data": {}}

        # Post-process amount and currency fields
        if "form_data" in parsed:
            if "amount" in parsed["form_data"]:
                parsed["form_data"]["amount"] = self._words_to_num(str(parsed["form_data"]["amount"]))
            if not parsed["form_data"].get("currency"):
                parsed["form_data"]["currency"] = self._detect_currency(user_text)

        # Add helpful message for frontend
        action = parsed.get("action", "unknown")
        if action == "topup_wallet":
            parsed["message"] = "Initiating 'wallet top-up' flow..."
        elif action == "send_money":
            parsed["message"] = "Initiating 'send money' flow..."
        elif action == "top_up":
            parsed["message"] = "Initiating 'mobile top-up' flow..."
        elif action == "card_to_card_transfer":
            parsed["message"] = "Initiating 'card to card transfer' flow..."
        else:
            parsed["message"] = "Intent recognized, but no specific action defined."

        logger.info("NLU classification completed.")
        return parsed

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
                result = {"action": "error", "message": "No transcription or NLU processor not initialized.", "form_data": {}}

                # Process with NLU
                if NLU_PROCESSOR and final_text != "No speech detected.":
                    try:
                        result = NLU_PROCESSOR.classify_and_extract(final_text)
                        logger.info(f"NLU result for {client_id}: {result}")
                    except Exception as e:
                        logger.exception(f"NLU processing error for {client_id}: {e}")
                        result = {"action": "error", "message": f"NLU processing failed: {str(e)}", "form_data": {}}

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
                        "route": "/topup.html",
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
        "nlu_model_loaded": NLU_PROCESSOR is not None,
        "device": str(DEVICE) if DEVICE else "not set"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)