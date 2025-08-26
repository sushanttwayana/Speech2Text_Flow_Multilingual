# ---------------------- MULTILINGUAL SPEECH TO TEXT WITH TRANSLATION ---------------------------------

import asyncio
import warnings
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import json
import base64
import logging
from datetime import datetime
import os
from contextlib import asynccontextmanager
from rasa.core.agent import Agent
import time


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Device Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for Whisper
MODEL_LOADED = False
PIPE = None  # ASR pipeline
SAMPLE_RATE = 16000  # Whisper model sample rate
client_languages: Dict[str, str] = {}

# Global variables for Silero VAD
VAD_MODEL = None
GET_SPEECH_TIMESTAMPS = None
COLLECT_CHUNKS = None

# Global variable for Rasa
NLU_PROCESSOR = None

# Global variable for Translation
TRANSLATOR = None

# --- IMPROVED TRANSLATION MODEL CLASS ---
class MobileTranslator:
    def __init__(self, model_size="distilled-600M"):
        self.model_name = f"facebook/nllb-200-{model_size}"

        # Language codes for Malay and Hindi
        self.language_codes = {
            "english": "eng_Latn",
            "hindi": "hin_Deva",
            "malay": "zsm_Latn"
        }

        logger.info("Loading translation model...")
        self.model, self.tokenizer = None, None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=False
            )

            if torch.cuda.is_available():
                logger.info("Trying GPU load for translation model...")

                # Estimate max GPU memory you want to use in bytes (e.g. 14GB)
                max_memory = {0: "14GB"}  # adjust based on your GPU RAM
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",      # very important: no manual .to(DEVICE)
                        max_memory=max_memory,  # control memory allocation
                        use_safetensors=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info(f"Translation model loaded on GPU: {self.model.device}")

                except Exception as e:
                    logger.warning(f"GPU load failed ({e}), falling back to CPU...")
                    self.model = None
            else:
                self.model = None

            # CPU fallback
            if self.model is None:
                logger.info("Loading translation model on CPU (may be slower)...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to('cpu')
                logger.info("Translation model loaded on CPU.")

            self.model.eval()
            self._cache_language_tokens()
            logger.info("✅ Translation model ready!")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            self.model, self.tokenizer = None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED, PIPE, DEVICE, NLU_PROCESSOR, VAD_MODEL, GET_SPEECH_TIMESTAMPS, COLLECT_CHUNKS, TRANSLATOR
    try:
        use_gpu = DEVICE.type == 'cuda'
        logger.info(f"Using device: {DEVICE}")

        print("Before loading the Whisper model")
        local_dir = "whisper_models/whisper-large-v3-turbo/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
        if not os.path.exists(local_dir):
            logger.warning(f"Local Whisper model not found at {local_dir}, downloading from HuggingFace...")
            model_id = "openai/whisper-large-v3-turbo"
        else:
            model_id = local_dir

        torch_dtype = torch.float16 if use_gpu and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model = model.to(DEVICE)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                logger.warning("GPU OOM for Whisper, falling back to CPU...")
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                model = model.to(torch.device('cpu'))
            else:
                raise

        print("After loading the Whisper model")
        processor = AutoProcessor.from_pretrained(model_id)

        PIPE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=1,
            torch_dtype=torch_dtype,
            device=0 if use_gpu else -1
        )
        MODEL_LOADED = True
        logger.info("Whisper model loaded and pipeline ready.")

        # Load Silero VAD (CPU)
        logger.info("Loading Silero VAD...")
        try:
            vad_bundle = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
            VAD_MODEL, vad_utils = vad_bundle
            (GET_SPEECH_TIMESTAMPS, _, _, _, COLLECT_CHUNKS) = vad_utils
            logger.info("Silero VAD loaded.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            VAD_MODEL = None

        # Load Translation Model
        logger.info("Loading Translation Model...")
        try:
            TRANSLATOR = MobileTranslator(model_size="distilled-600M")
            logger.info("✅ Translation model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            TRANSLATOR = None
            logger.warning("Translation functionality will be disabled.")

        # Load Rasa model
        logger.info("Loading Rasa NLU model...")
        try:
            NLU_MODEL_PATH = "models/nlu_two_intent_classifier.tar.gz"
            if os.path.exists(NLU_MODEL_PATH):
                NLU_PROCESSOR = NLUProcessor.create(NLU_MODEL_PATH)
                if NLU_PROCESSOR:
                    logger.info("✅ Rasa NLUProcessor initialized successfully.")
                else:
                    logger.error("Failed to initialize Rasa NLUProcessor.")
            else:
                logger.warning(f"Rasa model not found at {NLU_MODEL_PATH}")
                NLU_PROCESSOR = None
        except Exception as e:
            logger.error(f"Failed to load Rasa model: {e}")
            NLU_PROCESSOR = None

        yield

    except Exception as e:
        logger.exception("Failed to load models:")
        MODEL_LOADED = False
        NLU_PROCESSOR = None
        TRANSLATOR = None
        yield
    finally:
        logger.info("Shutting down...")


app = FastAPI(title="Multilingual Speech-to-Text and Intent Classification API", version="4.0.0", lifespan=lifespan)

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
    def __init__(self, agent: Agent):
        self.agent = agent
        if self.agent:
            logger.info("✅ Rasa NLUProcessor initialized successfully.")

    @classmethod
    def create(cls, model_path: str) -> Optional["NLUProcessor"]:
        try:
            agent = Agent.load(model_path=model_path)
            return cls(agent)
        except Exception as e:
            logger.error(f"❌ Error loading Rasa model: {e}")
            return None

    async def classify_intent(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.agent:
            logger.error("Agent not available. Cannot classify intent.")
            return None
        result = await self.agent.parse_message(text)
        return result or None

    async def process_command(self, text: str, confidence_threshold: float = 0.80):
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
                elif intent_name == "multicurrency_wallet":
                    logger.info("Action: Initiating 'multicurrency wallet' flow...")
                    return {
                        "action": "multicurrency_wallet",
                        "message": "Initiating 'multicurrency wallet' flow...",
                        "form_data": form_data,
                        "entities": entities
                    }
                elif intent_name == "card_to_card_transfer":
                    logger.info("Action: Initiating 'card to card transfer' flow...")
                    return {
                        "action": "card_to_card_transfer",
                        "message": "Initiating 'card to card transfer' flow...",
                        "form_data": form_data
                    }
                else:
                    logger.info(f"Action: Intent recognized, but no specific action defined: {intent_name}")
                    return {
                        "action": "unknown",
                        "message": "Intent recognized, but no specific action defined.",
                        "form_data": form_data
                    }
            else:
                logger.info(f"Action: Confidence ({confidence:.2f}) below threshold ({confidence_threshold}).")
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
        if VAD_MODEL and GET_SPEECH_TIMESTAMPS and COLLECT_CHUNKS:
            audio_torch = torch.from_numpy(audio_np_segment).float()
            speech_timestamps = GET_SPEECH_TIMESTAMPS(
                audio_torch,
                VAD_MODEL,
                sampling_rate=SAMPLE_RATE,
                threshold=0.3,
                min_speech_duration_ms=100,
                min_silence_duration_ms=100,
                return_seconds=False
            )
            if not speech_timestamps:
                return ""
            filtered_audio_torch = COLLECT_CHUNKS(speech_timestamps, audio_torch)
            filtered_audio_np = filtered_audio_torch.numpy()
        else:
            filtered_audio_np = audio_np_segment

        whisper_lang_map = {
            "ms": "ms",
            "my": "ms",
            "hi": "hi",
            "en": "en"
        }
        whisper_lang = whisper_lang_map.get(language, language)

        generate_kwargs = {
            "language": whisper_lang,
            "temperature": 0.0,
            "num_beams": 5,
            "task": "transcribe"
        }
        result = PIPE(filtered_audio_np, generate_kwargs=generate_kwargs)
        return result["text"].strip()
    except Exception:
        logger.exception("Transcription error:")
        return ""


async def translate_text(text: str, source_language: str) -> str:
    if not TRANSLATOR:
        logger.warning("Translation model not loaded, returning original text.")
        return text
    if not text or text.strip() == "":
        return text

    translator_lang_map = {
        "ms": "malay",
        "my": "malay",
        "hi": "hindi",
        "en": "english"
    }

    source_lang = translator_lang_map.get(source_language, "english")

    if source_lang == "english":
        logger.info(f"Text already in English, skipping translation: {text}")
        return text

    try:
        logger.info(f"Translating from {source_lang} to English: {text}")
        translation_result = TRANSLATOR.translate(text, source_lang, "english")
        if translation_result.get("success"):
            translated_text = translation_result["translated_text"]
            logger.info(f"Translation successful: {translated_text} (took {translation_result['processing_time']}s)")
            return translated_text
        else:
            logger.error(f"Translation failed: {translation_result.get('error')}")
            return text
    except Exception as e:
        logger.exception(f"Translation error: {e}")
        return text


async def transcribe_full_audio(client_id: str, full_audio: np.ndarray, language: str) -> str:
    MAX_CHUNK = SAMPLE_RATE * 30
    segments = []
    start = 0
    while start < len(full_audio):
        segment = full_audio[start: start + MAX_CHUNK]
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
                original_transcription = await transcribe_segment(chunk, language)
                if original_transcription:
                    english_transcription = await translate_text(original_transcription, language)
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "transcription",
                            "text": original_transcription,
                            "translated_text": english_transcription,
                            "is_final": False
                        }),
                        client_id
                    )
                    manager.transcribed_segments[client_id].append(original_transcription)
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
            original_transcription = await transcribe_segment(audio_buffer, language)
            if original_transcription:
                english_transcription = await translate_text(original_transcription, language)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "transcription",
                        "text": original_transcription,
                        "translated_text": english_transcription,
                        "is_final": False
                    }),
                    client_id
                )
                manager.transcribed_segments[client_id].append(original_transcription)
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
                supported = ["en", "ms", "my", "hi"]
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

                final_audio = np.array([], dtype=np.float32)
                if manager.audio_buffers[client_id]:
                    final_audio = np.concatenate(manager.audio_buffers[client_id])
                    logger.info(f"Processing remaining audio buffer for {client_id}, size: {len(final_audio)} samples")
                    manager.audio_buffers[client_id] = []

                final_segments = manager.transcribed_segments.get(client_id, [])
                if final_audio.size > 0:
                    original_transcription = await transcribe_segment(final_audio, client_languages.get(client_id, "en"))
                    if original_transcription:
                        final_segments.append(original_transcription)
                        logger.info(f"Appended final transcription for {client_id}: {original_transcription}")

                final_original_text = merge_transcriptions(final_segments) or "No speech detected."
                logger.info(f"Final original transcription for {client_id}: {final_original_text}")

                final_english_text = await translate_text(final_original_text, client_languages.get(client_id, "en"))
                logger.info(f"Final English translation for {client_id}: {final_english_text}")

                result = {"action": "error", "message": "No transcription or Rasa processor not initialized.", "form_data": {}}

                if NLU_PROCESSOR and final_english_text != "No speech detected.":
                    try:
                        result = await NLU_PROCESSOR.process_command(final_english_text)
                        logger.info(f"Rasa result for {client_id}: {result}")
                    except Exception as e:
                        logger.exception(f"Rasa processing error for {client_id}: {e}")
                        result = {"action": "error", "message": f"Rasa processing failed: {str(e)}", "form_data": {}}

                await manager.send_personal_message(
                    json.dumps({
                        "type": "transcription",
                        "text": final_original_text,
                        "translated_text": final_english_text,
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
                        }
                    }
                    logger.info(f"Sending navigate message to {client_id}: {navigate_message}")
                    await manager.send_personal_message(
                        json.dumps(navigate_message),
                        client_id
                    )

                elif result.get("action") == "multicurrency_wallet" and result.get("form_data", {}).get("amount"):
                    amount = result["form_data"]["amount"]
                    navigate_message = {
                        "type": "navigate",
                        "route": "/multicurrency.html",
                        "prefill": {
                            "amount": amount,
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


@app.get("/topup.html", response_class=HTMLResponse)
async def get_topup():
    try:
        with open("static/topup.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("topup.html not found in static directory")
        return HTMLResponse(content="Error: topup.html not found", status_code=404)


@app.get("/multicurrency.html", response_class=HTMLResponse)
async def get_multicurrency():
    try:
        with open("static/multicurrency.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("multicurrency.html not found in static directory")
        return HTMLResponse(content="Error: multicurrency.html not found", status_code=404)


@app.get("/", response_class=HTMLResponse)
async def get_root():
    return HTMLResponse(content=open("static/index.html").read())


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_model_loaded": MODEL_LOADED,
        "rasa_model_loaded": NLU_PROCESSOR is not None,
        "translator_loaded": TRANSLATOR is not None,
        "device": str(DEVICE)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
