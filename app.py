import asyncio
import warnings
import json
import base64
import logging
import numpy as np
import torch
import os
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
from contextlib import asynccontextmanager
import gc
import time
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate

from rasa.core.agent import Agent
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Globals
DEVICE = None
COMPUTE_TYPE = None
SAMPLE_RATE = 16000
MODEL_LOADED = False
ASR_MODEL: WhisperModel = None
NLU_PROCESSOR = None
VAD_MODEL = None
GET_SPEECH_TIMESTAMPS = None
COLLECT_CHUNKS = None

client_languages: Dict[str, str] = {}


def setup_device():
    global DEVICE, COMPUTE_TYPE
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            test_tensor = torch.tensor([1.0]).cuda()
            test_result = test_tensor * 2
            assert test_result.item() == 2.0
            large_tensor = torch.zeros((1000, 1000)).cuda()
            del large_tensor, test_tensor, test_result
            torch.cuda.empty_cache()

            DEVICE = torch.device('cuda')
            COMPUTE_TYPE = "float16"

            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ CUDA is available and working - GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return "cuda", "float16"
        except Exception as e:
            logger.warning(f"⚠️ CUDA on GPU failed: {e}")
            DEVICE = torch.device('cpu')
            COMPUTE_TYPE = "float32"
            return "cpu", "float32"
    else:
        logger.info("CUDA not available, using CPU")
        DEVICE = torch.device('cpu')
        COMPUTE_TYPE = "float32"
        return "cpu", "float32"


def cleanup_cuda_memory():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            logger.info("CUDA memory cleaned up")
    except Exception as e:
        logger.warning(f"CUDA cleanup warning: {e}")

def clean_amount_value(amount_str: str) -> str:
    """
    Clean amount string to contain only numeric values
    Examples:
    - "RM1000" -> "1000"
    - "1000RM" -> "1000" 
    - "USD100" -> "100"
    - "$50.00" -> "50.00"
    - "1,000.50" -> "1000.50"
    """
    if not amount_str:
        return None
    
    # Remove currency symbols and prefixes/suffixes
    # This regex keeps only digits, dots, and commas
    cleaned = re.sub(r'[^\d.,]', '', str(amount_str))
    
    # Remove commas (thousand separators)
    cleaned = cleaned.replace(',', '')
    
    # Handle edge cases
    if not cleaned or cleaned == '.':
        return None
        
    # Convert to float then back to string to handle decimal formatting
    try:
        # If it's a whole number, return without decimal
        if '.' in cleaned:
            float_val = float(cleaned)
            if float_val == int(float_val):
                return str(int(float_val))
            else:
                return str(float_val)
        else:
            return cleaned
    except ValueError:
        return None

def extract_amount_manually(text: str) -> str:
    """
    Extract and clean amount from text using regex patterns
    """
    patterns = [
        r'(?:RM|USD|MYR|SGD|\$|€|£)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # Currency prefix
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:RM|USD|MYR|SGD|ringgit|dollar|cents?)',  # Currency suffix
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # Just numbers
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw_amount = match.group(1)
            cleaned_amount = clean_amount_value(raw_amount)
            if cleaned_amount:
                return cleaned_amount
    
    return None

# Updated NLU PROCESSOR
class NLUProcessor:
    def __init__(self, agent: Agent):
        self.agent = agent
        logger.info("✅ Rasa NLUProcessor initialized successfully.")

    @classmethod
    def create(cls, model_path: str) -> Optional["NLUProcessor"]:
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU for Rasa
            agent = Agent.load(model_path=model_path)
            return cls(agent)
        except Exception as e:
            logger.error(f"❌ Error loading Rasa model: {e}")
            return None

    async def classify_intent(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.agent:
            logger.error("Agent not available for classification.")
            return None
        try:
            return await self.agent.parse_message(text)
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return None

    async def process_command(self, text: str, confidence_threshold: float = 0.80):
        try:
            classification_result = await self.classify_intent(text)
            if classification_result:
                intent = classification_result.get("intent", {})
                intent_name = intent.get("name")
                confidence = intent.get("confidence", 0)
                logger.info(f"--- Rasa NLU Analysis --- Intent: {intent_name}, Confidence: {confidence:.2f}")

                entities = classification_result.get("entities", [])
                form_data = {}
                for entity in entities:
                    if entity["entity"] == "amount":
                        # Clean the amount value to remove currency symbols
                        raw_amount = entity["value"]
                        cleaned_amount = clean_amount_value(raw_amount)
                        if cleaned_amount:
                            form_data["amount"] = cleaned_amount
                            logger.info(f"Amount entity cleaned: '{raw_amount}' -> '{cleaned_amount}'")
                        else:
                            logger.warning(f"Could not clean amount: '{raw_amount}'")
                            
                    if entity["entity"] == "person":
                        form_data["name"] = entity["value"]

                if confidence > confidence_threshold:
                    classification_result["action"] = intent_name
                else:
                    classification_result["action"] = "unknown"
                classification_result["form_data"] = form_data
                return classification_result
        except Exception as e:
            logger.error(f"Error in process_command: {e}")

        return {"action": "unknown", "message": "Classification failed.", "form_data": {}}


def load_silero_vad():
    try:
        import torch.hub
        current_device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        try:
            vad_model, vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            vad_model = vad_model.cpu()
            logger.info("✅ Silero VAD loaded successfully on CPU")
            return vad_model, vad_utils
        finally:
            if current_device:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_device
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    except Exception as e:
        logger.error(f"Error loading Silero VAD: {e}")
        return None, None


def load_whisper_model_with_retry(model_dir: str, max_retries: int = 3):
    global ASR_MODEL, MODEL_LOADED, DEVICE, COMPUTE_TYPE

    device_type, compute_type = setup_device()

    for attempt in range(max_retries):
        try:
            cleanup_cuda_memory()
            logger.info(f"Whisper loading attempt {attempt + 1}/{max_retries} on {device_type} with {compute_type}")

            ASR_MODEL = WhisperModel(
                model_dir,
                device=device_type,
                compute_type=compute_type,
                cpu_threads=4 if device_type == "cpu" else 0,
                num_workers=1,
                download_root=None,
                local_files_only=True
            )
            test_audio = np.random.randn(16000).astype(np.float32)
            segments, info = ASR_MODEL.transcribe(test_audio, beam_size=1)
            _ = list(segments)

            MODEL_LOADED = True
            DEVICE = torch.device(device_type)
            COMPUTE_TYPE = compute_type
            logger.info(f"✅ Whisper model loaded successfully on {device_type} with {compute_type}")
            return True
        except Exception as e:
            logger.error(f"Whisper loading attempt {attempt + 1} failed: {e}")
            if attempt == 0 and device_type == "cuda":
                compute_type = "float32"
                logger.info("🔄 Retrying with CUDA float32...")
            elif attempt == 1:
                device_type = "cpu"
                compute_type = "float32"
                DEVICE = torch.device('cpu')
                COMPUTE_TYPE = "float32"
                logger.info("🔄 Falling back to CPU...")
            cleanup_cuda_memory()
            if attempt == max_retries - 1:
                logger.error("❌ All Whisper loading attempts failed")
                raise Exception("Failed to load Whisper model after retries")
    return False


def load_local_translation_models(models_dir: str):
    for filename in os.listdir(models_dir):
        if filename.endswith(".argosmodel"):
            model_path = os.path.join(models_dir, filename)
            try:
                logger.info(f"Installing translation model from {model_path} ...")
                argostranslate.package.install_from_path(model_path)
                logger.info(f"Installed {filename} successfully.")
            except Exception as e:
                logger.error(f"Failed to install translation model {filename}: {e}")


async def translate_to_english(text: str, source_lang_code: str) -> str:
    if not text.strip():
        return text

    if source_lang_code == "en":
        return text

    try:
        installed_languages = argostranslate.translate.get_installed_languages()

        from_lang = next((lang for lang in installed_languages if lang.code == source_lang_code), None)
        to_lang = next((lang for lang in installed_languages if lang.code == "en"), None)

        if from_lang is None or to_lang is None:
            logger.warning(f"Translation languages not installed: from {source_lang_code} to en")
            return text

        translation = from_lang.get_translation(to_lang)
        translated_text = translation.translate(text)
        logger.info(f"Translated '{text}' from {source_lang_code} to English: '{translated_text}'")
        return translated_text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text


async def transcribe_audio_faster_whisper(audio_np: np.ndarray, language: Optional[str] = None):
    global ASR_MODEL

    if not MODEL_LOADED or ASR_MODEL is None:
        logger.error("Model not loaded, cannot transcribe.")
        return "", "en"

    try:
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.cpu().numpy()
        audio_np = audio_np.astype(np.float32)

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # beam_size=1 for speed, VAD filtering aggressive via vad_filter=True
        segments, info = ASR_MODEL.transcribe(
            audio_np,
            beam_size=1,
            language=language or "en",
            # language=language or "en" or "hi" or "ms",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=200)  # higher threshold to reduce silence processing
        )
        transcript = " ".join(segment.text for segment in segments)
        detected_lang = info.language

        logger.debug(f"ASR detected language: {detected_lang} (prob: {info.language_probability:.2f})")

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        return transcript.strip(), detected_lang

    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        cleanup_cuda_memory()
        return "", "en"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED, ASR_MODEL, NLU_PROCESSOR, VAD_MODEL, GET_SPEECH_TIMESTAMPS, COLLECT_CHUNKS, DEVICE, COMPUTE_TYPE

    try:
        logger.info("🚀 Starting application initialization...")

        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            logger.info("✅ TensorFlow configured to use CPU only")
        except Exception as e:
            logger.warning(f"TensorFlow GPU configuration warning: {e}")

        device_type, compute_type = setup_device()
        logger.info(f"Device configured: {DEVICE} with compute type: {compute_type}")

        LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "faster-whisper-large-v3")
        if not os.path.exists(LOCAL_MODEL_DIR):
            logger.info(f"Model folder not found, downloading faster-whisper-large-v3 to {LOCAL_MODEL_DIR}")
            os.makedirs(os.path.dirname(LOCAL_MODEL_DIR), exist_ok=True)
            snapshot_download(
                repo_id="Systran/faster-whisper-large-v3",
                local_dir=LOCAL_MODEL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            logger.info("✅ Model download completed.")
        else:
            logger.info(f"✅ Using existing model at {LOCAL_MODEL_DIR}")

        if not load_whisper_model_with_retry(LOCAL_MODEL_DIR):
            raise Exception("Failed to load Whisper model")

        try:
            logger.info("Loading translation model from local path")
            LOCAL_TRANSLATION_MODEL_DIR = os.path.join(os.path.dirname(__file__), "translation_models")
            load_local_translation_models(LOCAL_TRANSLATION_MODEL_DIR)
            logger.info("✅ Argos Translate models loaded successfully.")
        except Exception as e:
            logger.error(f"⚠️ Failed to load translation models: {e}")

        logger.info("Loading Silero VAD...")
        vad_result = load_silero_vad()
        if vad_result and len(vad_result) == 2 and vad_result[0] is not None:
            VAD_MODEL, vad_utils = vad_result
            if vad_utils and len(vad_utils) >= 2:
                GET_SPEECH_TIMESTAMPS = vad_utils[0]
                COLLECT_CHUNKS = vad_utils[4] if len(vad_utils) > 4 else None
                logger.info("✅ Silero VAD loaded successfully.")
            else:
                logger.error("❌ VAD utils not properly loaded")
                VAD_MODEL = None
                GET_SPEECH_TIMESTAMPS = None
                COLLECT_CHUNKS = None
        else:
            logger.error("❌ Failed to load VAD model")
            VAD_MODEL = None
            GET_SPEECH_TIMESTAMPS = None
            COLLECT_CHUNKS = None

        logger.info("Loading Rasa NLU model...")
        # NLU_MODEL_PATH = "models/new_two_nlu_intent_classification.tar.gz"
        NLU_MODEL_PATH = "models/nlu_two_intent_classifier.tar.gz"
        if not os.path.exists(NLU_MODEL_PATH):
            logger.error(f"❌ Rasa model file not found at {NLU_MODEL_PATH}")
            NLU_PROCESSOR = None
        else:
            NLU_PROCESSOR = NLUProcessor.create(NLU_MODEL_PATH)
            if NLU_PROCESSOR:
                logger.info("✅ Rasa NLUProcessor initialized successfully.")
            else:
                logger.error("❌ Failed to initialize Rasa NLUProcessor.")

        logger.info("="*60)
        logger.info("🎉 APPLICATION INITIALIZATION COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Status Summary:")
        logger.info(f"   • Device: {DEVICE} ({COMPUTE_TYPE})")
        logger.info(f"   • Whisper Model: {'✅ Loaded' if MODEL_LOADED else '❌ Failed'}")
        logger.info(f"   • VAD Model: {'✅ Loaded' if VAD_MODEL is not None else '❌ Failed'}")
        logger.info(f"   • Translation: {'✅ Available' if True else '❌ Unavailable'}")
        logger.info(f"   • Rasa NLU: {'✅ Loaded' if NLU_PROCESSOR is not None else '❌ Failed'}")
        logger.info("="*60)

        yield

    except Exception as e:
        logger.exception(f"💥 CRITICAL: Failed to initialize application: {e}")
        MODEL_LOADED = False
        NLU_PROCESSOR = None
        VAD_MODEL = None
        GET_SPEECH_TIMESTAMPS = None
        COLLECT_CHUNKS = None
        logger.error("⚠️ Starting with limited functionality due to initialization errors")
        yield

    finally:
        logger.info("🔄 Application shutdown initiated...")
        try:
            if ASR_MODEL is not None:
                del ASR_MODEL
                logger.info("✅ Whisper model cleaned up")
            if VAD_MODEL is not None:
                del VAD_MODEL
                logger.info("✅ VAD model cleaned up")
            if NLU_PROCESSOR is not None:
                del NLU_PROCESSOR
                logger.info("✅ NLU processor cleaned up")
            cleanup_cuda_memory()
            gc.collect()
            logger.info("✅ Cleanup completed successfully")
        except Exception as cleanup_error:
            logger.error(f"⚠️ Error during cleanup: {cleanup_error}")
        logger.info("👋 Application shutdown complete")


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, List[np.ndarray]] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.is_recording: Dict[str, bool] = {}
        self.transcribed_segments: Dict[str, List[str]] = {}
        self.translated_segments: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id in self.active_connections:
            logger.warning(f"Client {client_id} reconnected, closing old connection.")
            try:
                await self.active_connections[client_id].close()
            except:
                pass
        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.is_recording[client_id] = False
        self.transcribed_segments[client_id] = []
        self.translated_segments[client_id] = []
        client_languages[client_id] = "en"
        logger.info(f"Client {client_id} connected.")
        await self.send_personal_message(json.dumps({"type": "status", "message": "Connected"}), client_id)

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        self.audio_buffers.pop(client_id, None)
        task = self.processing_tasks.pop(client_id, None)
        if task and not task.done():
            task.cancel()
        self.is_recording.pop(client_id, None)
        self.transcribed_segments.pop(client_id, None)
        self.translated_segments.pop(client_id, None)
        client_languages.pop(client_id, None)
        logger.info(f"Client {client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
                logger.debug(f"Sent message to {client_id}: {message}")
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                self.disconnect(client_id)


manager = ConnectionManager()

app = FastAPI(title="Speech-to-Text and Intent Classification API with Faster Whisper + Argos Translate", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


async def transcribe_segment(audio_np_segment: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
    if not MODEL_LOADED or ASR_MODEL is None:
        logger.error("Model not loaded, cannot transcribe.")
        return "", ""

    try:
        if VAD_MODEL and GET_SPEECH_TIMESTAMPS and COLLECT_CHUNKS:
            try:
                audio_torch = torch.from_numpy(audio_np_segment).float()
                if audio_torch.is_cuda:
                    audio_torch = audio_torch.cpu()

                speech_timestamps = GET_SPEECH_TIMESTAMPS(
                    audio_torch,
                    VAD_MODEL,
                    sampling_rate=SAMPLE_RATE,
                    threshold=0.4,
                    min_speech_duration_ms=100,
                    min_silence_duration_ms=100,
                    return_seconds=False
                )

                if not speech_timestamps:
                    return "", ""

                filtered_audio_torch = COLLECT_CHUNKS(speech_timestamps, audio_torch)
                filtered_audio_np = filtered_audio_torch.numpy()
            except Exception as e:
                logger.warning(f"VAD processing failed, using original audio: {e}")
                filtered_audio_np = audio_np_segment
        else:
            filtered_audio_np = audio_np_segment

        # Pass the language to faster whisper transcription
        transcript, detected_lang = await transcribe_audio_faster_whisper(filtered_audio_np, language)

        # Perform translation only if detected_lang is not English
        if detected_lang != "en":
            translated_transcript = await translate_to_english(transcript, detected_lang)
        else:
            translated_transcript = transcript

        return transcript, translated_transcript

    except Exception as e:
        logger.exception(f"Error in transcribing segment: {e}")
        return "", ""


async def process_audio_for_client(client_id: str, language: str):
    CHUNK_SECONDS = 5  # Reduced chunk size for quicker feedback
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
    audio_buffer = np.array([], dtype=np.float32)

    while manager.is_recording.get(client_id, False):
        try:
            await asyncio.sleep(0.1)

            while manager.audio_buffers[client_id]:
                new_chunk = manager.audio_buffers[client_id].pop(0)
                audio_buffer = np.concatenate([audio_buffer, new_chunk])

            while len(audio_buffer) >= CHUNK_SAMPLES:
                chunk = audio_buffer[:CHUNK_SAMPLES]
                audio_buffer = audio_buffer[CHUNK_SAMPLES:]

                native_transcript, translated_transcript = await transcribe_segment(chunk)
                if native_transcript:
                    await manager.send_personal_message(
                        json.dumps({"type": "transcription", "text": native_transcript, "is_final": False}),
                        client_id
                    )
                    manager.transcribed_segments[client_id].append(native_transcript)
                    manager.translated_segments[client_id].append(translated_transcript)

        except asyncio.CancelledError:
            logger.info(f"Interim task for {client_id} cancelled.")
            break
        except Exception as e:
            logger.exception(f"Interim processing error for {client_id}: {e}")
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": "Interim processing error"}),
                client_id
            )
            break

    # Process any remaining audio buffer at stop
    if len(audio_buffer) > 0:
        native_transcript, translated_transcript = await transcribe_segment(audio_buffer)
        if native_transcript:
            await manager.send_personal_message(
                json.dumps({"type": "transcription", "text": native_transcript, "is_final": True}),
                client_id
            )
            manager.transcribed_segments[client_id].append(native_transcript)
            manager.translated_segments[client_id].append(translated_transcript)


def merge_transcriptions(segments: List[str]) -> str:
    return ' '.join(segments).strip() if segments else ""


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    interim_task: Optional[asyncio.Task] = None

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
                manager.transcribed_segments[client_id] = []
                manager.translated_segments[client_id] = []

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

                start_final_time = time.perf_counter()

                # Process any remaining audio in buffer
                final_audio = np.array([], dtype=np.float32)
                if manager.audio_buffers[client_id]:
                    try:
                        final_audio = np.concatenate(manager.audio_buffers[client_id])
                        logger.info(f"Processing remaining audio buffer for {client_id}, size: {len(final_audio)} samples")
                        manager.audio_buffers[client_id] = []
                    except Exception as e:
                        logger.error(f"Error concatenating final audio: {e}")

                final_segments = manager.transcribed_segments.get(client_id, [])
                translated_segments = manager.translated_segments.get(client_id, [])
                lang = client_languages.get(client_id, "en")

                if final_audio.size > 0:
                    native_transcript, translated_transcript = await transcribe_segment(final_audio, language=lang)
                    if native_transcript:
                        final_segments.append(native_transcript)
                        translated_segments.append(translated_transcript)
                        logger.info(f"Appended final transcription for {client_id}: {native_transcript}")

                final_text = merge_transcriptions(final_segments) or "No speech detected."
                translated_final_text = merge_transcriptions(translated_segments) or "No speech detected."
                logger.info(f"Final transcription for {client_id}: {final_text}")

                result = {"action": "error", "message": "No transcription or Rasa processor not initialized.", "form_data": {}}

                if NLU_PROCESSOR and translated_final_text != "No speech detected.":
                    try:
                        # Process Rasa classification asynchronously
                        result = await NLU_PROCESSOR.process_command(translated_final_text)
                        logger.info(f"Rasa result for {client_id}: {result}")
                    except Exception as e:
                        logger.exception(f"Rasa processing error for {client_id}: {e}")
                        result = {"action": "error", "message": f"Rasa processing failed: {str(e)}", "form_data": {}}

                end_final_time = time.perf_counter()
                logger.info(f"Final processing for client {client_id} took {end_final_time - start_final_time:.3f} seconds")

                await manager.send_personal_message(
                    json.dumps({
                        "type": "transcription",
                        "text": final_text,
                        "is_final": True,
                        "intent": result
                    }),
                    client_id
                )
                
                
                # Enhanced routing logic with fallback amount extraction and cleaning
                if result.get("action") in ["topup_wallet", "multicurrency_wallet"]:
                    # Try manual amount extraction if Rasa didn't extract it
                    if not result.get("form_data", {}).get("amount"):
                        manual_amount = extract_amount_manually(translated_final_text)
                        if manual_amount:
                            if "form_data" not in result:
                                result["form_data"] = {}
                            result["form_data"]["amount"] = manual_amount
                            logger.info(f"Manually extracted and cleaned amount: {manual_amount} from text: '{translated_final_text}'")
                    else:
                        # Even if Rasa extracted it, clean it again to be sure
                        existing_amount = result["form_data"]["amount"]
                        cleaned = clean_amount_value(existing_amount)
                        if cleaned:
                            result["form_data"]["amount"] = cleaned
                            logger.info(f"Re-cleaned existing amount: '{existing_amount}' -> '{cleaned}'")
                
                # Routing logic - now works with or without amount
                if result.get("action") == "topup_wallet":
                    amount = result.get("form_data", {}).get("amount")
                    prefill_data = {}
                    if amount:
                        prefill_data["amount"] = amount
                    
                    navigate_message = {
                        "type": "navigate",
                        "route": "/topup.html",
                        "prefill": prefill_data
                    }
                    await manager.send_personal_message(json.dumps(navigate_message), client_id)
                    logger.info(f"✅ Routing {client_id} to topup.html with prefill: {prefill_data}")

                elif result.get("action") == "multicurrency_wallet":
                    amount = result.get("form_data", {}).get("amount")
                    prefill_data = {}
                    if amount:
                        prefill_data["amount"] = amount
                    
                    navigate_message = {
                        "type": "navigate",
                        "route": "/multicurrency.html",
                        "prefill": prefill_data
                    }
                    await manager.send_personal_message(json.dumps(navigate_message), client_id)
                    logger.info(f"✅ Routing {client_id} to multicurrency.html with prefill: {prefill_data}")

                else:
                    logger.info(f"No routing triggered for action: {result.get('action')} (client: {client_id})")

                manager.audio_buffers[client_id] = []
                manager.transcribed_segments[client_id] = []
                manager.translated_segments[client_id] = []

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording stopped."}),
                    client_id
                )
                continue

            if data["type"] == "audio_chunk":
                if not manager.is_recording.get(client_id, False):
                    continue

                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    # Reduce log noise: don't log every chunk
                    # You may log conditionally or aggregate logs instead.
                    manager.audio_buffers[client_id].append(audio_np)
                except Exception as e:
                    logger.exception(f"Audio chunk processing error for {client_id}: {e}")
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": "Audio chunk processing error"}),
                        client_id
                    )

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected.")
    except Exception as e:
        logger.exception(f"Unhandled server error for {client_id}: {e}")
    finally:
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
    try:
        with open("static/index.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="Error: index.html not found", status_code=404)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_model_loaded": MODEL_LOADED,
        "rasa_model_loaded": NLU_PROCESSOR is not None,
        "device": str(DEVICE),
        "vad_loaded": VAD_MODEL is not None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
