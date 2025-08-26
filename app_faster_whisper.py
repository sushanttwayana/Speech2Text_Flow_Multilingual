import asyncio
import warnings
import json
import base64
import logging
import numpy as np
import torch
import os
from datetime import datetime
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager

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

# Global variables
DEVICE = None
SAMPLE_RATE = 16000  # standard for Whisper
MODEL_LOADED = False
ASR_MODEL: WhisperModel = None
NLU_PROCESSOR = None
VAD_MODEL = None
GET_SPEECH_TIMESTAMPS = None
COLLECT_CHUNKS = None

client_languages: Dict[str, str] = {}

def setup_device():
    """Setup device with proper CUDA error handling"""
    global DEVICE
    
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            DEVICE = torch.device('cuda')
            logger.info("✅ CUDA is available and working")
            return "cuda", "float16"
        except Exception as e:
            logger.warning(f"⚠️ CUDA available but not working properly: {e}")
            logger.info("Falling back to CPU")
            DEVICE = torch.device('cpu')
            return "cpu", "float32"
    else:
        logger.info("CUDA not available, using CPU")
        DEVICE = torch.device('cpu')
        return "cpu", "float32"

# -------- RASA NLU Processor Wrapper --------
class NLUProcessor:
    def __init__(self, agent: Agent):
        self.agent = agent
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
                        form_data["amount"] = entity["value"]
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

# -------- Argos Translate Setup --------
def install_local_translation_models(model_dir: str = "translation_models"):
    if not os.path.isdir(model_dir):
        logger.error(f"Translation model directory {model_dir} not found.")
        return
    logger.info(f"Installing Argos Translate models from '{model_dir}' ...")
    for filename in os.listdir(model_dir):
        if filename.endswith(".argosmodel"):
            path = os.path.join(model_dir, filename)
            try:
                argostranslate.package.install_from_path(path)
                logger.info(f"Installed translation model: {filename}")
            except Exception as e:
                logger.error(f"Failed to install {filename}: {e}")

def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == "en":
        return text
    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((l for l in installed_languages if l.code == source_lang), None)
        to_lang = next((l for l in installed_languages if l.code == "en"), None)
        if from_lang and to_lang:
            translation = from_lang.get_translation(to_lang)
            translated_text = translation.translate(text)
            logger.info(f"Translated from [{source_lang}] to English.")
            return translated_text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
    
    logger.warning(f"No translation model from {source_lang} to English found. Returning original text.")
    return text

# -------- ASR using Faster Whisper --------
async def transcribe_audio_faster_whisper(audio_np: np.ndarray, language: Optional[str] = None):
    global ASR_MODEL
    try:
        beam_size = 5
        segments, info = ASR_MODEL.transcribe(
            audio_np,
            beam_size=beam_size,
            language=language,
            condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text for segment in segments)
        detected_lang = info.language
        logger.info(f"ASR detected language: {detected_lang} (prob: {info.language_probability:.2f})")
        return transcript.strip(), detected_lang
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return "", "en"

# -------- VAD Loading with Error Handling --------
def load_silero_vad():
    try:
        import torch.hub
        # Force CPU for VAD to avoid CUDA issues
        vad_bundle = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad',
            force_reload=False
        )
        return vad_bundle
    except Exception as e:
        logger.error(f"Error loading Silero VAD: {e}")
        return None, None

# Local directory to store model files (relative to your app.py)
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "faster-whisper-large-v3")

# -------- Lifespan to load all models --------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED, ASR_MODEL, NLU_PROCESSOR, VAD_MODEL, GET_SPEECH_TIMESTAMPS, COLLECT_CHUNKS, DEVICE

    try:
        # Configure TensorFlow GPU settings early with error handling
        try:
            import tensorflow as tf
            # Disable GPU for TensorFlow to avoid conflicts
            tf.config.set_visible_devices([], 'GPU')
            logger.info("✅ TensorFlow configured to use CPU only")
        except Exception as e:
            logger.warning(f"TensorFlow GPU configuration warning: {e}")

        # Setup device with proper error handling
        device_type, compute_type = setup_device()
        logger.info(f"Device set to: {DEVICE}")

        # Check if model folder exists
        if not os.path.exists(LOCAL_MODEL_DIR):
            logger.info(f"Model folder not found, downloading faster-whisper-large-v3 to {LOCAL_MODEL_DIR}")
            snapshot_download(
                repo_id="Systran/faster-whisper-large-v3",
                local_dir=LOCAL_MODEL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            logger.info("Model download completed.")
        else:
            logger.info(f"Model folder exists at {LOCAL_MODEL_DIR}, using existing files")

        # Load Faster Whisper model with error handling
        try:
            ASR_MODEL = WhisperModel(
                LOCAL_MODEL_DIR,
                device=device_type,
                compute_type=compute_type,
                # Add these parameters to help with CUDA issues
                cpu_threads=4 if device_type == "cpu" else 0,
                num_workers=1
            )
            MODEL_LOADED = True
            logger.info(f"✅ Loaded Faster Whisper model on {device_type} with {compute_type}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to CPU
            if device_type == "cuda":
                logger.info("Retrying with CPU...")
                DEVICE = torch.device('cpu')
                ASR_MODEL = WhisperModel(
                    LOCAL_MODEL_DIR,
                    device="cpu",
                    compute_type="float32",
                    cpu_threads=4,
                    num_workers=1
                )
                MODEL_LOADED = True
                logger.info("✅ Loaded Faster Whisper model on CPU (fallback)")

        # Load Argos Translate models
        try:
            install_local_translation_models("translation_models")
            logger.info("✅ Argos Translate models loaded.")
        except Exception as e:
            logger.error(f"Failed to load translation models: {e}")

        # Load Silero VAD with error handling
        logger.info("Loading Silero VAD...")
        try:
            vad_result = load_silero_vad()
            if vad_result and len(vad_result) == 2:
                VAD_MODEL, vad_utils = vad_result
                if vad_utils and len(vad_utils) >= 5:
                    GET_SPEECH_TIMESTAMPS, _, _, _, COLLECT_CHUNKS = vad_utils
                    logger.info("✅ Silero VAD loaded successfully.")
                else:
                    logger.error("VAD utils not properly loaded")
                    VAD_MODEL = None
            else:
                logger.error("Failed to load VAD model")
                VAD_MODEL = None
        except Exception as e:
            logger.error(f"Error loading Silero VAD: {e}")
            VAD_MODEL = None

        # Load Rasa model with error handling
        logger.info("Loading Rasa NLU model...")
        try:
            NLU_MODEL_PATH = "models/nlu_two_intent_classifier.tar.gz"
            NLU_PROCESSOR = NLUProcessor.create(NLU_MODEL_PATH)
            if NLU_PROCESSOR:
                logger.info("✅ Rasa NLUProcessor initialized successfully.")
            else:
                logger.error("Failed to initialize Rasa NLUProcessor.")
        except Exception as e:
            logger.error(f"Error loading Rasa model: {e}")
            NLU_PROCESSOR = None

        yield

    except Exception as e:
        logger.exception(f"Failed to initialize models: {e}")
        MODEL_LOADED = False
        NLU_PROCESSOR = None
        yield
    finally:
        logger.info("Shutting down...")
        # Cleanup CUDA memory if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

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

# -------- ConnectionManager and websocket logic --------
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
            try:
                await self.active_connections[client_id].close()
            except:
                pass
        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.is_recording[client_id] = False
        self.transcribed_segments[client_id] = []
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

async def transcribe_segment(audio_np_segment: np.ndarray, language: str) -> str:
    if not MODEL_LOADED or ASR_MODEL is None:
        logger.error("Model not loaded, cannot transcribe.")
        return ""

    try:
        # Apply VAD to filter silence if available
        if VAD_MODEL and GET_SPEECH_TIMESTAMPS and COLLECT_CHUNKS:
            try:
                audio_torch = torch.from_numpy(audio_np_segment).float()
                # Force VAD to run on CPU to avoid CUDA issues
                if audio_torch.is_cuda:
                    audio_torch = audio_torch.cpu()
                    
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
            except Exception as e:
                logger.warning(f"VAD processing failed, using original audio: {e}")
                filtered_audio_np = audio_np_segment
        else:
            filtered_audio_np = audio_np_segment

        # Transcribe with Faster Whisper
        transcript, detected_lang = await transcribe_audio_faster_whisper(filtered_audio_np, language)

        # Translate to English if needed
        translated_transcript = translate_to_english(transcript, detected_lang)
        return translated_transcript
        
    except Exception as e:
        logger.exception(f"Error in transcribing segment: {e}")
        return ""

async def process_audio_for_client(client_id: str, language: str):
    CHUNK_SECONDS = 15
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
    audio_buffer = np.array([], dtype=np.float32)

    while manager.is_recording.get(client_id, False):
        try:
            await asyncio.sleep(0.2)
            
            # Process audio chunks
            while manager.audio_buffers[client_id]:
                try:
                    new_chunk = manager.audio_buffers[client_id].pop(0)
                    audio_buffer = np.concatenate([audio_buffer, new_chunk])
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    continue
                    
            # Process when buffer is large enough
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
        except Exception as e:
            logger.exception(f"Interim processing error for {client_id}: {e}")
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": "Interim processing error"}),
                client_id
            )
            break

    # Process remaining buffer
    if len(audio_buffer) > 0:
        try:
            transcription = await transcribe_segment(audio_buffer, language)
            if transcription:
                await manager.send_personal_message(
                    json.dumps({"type": "transcription", "text": transcription, "is_final": False}),
                    client_id
                )
                manager.transcribed_segments[client_id].append(transcription)
        except Exception as e:
            logger.exception(f"Error processing final chunk for {client_id}: {e}")

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
                    try:
                        final_audio = np.concatenate(manager.audio_buffers[client_id])
                        logger.info(f"Processing remaining audio buffer for {client_id}, size: {len(final_audio)} samples")
                        manager.audio_buffers[client_id] = []
                    except Exception as e:
                        logger.error(f"Error concatenating final audio: {e}")

                final_segments = manager.transcribed_segments.get(client_id, [])
                if final_audio.size > 0:
                    transcription = await transcribe_segment(final_audio, client_languages.get(client_id, "en"))
                    if transcription:
                        final_segments.append(transcription)
                        logger.info(f"Appended final transcription for {client_id}: {transcription}")

                final_text = merge_transcriptions(final_segments) or "No speech detected."
                logger.info(f"Final transcription for {client_id}: {final_text}")

                # Initialize result
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

                # Navigation logic
                if result.get("action") == "topup_wallet" and result.get("form_data", {}).get("amount"):
                    amount = result["form_data"]["amount"]
                    navigate_message = {
                        "type": "navigate",
                        "route": "/topup.html",
                        "prefill": {"amount": amount}
                    }
                    await manager.send_personal_message(json.dumps(navigate_message), client_id)
                     
                elif result.get("action") == "multicurrency_wallet" and result.get("form_data", {}).get("amount"):
                    amount = result["form_data"]["amount"]
                    navigate_message = {
                        "type": "navigate",
                        "route": "/multicurrency.html",
                        "prefill": {"amount": amount}
                    }
                    await manager.send_personal_message(json.dumps(navigate_message), client_id)

                manager.audio_buffers[client_id] = []
                manager.transcribed_segments[client_id] = []

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
                    logger.info(f"Received audio chunk for {client_id}, size: {len(audio_np)} samples")
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
    uvicorn.run(app, host="0.0.0.0", port=8001)