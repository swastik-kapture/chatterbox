from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTTS
from scipy.io.wavfile import write
import io
import re
import gc
import torch

app = FastAPI(
    title="ChatterBox TTS",
    summary="Generate speech using ChatterBox TTS.",
    version="0.1",
    redoc_url=f"/v1/chatterbox/redoc",
    docs_url=f"/v1/chatterbox/docs",
    openapi_url=f"/v1/chatterbox/openapi.json",
    swagger_ui_parameters={"displayRequestDuration":True,"displayOperationId":True},
)

templates = Jinja2Templates(directory="templates")

available_voices = ["aysha_0001"]

model = ChatterboxTTS.from_pretrained(device="cuda")
model.prepare_conditionals("aysha_0001.wav")
sample_rate = 24000


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "aysha_0001"
    response_format: str = "wav"
    stream: bool = False


def audio_chunk_to_wav_bytes(audio_chunk: torch.Tensor) -> bytes:
    audio_np = audio_chunk.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype("int16")
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_int16)
    return buffer.getvalue()


def audio_chunk_to_pcm_bytes(audio_chunk: torch.Tensor) -> bytes:
    audio_np = audio_chunk.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype("int16")
    return audio_int16.tobytes()

def split_into_sentences(text: str) -> list[str]:
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    return sentence_endings.split(text.strip())

@app.post("/v1/chatterbox/speech")
async def generate_speech(req: SpeechRequest):
    text = req.input
    stream = req.stream

    if req.response_format != "wav":
        return Response(status_code=400, content="Only 'wav' format is supported.")

    if stream:
        async def audio_stream():
            try:
                sentence_count = 0
                for sentence in split_into_sentences(text):
                    if not sentence.strip():
                        continue
                    sentence_count += 1
                    print(f"Synthesizing sentence: {sentence}")
                    for chunk, _ in model.generate_stream(sentence, temperature=0.4, chunk_size=24):
                        wav_bytes = audio_chunk_to_pcm_bytes(chunk)
                        yield wav_bytes
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        return StreamingResponse(audio_stream(), media_type="application/octet-stream", headers={
            "Content-Type": "audio/wav",
            "Transfer-Encoding": "chunked",
        })
    else:
        try:
            chunks = [chunk for chunk, _ in model.generate_stream(text)]
            full_audio = torch.cat(chunks, dim=-1)
            wav_bytes = audio_chunk_to_wav_bytes(full_audio)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return Response(content=wav_bytes, media_type="audio/wav", headers={
            "Content-Type": "audio/wav",
            "Content-Disposition": "attachment; filename=speech.wav"
        })


@app.get("/v1/chatterbox/ui", response_class=HTMLResponse)
async def frontend(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "voices": available_voices
    })