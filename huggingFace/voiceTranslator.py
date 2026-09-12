import torch
import sounddevice as sd
import scipy.io.wavfile as wav
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_LANGUAGE = "english"
TARGET_LANGUAGE = "hindi"

AUDIO_FILE = "input.wav"
OUTPUT_FILE = "translated.wav"

RECORD_SECONDS = 5
SAMPLE_RATE = 16000


# --------------------------------------------------
# 1. RECORD VOICE
# --------------------------------------------------

def record_audio():

    print("🎤 Speak now...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    wav.write(
        AUDIO_FILE,
        SAMPLE_RATE,
        audio
    )

    print("✅ Recording finished")


# --------------------------------------------------
# 2. SPEECH → TEXT
# --------------------------------------------------

def speech_to_text():

    print("🧠 Converting speech to text...")

    whisper = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )

    result = whisper(
        AUDIO_FILE,
        generate_kwargs={
            "language": "english"
        }
    )

    text = result["text"]

    print("\n📝 Original Text:")
    print(text)

    return text


# --------------------------------------------------
# 3. TEXT → TRANSLATED TEXT
# --------------------------------------------------

def translate_text(text):

    print("\n🌐 Translating...")

    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    # English → Hindi
    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=
        tokenizer.convert_tokens_to_ids("hin_Deva")
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    print("\n🌐 Translated Text:")
    print(translated_text)

    return translated_text


# --------------------------------------------------
# 4. TEXT → SPEECH
# --------------------------------------------------

def text_to_speech(text):

    print("\n🔊 Generating translated voice...")

    tts = pipeline(
        "text-to-speech",
        model="facebook/mms-tts-hin"
    )

    output = tts(text)

    audio = output["audio"]
    sampling_rate = output["sampling_rate"]

    wav.write(
        OUTPUT_FILE,
        sampling_rate,
        audio
    )

    print(
        f"✅ Translated audio saved as {OUTPUT_FILE}"
    )


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    # Voice input
    record_audio()

    # Speech → Text
    text = speech_to_text()

    # Translation
    translated_text = translate_text(text)

    # Text → Speech
    text_to_speech(translated_text)

    print("\n🎉 Translation completed!")


if __name__ == "__main__":
    main()