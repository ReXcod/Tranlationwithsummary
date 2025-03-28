import streamlit as st
import speech_recognition as sr
from langdetect import detect
from googletrans import Translator
from elevenlabs import generate
from gtts import gTTS
from pydub import AudioSegment
import os
import requests
from io import BytesIO
import base64
import textwrap
import random
import warnings
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Suppress SyntaxWarning from pydub
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub.utils")

# Supported languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Latin": "la",
    "Greek": "el",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-cn",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt"
}

# Function to get all available voices from ElevenLabs
@st.cache_data
def get_available_voices(api_key):
    try:
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": api_key})
        voices = response.json().get("voices", [])
        if voices:
            return [voice["voice_id"] for voice in voices]
        return ["Rachel"]
    except Exception:
        return ["Rachel"]

# Function to convert audio or video file to WAV
def convert_to_wav(audio_file, file_extension):
    try:
        # Handle both audio and video files (extracts audio from video)
        audio = AudioSegment.from_file(audio_file, format=file_extension)
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io.read()
    except Exception as e:
        st.error(f"Failed to convert file to WAV: {str(e)}")
        return None

# Cached function to convert audio file to text
@st.cache_data
def audio_to_text(audio_file_content, file_extension):
    recognizer = sr.Recognizer()
    if file_extension != "wav":
        audio_file_content = convert_to_wav(BytesIO(audio_file_content), file_extension)
        if audio_file_content is None:
            return "Conversion to WAV failed"
    with sr.AudioFile(BytesIO(audio_file_content)) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Could not request results; check your internet connection"

# Cached function to detect language
@st.cache_data
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"

# Cached function to translate text
@st.cache_data
def translate_text(text, dest_lang):
    translator = Translator()
    chunks = textwrap.wrap(text, 500)
    translated_chunks = [translator.translate(chunk, dest=dest_lang).text for chunk in chunks]
    return " ".join(translated_chunks)

# Function to summarize text
@st.cache_data
def summarize_text(text, language="english", sentence_count=2):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

# Function to convert text to audio with ElevenLabs
def text_to_audio_elevenlabs(text, lang, voices):
    try:
        voice = random.choice(voices)
        api_key = os.getenv("ELEVENLABS_API_KEY") or "sk_b92f5590f2870ebf5b9ee5f14d0f895007087eaad06a218e"
        audio = generate(
            text=text,
            voice=voice,
            model="eleven_multilingual_v2",
            api_key=api_key
        )
        audio_file = BytesIO(audio)
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        audio_html = f'<audio controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        return audio_html, voice, True
    except Exception as e:
        st.warning(f"ElevenLabs failed: {str(e)}. Switching to gTTS.")
        return None, None, False

# Function to convert text to audio with gTTS
def text_to_audio_gtts(text, lang):
    try:
        tld = 'co.in' if lang in ['hi', 'mr'] else 'com'
        tts = gTTS(text=text, lang=lang, slow=False, tld=tld)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        audio_html = f'<audio controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        return audio_html, "gTTS", True
    except Exception as e:
        return f"Error: gTTS also failed: {str(e)}", None, False

# Streamlit app
st.title("Language Translator with Audio/Video Support")
st.write("Upload an audio or video file (WAV, MP3, AAC, MKV, MP4, etc.), choose options, and get translated audio!")

# Toggle for TTS engine preference
tts_preference = st.radio("Text-to-Speech Engine", ("ElevenLabs (Random Voice)", "gTTS (Normal Voice)"), index=0)

# Option to include summary
include_summary = st.checkbox("Include Summary of Translated Text", value=False)

# Get available voices for ElevenLabs
api_key = os.getenv("ELEVENLABS_API_KEY") or "sk_b92f5590f2870ebf5b9ee5f14d0f895007087eaad06a218e"
available_voices = get_available_voices(api_key)
if tts_preference == "ElevenLabs (Random Voice)":
    st.write(f"Available ElevenLabs Voices: {', '.join(available_voices)}")

# Option to auto-detect or choose input language
input_mode = st.radio("Input Language Mode", ("Auto-Detect", "Manual Selection"))
input_lang_code = "auto"
if input_mode == "Manual Selection":
    input_lang_name = st.selectbox("Select Input Language", list(LANGUAGES.keys()))
    input_lang_code = LANGUAGES[input_lang_name]

# File uploader for audio and video input
upload_col, _ = st.columns(2)
with upload_col:
    uploaded_file = st.file_uploader("Choose an audio/video file (WAV, MP3, AAC, MKV, MP4, etc.)", 
                                    type=["wav", "mp3", "aac", "mkv", "mp4", "ogg", "avi"])

# Language selection for output
output_lang_name = st.selectbox("Select Output Language", list(LANGUAGES.keys()))
output_lang_code = LANGUAGES[output_lang_name]

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    audio_content = uploaded_file.read()

    with st.spinner("Processing audio/video..."):
        input_text = audio_to_text(audio_content, file_extension)
    st.write("Recognized Text:", input_text)

    if input_text not in ["Could not understand the audio", "Could not request results; check your internet connection", "Conversion to WAV failed"]:
        if input_mode == "Auto-Detect":
            detected_lang = detect_language(input_text)
            st.write(f"Detected Input Language Code: {detected_lang}")
            if detected_lang == "unknown":
                st.error("Could not detect input language.")
                st.stop()
            input_lang_code = detected_lang
        else:
            st.write(f"Selected Input Language: {input_lang_name} ({input_lang_code})")

        with st.spinner("Translating..."):
            translated_text = translate_text(input_text, output_lang_code)
        st.write(f"Translated Text ({output_lang_name}):", translated_text)

        # Summary option
        if include_summary:
            summary = summarize_text(translated_text, language=output_lang_name.lower())
            st.write(f"Summary ({output_lang_name}):", summary)

        with st.spinner("Generating audio..."):
            if tts_preference == "ElevenLabs (Random Voice)":
                audio_output, used_voice, success = text_to_audio_elevenlabs(translated_text, output_lang_code, available_voices)
                if not success:
                    audio_output, used_voice, gtts_success = text_to_audio_gtts(translated_text, output_lang_code)
                    if not gtts_success:
                        st.error("Natural voice not working: Both ElevenLabs and gTTS failed.")
            else:
                audio_output, used_voice, gtts_success = text_to_audio_gtts(translated_text, output_lang_code)
                if not gtts_success:
                    st.error("Normal voice not working: gTTS failed.")
            
            st.write(f"{output_lang_name} Audio Output (Voice: {used_voice}):")
            st.markdown(audio_output, unsafe_allow_html=True)
    else:
        st.error("Audio/video processing failed.")

st.write("Note: Supports WAV, MP3, AAC, MKV, MP4, and more. Extracts audio from video. Uses ElevenLabs with gTTS fallback.")
