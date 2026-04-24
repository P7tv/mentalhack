import os
from dotenv import load_dotenv

import pyaudio
import speech_recognition as sr
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)
CORS(app)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
MIC_INDEX = int(os.getenv("MIC_INDEX", "1"))

def listen_to_audio():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone(device_index=MIC_INDEX) as source:
            audio = recognizer.listen(source, timeout=10)

        text = recognizer.recognize_google(audio, language="th-TH")
        return text
    except sr.UnknownValueError:
        return "ไม่สามารถเข้าใจเสียง"
    except sr.RequestError as e:
        return f"ไม่สามารถเชื่อมต่อกับบริการ: {str(e)}"
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"

# ฟังก์ชันสำหรับวิเคราะห์ข้อความ
def analyze_text(text):
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-1.1-2b-it",
        max_length=512,
        temperature=0.5,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    template = """Question: {question}

    วิเคราะห์สุขภาพจิตจากอารมณ์จากข้อความข้างต้นและบอกว่าปัญหาเกิดจากอะไรและคิดว่าอยู่ในสภาวะอะไรของ mental health ตอบกลับมาเป็นบทคำพูด"""

    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.invoke(text)

    return output['text']

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        spoken_text = listen_to_audio()
        analysis = analyze_text(spoken_text)
        return jsonify({'spoken_text': spoken_text, 'result': analysis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
