import os
import re

import pyaudio
import speech_recognition as sr
from flask import Flask, jsonify, render_template, request
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

HUGGINGFACEHUB_API_TOKEN = "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"

# ฟังก์ชันสำหรับการฟังเสียง
def listen_to_audio():
    recognizer = sr.Recognizer()
    mic_index = 1  # ใช้ไมค์ตัวแรกในรายการ

    with sr.Microphone(device_index=mic_index) as source:
        print("กำลังฟังเสียง...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="th-TH")
        print("คุณพูดว่า: " + text)
        return text
    except sr.UnknownValueError:
        return "ไม่สามารถเข้าใจเสียง"
    except sr.RequestError:
        return "ไม่สามารถเชื่อมต่อกับบริการ"

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    spoken_text = listen_to_audio()  # ฟังเสียงและแปลงเป็นข้อความ
    analysis = analyze_text(spoken_text)  # วิเคราะห์ข้อความ
    
    # คืนค่าข้อความที่พูดและผลการวิเคราะห์ในรูปแบบ JSON
    return jsonify({'spoken_text': spoken_text, 'result': analysis})

if __name__ == '__main__':
    app.run(debug=True)
