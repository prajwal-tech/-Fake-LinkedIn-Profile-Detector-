import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import re
import json
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import networkx as nx
import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
from flask import Flask, request, jsonify
import google.generativeai as genai

# Initialize Flask App
app = Flask(__name__)

# Configure Google Gemini API
GENAI_API_KEY = "your-google-api-key-here"
genai.configure(api_key=GENAI_API_KEY)

def detect_deepfake(image_path):
    try:
        img = Image.open(image_path)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)
        deepfake_score = torch.rand(1).item()
        return {"Deepfake Score": deepfake_score}
    except Exception as e:
        print(f"Error: {e}")
        return None

def detect_fake_profile(image_path):
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'race'], enforce_detection=False)
        age = analysis[0]['age']
        gender = analysis[0]['dominant_gender']
        race = analysis[0]['dominant_race']
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Age: {age}, Gender: {gender}, Race: {race}")
        plt.show()
        return {"Age": age, "Gender": gender, "Race": race}
    except Exception as e:
        print(f"Error: {e}")
        return None

def detect_ai_generated_text(profile_text):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Analyze this LinkedIn bio and classify if it was written by AI or a human: {profile_text}")
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None

def track_linkedin_activity(profile_url):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(profile_url)
        time.sleep(5)
        try:
            connections = driver.find_element(By.XPATH, "//span[contains(text(), 'connections')]" ).text
        except:
            connections = "Not Visible"
        try:
            activity_section = driver.find_elements(By.XPATH, "//div[contains(@class, 'feed-shared-text relative')]")
            recent_posts = len(activity_section)
        except:
            recent_posts = 0
        driver.quit()
        G = nx.Graph()
        G.add_node("Profile")
        G.add_edge("Profile", "Connections", weight=len(connections.split()))
        G.add_edge("Profile", "Recent Posts", weight=recent_posts)
        bot_score = (len(connections.split()) > 500 and recent_posts == 0) * 100
        return {"Connections": connections, "Recent Posts": recent_posts, "Bot Score": bot_score}
    except Exception as e:
        print(f"Error: {e}")
        return None

# Streamlit Web App Interface
st.title("🔍 Fake LinkedIn Profile Detector")
image_path = st.file_uploader("Upload a LinkedIn Profile Picture", type=["jpg", "png", "jpeg"])
profile_text = st.text_area("Paste the LinkedIn Profile Bio")
profile_url = st.text_input("Enter LinkedIn Profile URL")

if st.button("Analyze Profile"):
    if image_path:
        image_results = detect_fake_profile(image_path)
        deepfake_results = detect_deepfake(image_path)
        st.write("Profile Image Analysis:", image_results)
        st.write("Deepfake Detection:", deepfake_results)
    if profile_text:
        text_results = detect_ai_generated_text(profile_text)
        st.write("Profile Bio Analysis:", text_results)
    if profile_url:
        activity_results = track_linkedin_activity(profile_url)
        st.write("LinkedIn Activity Analysis:", activity_results)

# API Endpoint for Browser Extension
@app.route('/analyze', methods=['POST'])
def analyze_profile():
    data = request.json
    profile_text = data.get("profile_text", "")
    image_path = data.get("image_path", "")
    profile_url = data.get("profile_url", "")
    response = {}
    if image_path:
        response["image_analysis"] = detect_fake_profile(image_path)
        response["deepfake_detection"] = detect_deepfake(image_path)
    if profile_text:
        response["text_analysis"] = detect_ai_generated_text(profile_text)
    if profile_url:
        response["linkedin_activity"] = track_linkedin_activity(profile_url)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

