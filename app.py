from flask import Flask, request, jsonify, render_template_string
import csv
import json
import os
import re
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Groq API configuration from environment variables
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def analyze_with_groq(transcript):
    """Enhanced Groq analysis with better error handling"""
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY not found in environment variables. Please check your .env file."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes transcripts. Always respond with valid JSON containing exactly two fields: 'summary' and 'sentiment'."
            },
            {
                "role": "user", 
                "content": f"""Analyze this transcript and return ONLY valid JSON with summary and sentiment fields:

Transcript: {transcript}

Required JSON format:
{{"summary": "your summary here", "sentiment": "positive/negative/neutral"}}"""
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    try:
        print("Making Groq API request...")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        if not response.ok:
            error_text = response.text
            print(f"API Error Response: {error_text}")
            return None, f"Groq API error ({response.status_code}): {error_text}"
        
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"].strip()
        print(f"Raw Groq response: {content}")
        
        # Enhanced JSON parsing with multiple fallback methods
        result = None
        
        # Method 1: Direct JSON parsing
        try:
            result = json.loads(content)
            print("Direct JSON parsing successful")
        except json.JSONDecodeError:
            pass
        
        # Method 2: Remove markdown if present
        if not result:
            try:
                clean_content = content
                if content.startswith("```"):
                    clean_content = content[7:-3].strip()
                elif content.startswith("```"):
                    clean_content = content[3:-3].strip()
                
                result = json.loads(clean_content)
                print("Markdown removal parsing successful")
            except json.JSONDecodeError:
                pass
        
        # Method 3: Extract JSON using regex
        if not result:
            try:
                match = re.search(r'\{[^}]*"summary"[^}]*"sentiment"[^}]*\}', content, re.DOTALL)
                if not match:
                    match = re.search(r'\{.*?\}', content, re.DOTALL)
                
                if match:
                    json_str = match.group(0)
                    result = json.loads(json_str)
                    print("Regex extraction parsing successful")
            except json.JSONDecodeError:
                pass
        
        if not result:
            return None, f"Could not parse JSON from response: {content}"
        
        # Validate required fields
        if "summary" not in result or "sentiment" not in result:
            return None, f"Missing required fields in response: {result}"
        
        # Validate and fix sentiment
        valid_sentiments = ["positive", "negative", "neutral"]
        if result["sentiment"].lower() not in valid_sentiments:
            print(f"Invalid sentiment '{result['sentiment']}', defaulting to neutral")
            result["sentiment"] = "neutral"
        
        result["sentiment"] = result["sentiment"].lower()
        
        print(f"Analysis completed successfully: {result}")
        return result, None
        
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def append_to_csv(transcript, analysis_result):
    """Append analysis result to CSV file"""
    csv_filename = 'call_analysis.csv'
    
    try:
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'transcript', 'summary', 'sentiment'])
            print(f"Created new CSV file: {csv_filename}")
        
        with open(csv_filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(),
                transcript,
                analysis_result['summary'],
                analysis_result['sentiment']
            ])
        
        print(f"Successfully saved to {csv_filename}")
        
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Advanced animated UI home page"""
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>AI Transcript Analyzer Pro | Advanced Analytics Platform</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
        <style>
            :root {
                --primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                --danger: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                --warning: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                --dark: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                --glass: rgba(255, 255, 255, 0.25);
                --glass-border: rgba(255, 255, 255, 0.18);
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
                min-height: 100vh;
                overflow-x: hidden;
            }
            
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            .glass-morphism {
                background: var(--glass);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid var(--glass-border);
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            }
            
            .hero-section {
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: hidden;
            }
            
            .floating-elements {
                position: absolute;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: 1;
            }
            
            .floating-element {
                position: absolute;
                background: var(--glass);
                backdrop-filter: blur(10px);
                border-radius: 50%;
                animation: float 6s ease-in-out infinite;
            }
            
            .floating-element:nth-child(1) {
                width: 80px;
                height: 80px;
                top: 10%;
                left: 10%;
                animation-delay: 0s;
            }
            
            .floating-element:nth-child(2) {
                width: 120px;
                height: 120px;
                top: 20%;
                right: 10%;
                animation-delay: 2s;
            }
            
            .floating-element:nth-child(3) {
                width: 60px;
                height: 60px;
                bottom: 30%;
                left: 15%;
                animation-delay: 4s;
            }
            
            .floating-element:nth-child(4) {
                width: 100px;
                height: 100px;
                bottom: 20%;
                right: 20%;
                animation-delay: 1s;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-20px) rotate(180deg); }
            }
            
            .main-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                position: relative;
                z-index: 10;
            }
            
            .hero-card {
                border-radius: 30px;
                border: none;
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
                transition: all 0.3s ease;
                overflow: hidden;
            }
            
            .hero-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 35px 70px rgba(0, 0, 0, 0.2);
            }
            
            .card-header {
                background: var(--primary);
                color: white;
                text-align: center;
                padding: 50px 30px;
                position: relative;
                overflow: hidden;
            }
            
            .card-header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
                transform: rotate(45deg);
                animation: shimmer 3s linear infinite;
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
                100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
            }
            
            .hero-title {
                font-size: 3.5rem;
                font-weight: 800;
                margin-bottom: 15px;
                text-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                animation: titlePulse 2s ease-in-out infinite alternate;
            }
            
            @keyframes titlePulse {
                0% { text-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); }
                100% { text-shadow: 0 4px 25px rgba(0, 0, 0, 0.4); }
            }
            
            .hero-subtitle {
                font-size: 1.3rem;
                opacity: 0.95;
                font-weight: 400;
                margin-bottom: 20px;
            }
            
            .feature-badges {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 15px;
                margin-top: 25px;
            }
            
            .feature-badge {
                background: var(--glass);
                backdrop-filter: blur(10px);
                border-radius: 25px;
                padding: 10px 20px;
                font-size: 0.9rem;
                font-weight: 600;
                border: 1px solid var(--glass-border);
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .feature-badge:hover {
                transform: scale(1.05) translateY(-2px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                background: rgba(255, 255, 255, 0.35);
            }
            
            .stats-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 25px;
                margin: 40px 0;
            }
            
            .stat-card {
                background: var(--glass);
                backdrop-filter: blur(15px);
                border-radius: 20px;
                padding: 30px;
                text-align: center;
                border: 1px solid var(--glass-border);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .stat-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .stat-card:hover::before {
                left: 100%;
            }
            
            .stat-card:hover {
                transform: translateY(-5px) scale(1.02);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
            }
            
            .stat-number {
                font-size: 2.5rem;
                font-weight: 800;
                background: var(--primary);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
                position: relative;
                z-index: 2;
            }
            
            .stat-label {
                color: #64748b;
                font-weight: 600;
                font-size: 1rem;
                position: relative;
                z-index: 2;
            }
            
            .form-container {
                background: var(--glass);
                backdrop-filter: blur(20px);
                border-radius: 25px;
                padding: 40px;
                border: 1px solid var(--glass-border);
                margin: 30px 0;
            }
            
            .form-label {
                font-weight: 600;
                color: #1e293b;
                margin-bottom: 15px;
                font-size: 1.1rem;
            }
            
            .form-control {
                border: 2px solid transparent;
                border-radius: 15px;
                padding: 18px 25px;
                font-size: 1rem;
                transition: all 0.3s ease;
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(10px);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                resize: vertical;
                min-height: 150px;
            }
            
            .form-control:focus {
                border: 2px solid;
                border-image: var(--primary) 1;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
                transform: translateY(-2px);
                outline: none;
            }
            
            .analyze-btn {
                background: var(--primary);
                border: none;
                border-radius: 15px;
                padding: 18px 40px;
                font-weight: 700;
                font-size: 1.2rem;
                color: white;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }
            
            .analyze-btn:hover:not(:disabled) {
                transform: translateY(-3px);
                box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
            }
            
            .analyze-btn:active {
                transform: translateY(0);
            }
            
            .analyze-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .analyze-btn:hover::before {
                left: 100%;
            }
            
            .loading-animation {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .result-card {
                background: var(--glass);
                backdrop-filter: blur(20px);
                border-radius: 25px;
                border: 1px solid var(--glass-border);
                margin-top: 30px;
                overflow: hidden;
                transform: translateY(20px);
                opacity: 0;
                transition: all 0.5s ease;
            }
            
            .result-card.show {
                transform: translateY(0);
                opacity: 1;
            }
            
            .result-header {
                padding: 25px 30px;
                border-bottom: 1px solid var(--glass-border);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .sentiment-badge {
                padding: 10px 20px;
                border-radius: 25px;
                font-weight: 700;
                font-size: 0.9rem;
                border: 2px solid transparent;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .sentiment-positive {
                background: var(--success);
                color: white;
                box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
            }
            
            .sentiment-negative {
                background: var(--danger);
                color: white;
                box-shadow: 0 8px 25px rgba(250, 112, 154, 0.3);
            }
            
            .sentiment-neutral {
                background: var(--warning);
                color: #8b5a00;
                box-shadow: 0 8px 25px rgba(255, 230, 210, 0.3);
            }
            
            .result-content {
                padding: 30px;
            }
            
            .summary-text {
                font-size: 1.1rem;
                line-height: 1.7;
                color: #334155;
                background: rgba(255, 255, 255, 0.6);
                padding: 20px 25px;
                border-radius: 15px;
                border-left: 5px solid transparent;
                border-image: var(--primary) 1;
                backdrop-filter: blur(5px);
            }
            
            .error-card {
                background: linear-gradient(135deg, #ff6b6b, #ee5a52);
                color: white;
                border-radius: 15px;
                padding: 20px;
                margin-top: 20px;
                box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
                animation: shake 0.5s ease-in-out;
            }
            
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-5px); }
                75% { transform: translateX(5px); }
            }
            
            .pulse {
                animation: pulse 2s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            .fade-in {
                animation: fadeIn 0.6s ease-out forwards;
            }
            
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .tech-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: var(--dark);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                margin: 5px;
                transition: all 0.3s ease;
            }
            
            .tech-badge:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            }
            
            .scroll-indicator {
                position: fixed;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                color: white;
                font-size: 1.5rem;
                animation: bounce 2s infinite;
                z-index: 1000;
            }
            
            @keyframes bounce {
                0%, 20%, 50%, 80%, 100% { transform: translateX(-50%) translateY(0); }
                40% { transform: translateX(-50%) translateY(-10px); }
                60% { transform: translateX(-50%) translateY(-5px); }
            }
            
            .env-status {
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--glass);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 8px 16px;
                border: 1px solid var(--glass-border);
                font-size: 0.8rem;
                font-weight: 600;
                z-index: 1000;
                color: #22c55e;
            }
            
            @media (max-width: 768px) {
                .hero-title { font-size: 2.5rem; }
                .hero-subtitle { font-size: 1.1rem; }
                .main-container { padding: 15px; }
                .form-container { padding: 25px 20px; }
                .stats-container { grid-template-columns: 1fr 1fr; gap: 15px; }
                .stat-card { padding: 20px; }
                .stat-number { font-size: 2rem; }
                .env-status { display: none; }
            }
        </style>
    </head>
    <body>
        <!-- Environment Status Indicator -->
        <div class="env-status">
            <i class="bi bi-shield-check me-2"></i>
            Environment Configured
        </div>
        
        <!-- Floating Background Elements -->
        <div class="floating-elements">
            <div class="floating-element"></div>
            <div class="floating-element"></div>
            <div class="floating-element"></div>
            <div class="floating-element"></div>
        </div>
        
        <div class="hero-section">
            <div class="main-container">
                <div class="hero-card card glass-morphism" data-aos="fade-up" data-aos-duration="1000">
                    <div class="card-header">
                        <h1 class="hero-title">
                            <i class="bi bi-cpu-fill me-3"></i>
                            AI Transcript Analyzer Pro
                        </h1>
                        <p class="hero-subtitle">
                            Next-generation AI-powered sentiment analysis and intelligent summarization platform
                        </p>
                        
                        <div class="feature-badges">
                            <div class="feature-badge" data-aos="zoom-in" data-aos-delay="200">
                                <i class="bi bi-lightning-charge-fill me-2"></i>
                                Real-time Processing
                            </div>
                            <div class="feature-badge" data-aos="zoom-in" data-aos-delay="400">
                                <i class="bi bi-shield-check me-2"></i>
                                Environment Variables
                            </div>
                            <div class="feature-badge" data-aos="zoom-in" data-aos-delay="600">
                                <i class="bi bi-graph-up me-2"></i>
                                Advanced Analytics
                            </div>
                            <div class="feature-badge" data-aos="zoom-in" data-aos-delay="800">
                                <i class="bi bi-cloud-arrow-down me-2"></i>
                                Auto Export
                            </div>
                        </div>
                    </div>
                    
                    <div class="card-body">
                        <!-- Statistics Dashboard -->
                        <div class="stats-container" data-aos="fade-up" data-aos-delay="200">
                            <div class="stat-card" data-aos="flip-left" data-aos-delay="300">
                                <div class="stat-number" id="totalAnalyses">0</div>
                                <div class="stat-label">Analyses Completed</div>
                            </div>
                            <div class="stat-card" data-aos="flip-left" data-aos-delay="500">
                                <div class="stat-number" id="avgWords">0</div>
                                <div class="stat-label">Avg. Word Count</div>
                            </div>
                            <div class="stat-card" data-aos="flip-left" data-aos-delay="700">
                                <div class="stat-number" id="successRate">100%</div>
                                <div class="stat-label">Success Rate</div>
                            </div>
                            <div class="stat-card" data-aos="flip-left" data-aos-delay="900">
                                <div class="stat-number" id="lastSentiment">-</div>
                                <div class="stat-label">Last Result</div>
                            </div>
                        </div>
                        
                        <!-- Analysis Form -->
                        <div class="form-container" data-aos="fade-up" data-aos-delay="400">
                            <form id="transcriptForm">
                                <div class="mb-4">
                                    <label for="transcript" class="form-label">
                                        <i class="bi bi-mic-fill me-2"></i>
                                        Enter Your Transcript
                                    </label>
                                    <textarea 
                                        class="form-control" 
                                        id="transcript" 
                                        name="transcript" 
                                        placeholder="Paste your conversation transcript, meeting notes, call recording text, customer feedback, or any text content you'd like to analyze for sentiment and generate an intelligent summary..."
                                        required
                                        data-aos="zoom-in" 
                                        data-aos-delay="600"
                                    ></textarea>
                                    <div class="form-text mt-3">
                                        <i class="bi bi-info-circle me-2"></i>
                                        <strong>Secure Environment Variables:</strong> API keys are safely stored in .env file for enhanced security. Results are automatically saved to CSV for your records.
                                    </div>
                                </div>
                                
                                <button type="submit" class="analyze-btn w-100" data-aos="zoom-in" data-aos-delay="800">
                                    <span class="btn-text">
                                        <i class="bi bi-rocket-takeoff-fill me-2"></i>
                                        Analyze with AI
                                    </span>
                                </button>
                            </form>
                            
                            <!-- Tech Stack Display -->
                            <div class="text-center mt-4" data-aos="fade-up" data-aos-delay="1000">
                                <div class="mb-3" style="color: #64748b; font-weight: 600;">Powered by:</div>
                                <div>
                                    <span class="tech-badge">
                                        <i class="bi bi-cpu"></i> Groq AI
                                    </span>
                                    <span class="tech-badge">
                                        <i class="bi bi-code-slash"></i> Python Flask
                                    </span>
                                    <span class="tech-badge">
                                        <i class="bi bi-file-earmark-code"></i> Environment Variables
                                    </span>
                                    <span class="tech-badge">
                                        <i class="bi bi-database"></i> Auto CSV Export
                                    </span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Results Container -->
                        <div id="resultContainer" style="display: none;">
                            <div class="result-card">
                                <div class="result-header">
                                    <h5 class="mb-0">
                                        <i class="bi bi-clipboard-data me-2"></i>
                                        Analysis Results
                                    </h5>
                                    <span id="sentimentBadge" class="sentiment-badge"></span>
                                </div>
                                <div class="result-content">
                                    <div class="mb-4">
                                        <h6 class="mb-3" style="color: #475569; font-weight: 600;">
                                            <i class="bi bi-file-text-fill me-2"></i>
                                            AI-Generated Summary
                                        </h6>
                                        <div id="summaryText" class="summary-text"></div>
                                    </div>
                                    
                                    <div class="row text-center" style="color: #64748b;">
                                        <div class="col-md-6">
                                            <small>
                                                <i class="bi bi-clock me-1"></i>
                                                Analyzed at <span id="timestamp"></span>
                                            </small>
                                        </div>
                                        <div class="col-md-6">
                                            <small>
                                                <i class="bi bi-check-circle-fill me-1"></i>
                                                Saved to call_analysis.csv
                                            </small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Error Container -->
                        <div id="errorContainer" style="display: none;">
                            <div class="error-card">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-exclamation-triangle-fill me-3" style="font-size: 1.5rem;"></i>
                                    <div>
                                        <div style="font-weight: 600; margin-bottom: 5px;">Analysis Failed</div>
                                        <div id="errorText"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Scroll Indicator -->
        <div class="scroll-indicator" id="scrollIndicator">
            <i class="bi bi-chevron-double-down"></i>
        </div>

        <!-- Scripts -->
        <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
        <script>
            // Initialize AOS animations
            AOS.init({
                duration: 800,
                easing: 'ease-in-out',
                once: false,
                mirror: true
            });
            
            // Global statistics
            let analysisCount = 0;
            let successCount = 0;
            let totalWords = 0;
            
            // Auto-hide scroll indicator after initial load
            setTimeout(() => {
                document.getElementById('scrollIndicator').style.display = 'none';
            }, 5000);
            
            // Form submission with advanced animations
            document.getElementById('transcriptForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                
                const transcript = document.getElementById('transcript').value.trim();
                if (!transcript) return;
                
                const button = this.querySelector('button');
                const btnText = button.querySelector('.btn-text');
                const resultContainer = document.getElementById('resultContainer');
                const errorContainer = document.getElementById('errorContainer');
                
                // Hide previous results with animation
                if (resultContainer.style.display !== 'none') {
                    resultContainer.querySelector('.result-card').classList.remove('show');
                    setTimeout(() => {
                        resultContainer.style.display = 'none';
                    }, 300);
                }
                errorContainer.style.display = 'none';
                
                // Animated loading state
                button.disabled = true;
                btnText.innerHTML = '<span class="loading-animation"></span> Processing with AI...';
                button.classList.add('pulse');
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ transcript })
                    });
                    
                    analysisCount++;
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.error || 'Analysis failed');
                    }
                    
                    const data = await response.json();
                    successCount++;
                    
                    // Update statistics with animation
                    const wordCount = transcript.split(/\s+/).length;
                    totalWords += wordCount;
                    
                    animateNumber('totalAnalyses', analysisCount);
                    animateNumber('avgWords', Math.round(totalWords / analysisCount));
                    animateNumber('successRate', Math.round((successCount / analysisCount) * 100), '%');
                    
                    const lastSentimentEl = document.getElementById('lastSentiment');
                    lastSentimentEl.textContent = data.sentiment.charAt(0).toUpperCase() + data.sentiment.slice(1);
                    lastSentimentEl.classList.add('pulse');
                    setTimeout(() => lastSentimentEl.classList.remove('pulse'), 2000);
                    
                    // Display results with animation
                    document.getElementById('summaryText').textContent = data.summary;
                    document.getElementById('timestamp').textContent = new Date().toLocaleString();
                    
                    const badge = document.getElementById('sentimentBadge');
                    
                    // Apply sentiment-specific styling with animation
                    if (data.sentiment.toLowerCase() === 'positive') {
                        badge.innerHTML = '<i class="bi bi-emoji-smile-fill me-2"></i>Positive';
                        badge.className = 'sentiment-badge sentiment-positive';
                    } else if (data.sentiment.toLowerCase() === 'negative') {
                        badge.innerHTML = '<i class="bi bi-emoji-frown-fill me-2"></i>Negative';
                        badge.className = 'sentiment-badge sentiment-negative';
                    } else {
                        badge.innerHTML = '<i class="bi bi-emoji-neutral-fill me-2"></i>Neutral';
                        badge.className = 'sentiment-badge sentiment-neutral';
                    }
                    
                    // Show results with smooth animation
                    resultContainer.style.display = 'block';
                    setTimeout(() => {
                        resultContainer.querySelector('.result-card').classList.add('show');
                        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 100);
                    
                } catch (error) {
                    document.getElementById('errorText').textContent = error.message;
                    errorContainer.style.display = 'block';
                    errorContainer.scrollIntoView({ behavior: 'smooth' });
                    
                    // Update success rate even for failures
                    animateNumber('successRate', Math.round((successCount / analysisCount) * 100), '%');
                }
                
                // Reset button with animation
                button.disabled = false;
                button.classList.remove('pulse');
                btnText.innerHTML = '<i class="bi bi-rocket-takeoff-fill me-2"></i>Analyze with AI';
            });
            
            // Animated number counter
            function animateNumber(elementId, targetValue, suffix = '') {
                const element = document.getElementById(elementId);
                const startValue = parseInt(element.textContent) || 0;
                const increment = Math.ceil((targetValue - startValue) / 20);
                let currentValue = startValue;
                
                const timer = setInterval(() => {
                    currentValue += increment;
                    if (currentValue >= targetValue) {
                        currentValue = targetValue;
                        clearInterval(timer);
                    }
                    element.textContent = currentValue + suffix;
                }, 50);
            }
            
            // Auto-resize textarea with smooth animation
            const textarea = document.getElementById('transcript');
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = this.scrollHeight + 'px';
            });
            
            // Add typing animation to textarea placeholder
            let placeholderIndex = 0;
            const placeholderTexts = [
                "Paste your conversation transcript here...",
                "Enter meeting notes for analysis...",
                "Add customer feedback to analyze...",
                "Input call recording text..."
            ];
            
            function rotatePlaceholder() {
                textarea.placeholder = placeholderTexts[placeholderIndex];
                placeholderIndex = (placeholderIndex + 1) % placeholderTexts.length;
            }
            
            // Rotate placeholder every 3 seconds
            setInterval(rotatePlaceholder, 3000);
            
            // Add smooth scroll behavior for the entire page
            document.documentElement.style.scrollBehavior = 'smooth';
            
            // Parallax effect for floating elements
            window.addEventListener('scroll', () => {
                const scrolled = window.pageYOffset;
                const rate = scrolled * -0.5;
                
                document.querySelectorAll('.floating-element').forEach((element, index) => {
                    const speed = (index + 1) * 0.3;
                    element.style.transform = `translateY(${rate * speed}px) rotate(${scrolled * 0.1}deg)`;
                });
            });
            
            // Add entrance animations for stats cards
            const observerOptions = {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            };
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('fade-in');
                    }
                });
            }, observerOptions);
            
            document.querySelectorAll('.stat-card').forEach(card => {
                observer.observe(card);
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Enhanced analysis endpoint"""
    try:
        data = request.get_json() if request.is_json else {"transcript": request.form.get("transcript", "")}
        transcript = data.get("transcript", "").strip()
        
        if not transcript:
            return jsonify({"error": "Transcript is required"}), 400
        
        if len(transcript) > 10000:
            return jsonify({"error": "Transcript too long (max 10,000 characters)"}), 400
        
        print(f"\nAnalyzing transcript ({len(transcript)} chars): {transcript[:100]}...")
        
        result, error = analyze_with_groq(transcript)
        if error:
            print(f"Analysis failed: {error}")
            return jsonify({"error": error}), 500
        
        print(f"Analysis successful: {result}")
        
        # Save to CSV
        try:
            append_to_csv(transcript, result)
        except Exception as csv_error:
            print(f"CSV save failed: {csv_error}")
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "groq_api_configured": bool(GROQ_API_KEY),
        "environment_loaded": True,
        "version": "2.0.0-env"
    })

if __name__ == '__main__':
    # Check if environment variables are loaded
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in environment variables!")
        print("Please check your .env file and ensure it contains:")
        print("GROQ_API_KEY=your_api_key_here")
        exit(1)
    
    print("Starting AI Transcript Analyzer Pro")
    print(f"API Key configured from environment: {bool(GROQ_API_KEY)}")
    print("Environment variables loaded successfully")
    print("Server starting at http://localhost:5000")
    
    # Set Flask environment variables if they exist
    if os.getenv('FLASK_DEBUG'):
        app.config['DEBUG'] = os.getenv('FLASK_DEBUG').lower() == 'true'
    
    app.run(debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true', host='0.0.0.0', port=5000)
