from dotenv import load_dotenv
load_dotenv()
import os
from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
from flask_session import Session
import google.generativeai as genai
import logging
import json
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import openai  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")
Session(app)


from dotenv import load_dotenv
import os

load_dotenv() 

gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-1.5-pro-latest')


openai_api_key = os.getenv("OPENAI_API_KEY")

class FinanceAgent:
    def __init__(self):
        self.supported_languages = {
            "english": {"code": "en", "name": "English"},
            "spanish": {"code": "es", "name": "Spanish"},
            "french": {"code": "fr", "name": "French"},
            "german": {"code": "de", "name": "German"},
            "chinese": {"code": "zh", "name": "Chinese"},
            "arabic": {"code": "ar", "name": "Arabic"},
        }
        self.language = "english"
        self.voices = {
            "voice_01": {"name": "Professional", "style": "formal"},
            "voice_02": {"name": "Friendly", "style": "casual"},
            "voice_03": {"name": "Authoritative", "style": "formal"},
            "voice_04": {"name": "Enthusiastic", "style": "energetic"},
            "voice_05": {"name": "Calm", "style": "relaxed"},
            "voice_06": {"name": "Technical", "style": "precise"}
        }
        self.current_voice = "voice_01"
        self.rag_data = self._load_rag_data()
        self.api_endpoints = {
            "get_balance": "https://www.alphavantage.co/query?function=BALANCE_SHEET",
            "get_stock": "https://www.alphavantage.co/query?function=GLOBAL_QUOTE",
            "get_income": "https://www.alphavantage.co/query?function=INCOME_STATEMENT",
            "get_crypto": "https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE"
        }
        self.finance_keywords = ["stock", "balance", "finance", "invest", "price", "portfolio", "account"]
        self.finance_keywords.extend(["crypto", "bitcoin", "btc", "ethereum", "xrp"])
        
    def set_language(self, language):
        if language in self.supported_languages:
            self.language = language
            return True
        return False
        
    def _load_rag_data(self):
        return [
            {"keywords": ["balance", "account"], "content": "Alpha Vantage provides balance sheet data via BALANCE_SHEET function"},
            {"keywords": ["stock", "price"], "content": "Use GLOBAL_QUOTE for current stock prices. Always specify a symbol parameter"}
        ]

    def _has_api_access(self):
        return os.getenv("ALPHA_VANTAGE_KEY") is not None

    def _call_alpha_vantage(self, function, symbol):
        params = {
            "apikey": os.getenv("ALPHA_VANTAGE_KEY"),
            "symbol": symbol,
            "datatype": "json"
        }
        try:
            response = requests.get(self.api_endpoints[function], params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Alpha Vantage Error: {e}")
            return None

    def _get_web_fallback(self, query):
        """Web fallback using BeautifulSoup to scrape financial data"""
        try:
            # Focus on reliable financial sources
            if any(crypto in query.lower() for crypto in ["bitcoin", "btc", "crypto"]):
                url = "https://www.coingecko.com/en"
                params = {"q": query.replace(" ", "+")}
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            else:
                url = "https://www.google.com/search"
                params = {"q": f"{query} site:finance.yahoo.com OR site:bloomberg.com"}
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Crypto-specific parsing
            if "coingecko" in url:
                price_element = soup.find('span', {'data-coin-symbol': 'btc'})
                if price_element:
                    return f"Current Bitcoin price: {price_element.text} (via CoinGecko)"
        
            # General finance parsing
            results = []
            for snippet in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
                text = snippet.get_text()
                if "$" in text or any(word in text.lower() for word in ["price", "stock", "value"]):
                    results.append(text)
        
            return " | ".join(results[:2]) if results else None

        except Exception as e:
            logger.error(f"Web scrape error: {e}")
            return None
            
    def is_finance_query(self, text):
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.finance_keywords)

    def process_command(self, input_data):
        if not self.is_finance_query(input_data.get('text', '')):
            return None
            
        try:
            intent = self._understand_intent(input_data['text'])
            operation_result = self._execute_operation(intent)
            return self._generate_response(intent, operation_result)
        except Exception as e:
            logger.error(f"Finance processing error: {e}")
            return {"error": str(e)}
    
    def _execute_operation(self, intent):
        action = intent.get("action")
        symbol = intent.get("symbol", "IBM")  # Default symbol
        
        if action == "get_crypto":
            symbol = intent.get("symbol", "BTC").upper()
            if "/" not in symbol:
                symbol = f"{symbol}/USD"
            if self._has_api_access():
                params = {
                    "apikey": os.getenv("ALPHA_VANTAGE_KEY"),
                    "from_currency": symbol.split("/")[0],
                    "to_currency": symbol.split("/")[1]
                }
                try:
                    response = requests.get(self.api_endpoints["get_crypto"], params=params)
                    data = response.json()
                    if "Realtime Currency Exchange Rate" in data:
                        return {
                            "status": "api_success",
                            "data": data["Realtime Currency Exchange Rate"],
                            "symbol": symbol
                        }
                except Exception as e:
                    logger.error(f"Crypto API error: {e}")
        
        if self._has_api_access() and action in self.api_endpoints:
            api_data = self._call_alpha_vantage(action, symbol)
            if api_data:
                return {
                    "status": "api_success",
                    "data": api_data,
                    "symbol": symbol
                }
        
        # Fallback to web search
        web_data = self._get_web_fallback(f"{action} {symbol}")
        return {
            "status": "search_based",
            "data": web_data or "No financial data available",
            "symbol": symbol
        }
        
    def _understand_intent(self, text):
        """Determine financial intent using Gemini"""
        prompt = f"""
        Analyze this financial query and identify the intent:
        "{text}"
    
        Return JSON with:
        {{
            "action": "get_stock|get_crypto|get_balance|get_income",
            "symbol": "ticker or crypto pair (default: BTC/USD)",
            "timeframe": "if relevant"
        }}
        """
    
        try:
            response = model.generate_content(prompt)
            # Extract JSON from response
            json_str = response.text.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]  # Remove markdown code block
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Intent understanding error: {e}")
            return {"action": "error", "error": str(e)}

    def _generate_response(self, intent, operation_result):
        # Special case for crypto responses
        if intent.get("action") == "get_crypto":
            if operation_result["status"] == "api_success":
                rate = operation_result["data"]
                return {
                    "text": f"The current price of {rate['1. From_Currency Name']} ({rate['1. From_Currency Code']}) "
                            f"is {rate['5. Exchange Rate']} {rate['3. To_Currency Code']} "
                            f"(as of {rate['6. Last Refreshed']})",
                    "voice": self.current_voice,
                    "source": "alpha_vantage",
                    "is_finance": True
                }
        prompt = f"""
        Financial request: {intent}
        Operation result: {operation_result}
        
        As a financial assistant, create:
        1. A concise (1-2 sentence) response
        2. In {self.voices[self.current_voice]['style']} style
        3. {"Mention this uses web data" if operation_result['status'] == 'search_based' else ""}
        """
        
        response = model.generate_content(prompt)
        return {
            "text": response.text,
            "voice": self.current_voice,
            "source": "alpha_vantage" if operation_result['status'] == 'api_success' else "web",
            "is_finance": True
        }

    def set_voice(self, voice_id):
        if voice_id in self.voices:
            self.current_voice = voice_id
            return True
        return False

    def get_settings(self):
        return {
            "current_voice": self.current_voice,
            "voices": self.voices,
            "language": self.language
        }

# Initialize agent
agent = FinanceAgent()

def text_to_speech(text, voice="alloy"):
    """
    Convert text to speech using OpenAI's TTS API
    voice options: alloy, echo, fable, onyx, nova, shimmer
    """
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"static/audio/tts_{timestamp}.mp3"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        response.stream_to_file(filename)
        
        return {
            "status": "success",
            "audio_url": f"/{filename}"
        }
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def save_to_history(user_input, ai_response):
    if "chat_history" not in session:
        session["chat_history"] = []
    
    session["chat_history"].append({
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "ai": ai_response
    })
    session.modified = True

def handle_general_conversation(query):
    try:
        response = model.generate_content(f"""
        Respond naturally to this general conversation in 1-2 sentences:
        User: {query}
        """)
        return {
            "text": response.text,
            "source": "gemini",
            "is_finance": False
        }
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {
            "text": "I couldn't process your request. Please try again.",
            "source": "error",
            "is_finance": False
        }



@app.route('/process-command', methods=['POST'])
def process_command():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_input = data.get("text", "")
    generate_audio = data.get("generate_audio", False)
    
    # Try finance first
    finance_response = agent.process_command(data)
    if finance_response and not finance_response.get("error"):
        response = finance_response
    else:
        # Fallback to general conversation
        response = handle_general_conversation(user_input)
        
    # Generate speech if requested
    if generate_audio:
        # Map your voice styles to OpenAI voices
        voice_mapping = {
            "voice_01": "onyx",  # Professional -> onyx
            "voice_02": "nova",  # Friendly -> nova
            "voice_03": "echo",  # Authoritative -> echo
            "voice_04": "shimmer",  # Enthusiastic -> shimmer
            "voice_05": "alloy",  # Calm -> alloy
            "voice_06": "fable"   # Technical -> fable
        }
        
        voice = voice_mapping.get(agent.current_voice, "alloy")
        tts_result = text_to_speech(response["text"], voice)
        
        # Check if audio was generated successfully and convert file to base64
        if tts_result["status"] == "success":
            try:
                audio_path = tts_result["audio_url"].lstrip('/')
                with open(audio_path, 'rb') as audio_file:
                    import base64
                    audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
                    response["audio"] = audio_data
                    logger.info(f"Audio encoded successfully from {audio_path}")
            except Exception as e:
                logger.error(f"Audio file reading error: {e}")
                response["audio_error"] = str(e)
        else:
            response["audio_error"] = tts_result.get("message", "Unknown TTS error")
            logger.error(f"TTS error: {tts_result.get('message')}")
    
    save_to_history(user_input, response)
    return jsonify(response)

@app.route('/get-chat-history', methods=['GET'])
def get_chat_history():
    return jsonify(session.get("chat_history", []))
@app.route("/")
def home():
    return render_template("index6.html")
@app.route("/set-voice", methods=["POST"])
def set_voice():
    data = request.json
    if agent.set_voice(data.get("voice_id", "")):
        return jsonify({"status": "success"})
    return jsonify({"error": "Invalid voice ID"}), 400

@app.route("/get-settings", methods=["GET"])
def get_settings():
    return jsonify(agent.get_settings())

@app.route("/set-language", methods=["POST"])
def set_language():
    data = request.get_json()
    if not data or "language" not in data:
        return jsonify({"error": "Language not specified"}), 400
    
    language = data["language"].lower()
    if language not in ["english", "spanish", "french", "german", "chinese", "arabic"]:  
        return jsonify({"error": "Unsupported language"}), 400
    
    session["current_language"] = language  # Store in session
    if agent.set_language(language):
        return jsonify({
            "status": "success",
            "message": f"Language changed to {language}",
            "current_language": language
        })
    return jsonify({"error": "Language change failed"}), 500

@app.route("/test-connection", methods=["GET"])
def test_connection():
    return jsonify({
        "status": "success",
        "message": "Backend connected",
        "services": {
            "gemini": gemini_key is not None,
            "alpha_vantage": os.getenv("ALPHA_VANTAGE_KEY") is not None,
            "serpapi": os.getenv("SERPAPI_KEY") is not None
        }
    })

import os
os.makedirs("static/audio", exist_ok=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
