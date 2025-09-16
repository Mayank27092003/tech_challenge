import os
import json
import requests
import time
import logging
import re
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"

@dataclass
class AnalysisResult:
    """Structured result for transcript analysis"""
    summary: str
    sentiment: SentimentType
    confidence_score: Optional[float] = None
    key_topics: Optional[List[str]] = None
    word_count: Optional[int] = None

@dataclass 
class GroqConfig:
    """Configuration for Groq API calls"""
    model: str = "llama-3.1-8b-instant"   # use available chat model
    temperature: float = 0.1
    max_tokens: int = 1024
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30

class GroqTranscriptAnalyzer:
    """Advanced transcript analyzer using Groq API"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[GroqConfig] = None):
        # Load API key from environment variables or use provided key
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.config = config or GroqConfig()
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment variables. Please check your .env file.")
    
    def _validate_transcript(self, transcript: str) -> None:
        """Validate input transcript"""
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        
        if len(transcript) > 50000:  # Reasonable limit
            logger.warning("Transcript is very long, consider chunking")
    
    def _create_enhanced_prompt(self, transcript: str) -> str:
        """Create enhanced prompt for better analysis"""
        return f"""
        Analyze the following transcript and return ONLY a valid JSON object with the specified fields:
        
        Transcript: {transcript}
        
        Instructions:
        1. Provide a concise but comprehensive summary (2-3 sentences)
        2. Determine sentiment: positive, negative, or neutral
        3. Estimate confidence score (0.0-1.0) for sentiment analysis
        4. Extract 3-5 key topics or themes
        5. Count approximate word count
        
        Return format (MUST be valid JSON):
        {{
            "summary": "brief but comprehensive summary here",
            "sentiment": "positive/negative/neutral",
            "confidence_score": 0.85,
            "key_topics": ["topic1", "topic2", "topic3"],
            "word_count": 150
        }}
        """
    
    def _make_api_request(self, transcript: str) -> Dict[str, Any]:
        """Make API request with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user", 
                    "content": self._create_enhanced_prompt(transcript)
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"}  # Force strict JSON output
        }
        
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"API request attempt {attempt + 1}")
                
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
        
        raise Exception("Max retries exceeded")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate API response with JSON rescue fallback"""
        try:
            content = response_data["choices"][0]["message"]["content"].strip()
            
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            # Primary JSON parsing attempt
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # JSON rescue fallback - extract JSON from mixed content
                logger.warning("Direct JSON parsing failed, trying extraction...")
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                    logger.info("Successfully extracted JSON from mixed content")
                else:
                    logger.error(f"Raw content: {content}")
                    raise ValueError("Failed to parse JSON response from Groq")
            
            # Validate required fields
            required_fields = ["summary", "sentiment"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate sentiment value
            if result["sentiment"] not in [s.value for s in SentimentType]:
                logger.warning(f"Invalid sentiment value: {result['sentiment']}, defaulting to neutral")
                result["sentiment"] = SentimentType.NEUTRAL.value
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed completely: {e}")
            logger.error(f"Raw content: {content}")
            raise ValueError("Failed to parse JSON response from Groq")
    
    def analyze(self, transcript: str) -> Tuple[Optional[AnalysisResult], Optional[str]]:
        """
        Analyze transcript and return structured result
        
        Args:
            transcript: Text transcript to analyze
            
        Returns:
            Tuple of (AnalysisResult, error_message)
        """
        try:
            # Validate input
            self._validate_transcript(transcript)
            
            # Make API request
            response_data = self._make_api_request(transcript)
            
            # Parse response
            result_dict = self._parse_response(response_data)
            
            # Create structured result
            result = AnalysisResult(
                summary=result_dict["summary"],
                sentiment=SentimentType(result_dict["sentiment"]),
                confidence_score=result_dict.get("confidence_score"),
                key_topics=result_dict.get("key_topics"),
                word_count=result_dict.get("word_count")
            )
            
            logger.info("Analysis completed successfully")
            return result, None
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def analyze_batch(self, transcripts: List[str]) -> List[Tuple[Optional[AnalysisResult], Optional[str]]]:
        """Analyze multiple transcripts with rate limiting"""
        results = []
        
        for i, transcript in enumerate(transcripts):
            logger.info(f"Processing transcript {i+1}/{len(transcripts)}")
            
            result = self.analyze(transcript)
            results.append(result)
            
            # Rate limiting - avoid overwhelming API
            if i < len(transcripts) - 1:
                time.sleep(0.5)
        
        return results

def analyze_with_groq(transcript: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Backward compatible version with environment variables and JSON fixes"""
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
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
                "role": "user",
                "content": f"""Analyze the following transcript and return ONLY a valid JSON object with exactly two fields: 'summary' and 'sentiment'.
Transcript: {transcript}
Return format:
{{"summary": "brief summary here", "sentiment": "positive/negative/neutral"}}"""
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}  # Force strict JSON
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        
        # JSON rescue fallback
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                return None, "Failed to parse JSON response"
        
        if "summary" not in result or "sentiment" not in result:
            return None, "Missing required fields in Groq response"
        return result, None
    except Exception as e:
        return None, str(e)

# Example usage with environment variables
if __name__ == "__main__":
    # Check if environment variables are loaded
    if not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY not found in environment variables!")
        print("Please ensure you have a .env file with:")
        print("GROQ_API_KEY=your_api_key_here")
        exit(1)
    
    print("Environment variables loaded successfully")
    print(f"API Key configured: {bool(os.getenv('GROQ_API_KEY'))}")
    
    sample_transcript = """
    Hi everyone, welcome to today's meeting. I'm really excited to share the quarterly results with you. 
    Our team has achieved outstanding performance this quarter, exceeding all our targets by 15%. 
    The customer satisfaction scores are at an all-time high, and we've successfully launched three new products. 
    I want to thank everyone for their hard work and dedication. Let's keep this momentum going into the next quarter.
    """
    
    try:
        # Using the enhanced analyzer
        print("\n--- Enhanced Analyzer Test ---")
        analyzer = GroqTranscriptAnalyzer()
        result, error = analyzer.analyze(sample_transcript)
        
        if result:
            print("Analysis Results:")
            print(f"Summary: {result.summary}")
            print(f"Sentiment: {result.sentiment.value}")
            print(f"Confidence: {result.confidence_score}")
            print(f"Key Topics: {result.key_topics}")
            print(f"Word Count: {result.word_count}")
        else:
            print(f"Error: {error}")
        
        # Using the backward compatible function
        print("\n--- Backward Compatible Function Test ---")
        result, error = analyze_with_groq(sample_transcript)
        if result:
            print(f"Summary: {result['summary']}")
            print(f"Sentiment: {result['sentiment']}")
        else:
            print(f"Error: {error}")
            
    except Exception as e:
        print(f"Setup error: {e}")
        print("Please check your .env file and API key configuration")
