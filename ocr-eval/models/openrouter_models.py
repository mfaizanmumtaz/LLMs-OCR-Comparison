import base64
import io
import json
import os
import time
from PIL import Image
import aiohttp
import aiofiles
import asyncio
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# List of available models for command line choices
AVAILABLE_MODELS = ["google", "gpt-5-mini", "gpt-4.1-mini", "mistralai"]

async def load_timing_data_async(filename="llm_timing_data.json"):
    """Load existing timing data from JSON file asynchronously"""
    try:
        async with aiofiles.open(filename, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Failed to load timing data: {e}")
        return {}

async def save_timing_data_async(data, filename="llm_timing_data.json"):
    """Save timing data to JSON file asynchronously"""
    try:
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to save timing data: {e}")

class AsyncOpenRouterModel:
    def __init__(self, **kwargs):
        print(f"AsyncOpenRouterModel is being initialized with model: {kwargs['model_name']}")
        self.max_tokens = kwargs["max_tokens"]
        self.model_name = kwargs["model_name"]
        self.api_key = os.getenv("API_KEY_REF")
        if not self.api_key:
            raise ValueError("API_KEY_REF environment variable not set")
        
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timing_data = []  # Store timing data for this session
        
        # Create a session that will be reused for all requests
        self.session = None
        self._session_lock = asyncio.Lock()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with robust connection settings"""
        if self.session is None or self.session.closed:
            async with self._session_lock:
                if self.session is None or self.session.closed:
                    # Extended timeout for OpenRouter
                    timeout = aiohttp.ClientTimeout(
                        total=600,      # 10 minutes total timeout
                        connect=30,     # 30s to establish connection
                        sock_read=300   # 5 minutes to read response
                    )
                    
                    # Connection settings optimized for reliability
                    connector = aiohttp.TCPConnector(
                        limit=50,                    # Connection pool limit
                        limit_per_host=25,           # Limit per host
                        keepalive_timeout=120,       # Keep connections alive longer
                        enable_cleanup_closed=True,  # Clean up closed connections
                        ttl_dns_cache=300,          # DNS cache TTL
                        use_dns_cache=True,         # Enable DNS caching
                    )
                    
                    self.session = aiohttp.ClientSession(
                        timeout=timeout,
                        connector=connector,
                        headers=self.headers
                    )
        return self.session
    
    def _enc(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64"""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def _get_messages(self, image: Image, prompt: str) -> List[Dict]:
        """Convert image and prompt to message format"""
        # Run image processing in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def process_image():
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            return buf.read()
        
        image_bytes = await loop.run_in_executor(None, process_image)
        image_base64 = self._enc(image_bytes)

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

    async def save_timing_stats_async(self):
        """Save timing statistics to JSON file asynchronously"""
        if not self.timing_data:
            return
            
        # Calculate average
        avg_time = sum(self.timing_data) / len(self.timing_data)
        
        # Load existing data
        all_timing_data = await load_timing_data_async()
        
        # Update with current model data
        all_timing_data[self.model_name] = {
            "call_times": self.timing_data,
            "average_time": round(avg_time, 3),
            "total_calls": len(self.timing_data)
        }
        
        # Save updated data
        await save_timing_data_async(all_timing_data)
        
        logger.info(f"Timing stats for {self.model_name}:")
        logger.info(f"Total calls: {len(self.timing_data)}")
        logger.info(f"Average time: {avg_time:.3f}s")
        logger.info(f"Data saved to llm_timing_data.json")

    def _is_retryable_error(self, error: Exception, response_data: dict = None) -> bool:
        """Determine if an error is retryable"""
        error_str = str(error).lower()
        
        # Network connectivity issues
        network_errors = [
            'connection', 'timeout', 'network', 'dns', 'socket',
            'ssl', 'certificate', 'handshake', 'reset', 'refused'
        ]
        
        # OpenRouter specific retryable errors
        openrouter_retryable = [
            'rate limit', 'too many requests', 'max retries',
            'server error', 'internal error', 'service unavailable',
            'temporarily unavailable', 'overloaded', 'capacity',
            'quota exceeded'
        ]
        
        # JSON parsing errors (might be temporary server issues)
        json_errors = ['json', 'parsing', 'decode']
        
        # Check error message
        for error_type in network_errors + openrouter_retryable + json_errors:
            if error_type in error_str:
                return True
        
        # Check response data for specific OpenRouter error codes
        if response_data and isinstance(response_data, dict):
            error_info = response_data.get('error', {})
            if isinstance(error_info, dict):
                error_code = error_info.get('code', '')
                error_type = error_info.get('type', '')
                
                # Retryable HTTP status codes and error types
                retryable_codes = ['rate_limit_exceeded', 'server_error', 'service_unavailable']
                if error_code in retryable_codes or error_type in retryable_codes:
                    return True
        
        return False

    def _calculate_retry_delay(self, attempt: int, base_delay: float = 2.0) -> float:
        """Calculate retry delay with jitter to avoid thundering herd"""
        import random
        
        # Exponential backoff with jitter
        delay = base_delay * (2 ** attempt)
        
        # Add random jitter (±25% of delay)
        jitter = delay * 0.25 * random.random()
        final_delay = delay + jitter
        
        # Cap maximum delay at 60 seconds
        return min(final_delay, 60.0)

    async def __call__(self, prompt: str, image: Image, max_retries: int = 5) -> str:
        """Make async API call with robust retry logic"""
        print(f"AsyncOpenRouterModel.__call__ is being called for model: {self.model_name}")
        messages = await self._get_messages(image, prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        
        last_exception = None
        non_retryable_errors = 0
        
        for attempt in range(max_retries):
            try:
                # Add delay for retries
                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt - 1)
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {self.model_name} after {delay:.1f}s delay")
                    await asyncio.sleep(delay)
                
                # Time the API call
                start_time = time.time()
                
                session = await self._get_session()
                
                try:
                    print(f"Making API request to OpenRouter for model: {self.model_name}")
                    async with session.post(self.url, json=payload) as response:
                        end_time = time.time()
                        
                        # Calculate and store timing
                        call_time = round(end_time - start_time, 3)
                        self.timing_data.append(call_time)
                        logger.debug(f"LLM call took: {call_time}s")
                        
                        # Read response
                        response_text = await response.text()
                        response_data = None
                        
                        # Try to parse JSON
                        try:
                            response_data = json.loads(response_text)
                        except json.JSONDecodeError as e:
                            json_error = Exception(f"OpenRouter returned non-JSON response: {e}; text={response_text[:200]}...")
                            if self._is_retryable_error(json_error):
                                raise json_error
                            else:
                                # Non-retryable JSON error
                                non_retryable_errors += 1
                                raise json_error
                        
                        # Check for HTTP errors
                        if response.status != 200:
                            if response_data:
                                error_info = response_data.get('error', {})
                                if isinstance(error_info, dict):
                                    error_msg = error_info.get('message', str(response_data))
                                    error_code = error_info.get('code', '')
                                else:
                                    error_msg = str(response_data)
                                    error_code = ''
                            else:
                                error_msg = f"HTTP {response.status}"
                                error_code = ''
                            
                            api_error = Exception(f"OpenRouter API error {response.status}: {error_msg} (code: {error_code})")
                            
                            # Check if this is a retryable error
                            if not self._is_retryable_error(api_error, response_data):
                                non_retryable_errors += 1
                                if non_retryable_errors >= 2:  # Stop after 2 non-retryable errors
                                    logger.error(f"Multiple non-retryable errors, stopping retries: {error_msg}")
                                    return ""
                            
                            raise api_error
                        
                        # Extract content from response
                        if not isinstance(response_data, dict) or 'choices' not in response_data or not response_data['choices']:
                            format_error = Exception(f"Unexpected OpenRouter response format (missing 'choices'): {response_data}")
                            if not self._is_retryable_error(format_error, response_data):
                                non_retryable_errors += 1
                            raise format_error
                        
                        choice = response_data['choices'][0]
                        message = choice.get('message', {})
                        content = message.get('content')
                        
                        if content is None:
                            content_error = Exception(f"OpenRouter response has no message content: {response_data}")
                            if not self._is_retryable_error(content_error, response_data):
                                non_retryable_errors += 1
                            raise content_error
                        
                        logger.debug(f"Successfully got response: {content[:100]}...")
                        return content
                
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                    # Network-related errors are always retryable
                    network_error = Exception(f"Network error: {str(e)}")
                    raise network_error
                    
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                # Log different types of errors differently
                if self._is_retryable_error(e):
                    logger.warning(f"Retryable error on attempt {attempt + 1}/{max_retries} for {self.model_name}: {error_msg}")
                else:
                    logger.error(f"Non-retryable error on attempt {attempt + 1}/{max_retries} for {self.model_name}: {error_msg}")
                
                # If this is the last attempt, log the final failure
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for {self.model_name}. Last error: {error_msg}")
                    break
                
                # If we've hit too many non-retryable errors, stop early
                if non_retryable_errors >= 2:
                    logger.error(f"Too many non-retryable errors for {self.model_name}, stopping early")
                    break
        
        # If all retries failed, return empty string
        logger.error(f"Failed to get response from {self.model_name} after {max_retries} attempts")
        return ""

    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Save timing stats when closing
        await self.save_timing_stats_async()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

# Keep the old synchronous version for backwards compatibility
class OpenRouterModel:
    def __init__(self, **kwargs):
        print("i am getting called")
        self.max_tokens = kwargs["max_tokens"]
        self.model_name = kwargs["model_name"]
        self.api_key = os.getenv("API_KEY_REF")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timing_data = []  # Store timing data for this session
    
    def _enc(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def _get_messages(self, image: Image, prompt):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        image_bytes = buf.read()
        image_base64 = self._enc(image_bytes)

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

    def save_timing_stats(self):
        """Save timing statistics to JSON file"""
        if not self.timing_data:
            return
            
        # Calculate average
        avg_time = sum(self.timing_data) / len(self.timing_data)
        
        # Load existing data
        try:
            with open("llm_timing_data.json", 'r', encoding='utf-8') as f:
                all_timing_data = json.load(f)
        except FileNotFoundError:
            all_timing_data = {}
        
        # Update with current model data
        all_timing_data[self.model_name] = {
            "call_times": self.timing_data,
            "average_time": round(avg_time, 3),
            "total_calls": len(self.timing_data)
        }
        
        # Save updated data
        with open("llm_timing_data.json", 'w', encoding='utf-8') as f:
            json.dump(all_timing_data, f, indent=2)
        
        print(f"\nTiming stats for {self.model_name}:")
        print(f"Total calls: {len(self.timing_data)}")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Call times: {self.timing_data}")
        print(f"Data saved to llm_timing_data.json")
    
    def force_save_timing_stats(self):
        """Force save timing statistics to JSON file"""
        self.save_timing_stats()

    def __call__(self, prompt, image):
        import requests
        
        messages = self._get_messages(image, prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        
        # Time the LLM call
        start_time = time.time()
        response = requests.post(self.url, headers=self.headers, json=payload)
        end_time = time.time()
        
        # Calculate and store timing
        call_time = round(end_time - start_time, 3)  # Keep in seconds with 3 decimal places
        self.timing_data.append(call_time)
        print(f"LLM call took: {call_time}s")
        
        # Save timing stats after each call to ensure data is preserved
        if len(self.timing_data) % 10 == 0:  # Save every 10 calls to avoid too frequent writes
            self.save_timing_stats()
        
        # Try to parse JSON and handle unexpected responses
        try:
            response_data = response.json()
        except Exception as e:
            # Non-JSON response (or empty). Include raw text for debugging.
            raise Exception(f"OpenRouter returned non-JSON response: {e}; text={response.text}")

        # If the API returned an error status, include the body for debugging
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error {response.status_code}: {response_data}")
        print(response_data)
        # Extract content from OpenRouter response format
        if not isinstance(response_data, dict) or 'choices' not in response_data or not response_data['choices']:
            # Unexpected format - surface the whole response so caller can log it
            raise Exception(f"Unexpected OpenRouter response format (missing 'choices'): {response_data}")

        out_text = response_data['choices'][0].get('message', {}).get('content')
        if out_text is None:
            # If content is absent, show full response for debugging
            raise Exception(f"OpenRouter response has no message content: {response_data}")

        print(out_text)
        return out_text

    def __del__(self):
        """Save timing stats when the object is destroyed"""
        try:
            self.save_timing_stats()
        except:
            pass  # Ignore errors during cleanup