import requests
from typing import List, Dict

class LocalLLMEngine:
    def __init__(self, api_url: str = "http://127.0.0.1:8080"):
        self.api_url = api_url
        # Test connection on init
        self._test_connection()
        self.default_model = self._select_default_model()

    def _test_connection(self) -> None:
        try:
            response = requests.get(f"{self.api_url}/v1/models")
            if response.status_code != 200:
                raise ConnectionError(f"Could not connect to LLM server at {self.api_url}")
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to LLM server: {str(e)}")

    def _get_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.api_url}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = []
            # handle a few common response shapes
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                for m in data["data"]:
                    if isinstance(m, dict):
                        models.append(m.get("id") or m.get("name") or m.get("model"))
                    elif isinstance(m, str):
                        models.append(m)
            elif isinstance(data, list):
                for m in data:
                    if isinstance(m, dict):
                        models.append(m.get("id") or m.get("name") or m.get("model"))
                    elif isinstance(m, str):
                        models.append(m)
            # filter out None
            return [m for m in models if m]
        except Exception:
            return []

    def _select_default_model(self) -> str:
        models = self._get_models()
        return models[0] if models else None

    def _extract_content(self, resp_json: Dict) -> str:
        # common OpenAI-like shape
        try:
            return resp_json["choices"][0]["message"]["content"]
        except Exception:
            pass
        try:
            return resp_json["choices"][0]["text"]
        except Exception:
            pass
        try:
            return resp_json["output"][0].get("content") or resp_json["output"][0].get("text")
        except Exception:
            pass
        # fallback: return full json as string
        return str(resp_json)

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0, model: str = None) -> str:
        model_to_use = model or self.default_model
        if not model_to_use:
            raise Exception("No model specified and unable to auto-detect a model from the server. "
                            "Either provide --local_llm_url pointing to a server with /v1/models or call generate(..., model='model-name').")

        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
            "stop": ["\n\n"],
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return self._extract_content(response.json())
            else:
                raise Exception(f"API call failed ({response.status_code}): {response.text}")
        except requests.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
