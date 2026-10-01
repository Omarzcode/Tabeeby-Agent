from ..LLMEnums import OllamaEnums
from ..LLMInterface import LLMInterface
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import logging

try:
    from ollama import Client
except ImportError:
    Client = None
    ChatResponse = Any 


class OllamaProvider(LLMInterface):

    def __init__(
        self,
        api_key:Optional[str],
        api_url:str="http://localhost:11434",
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
        ):

        self.api_key=api_key
        self.api_url = api_url or "http://localhost:11434"

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.client=None
        self.logger= logging.getLogger(__name__)

        if Client is None:
            self.logger.error("Ollama package is not installed")
            return

        
        # Ollama Cloud (https://ollama.com) needs the API key as a bearer token;
        # a local daemon ignores it (cloud models there need `ollama signin`).
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self.client=Client(host=self.api_url, headers=headers)

    def set_generation_model(self,model_id:str)->None:
        self.generation_model_id=model_id

    def generate_response(
            self, 
            prompt: str,
            chat_history: Optional[List[Dict[str,Any]]]= None, 
            max_output_tokens: Optional[int]=None,
            temperature: Optional[float] = None,
            tools: Optional[Sequence[Callable[..., Any]]] = None,
            truncate:bool=False
            )-> Any:
        
        if not self.client:
            self.logger.error("Ollama client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Ollama was not set")
            return None

        max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.default_generation_max_output_tokens
        )

        temperature = (
            temperature
            if temperature is not None
            else self.default_generation_temperature
        )

        messages = list(chat_history) if chat_history else []
        if prompt and prompt.strip():
            messages.append(self.construct_prompt(
                prompt=prompt,
                role=OllamaEnums.USER.value,
                truncate=truncate))

        try:
            response=self.client.chat(
                model=self.generation_model_id,
                messages=messages,
                stream=False,
                options={
                    "temperature": temperature,
                    "num_predict": max_output_tokens,
                },
                tools=list(tools) if tools else None
            )

            return response
        
        except Exception as e:
            self.logger.error(f"Error while generating text with Ollama: {e}")
            return None


       

        

    def process_text(self,text:str)->str:
        return text[:self.default_input_max_characters].strip()
    
    def construct_prompt(
        self,
        prompt:str,
        role:str,
        truncate:bool=False
        )->Dict[str,str]:

        if truncate:
            return  {
                "role":role,
                "content":self.process_text(prompt)
            }
        

        return {
                "role":role,
                "content":prompt
            } 

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size


    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("Ollama client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Ollama was not set")
            return None

        response = self.client.embed(
            model=self.embedding_model_id,
            input=text.strip(),
            dimensions=self.embedding_size
        )

        if not response:
            self.logger.error("Error while embedding text with Ollama")
            return None

        embeddings = None
        if hasattr(response, "embeddings"):
            embeddings = response.embeddings
        elif isinstance(response, dict):
            embeddings = response.get("embeddings")

        if not embeddings or len(embeddings) == 0:
            self.logger.error("Error while embedding text with Ollama")
            return None

        return embeddings[0]