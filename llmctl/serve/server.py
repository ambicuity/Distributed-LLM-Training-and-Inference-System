"""
Inference server implementation with dynamic batching and KV cache management.
"""

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console

console = Console()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False

class GenerationResponse(BaseModel):
    id: str
    text: str
    finish_reason: str
    usage: Dict[str, int]

@dataclass
class RequestState:
    """State for a generation request."""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stream: bool
    input_ids: torch.Tensor
    generated_tokens: List[int]
    created_at: float
    prompt_tokens: int
    completion_tokens: int = 0
    finished: bool = False
    finish_reason: str = ""

class KVCacheManager:
    """Manages KV cache for efficient attention computation."""
    
    def __init__(self, max_cache_size: int = 1024):
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.access_times: Dict[str, float] = {}
    
    def get_cache(self, request_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached key-value tensors for a request."""
        if request_id in self.cache:
            self.access_times[request_id] = time.time()
            return self.cache[request_id]
        return None
    
    def set_cache(self, request_id: str, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """Cache key-value tensors for a request."""
        # Evict oldest entries if cache is full
        while len(self.cache) >= self.max_cache_size:
            oldest_request = min(self.access_times.keys(), key=self.access_times.get)
            del self.cache[oldest_request]
            del self.access_times[oldest_request]
        
        self.cache[request_id] = (k_cache, v_cache)
        self.access_times[request_id] = time.time()
    
    def clear_cache(self, request_id: str):
        """Clear cache for a completed request."""
        if request_id in self.cache:
            del self.cache[request_id]
            del self.access_times[request_id]

class DynamicBatchScheduler:
    """Dynamic batching scheduler for efficient inference."""
    
    def __init__(self, max_batch_size: int = 8, max_batch_tokens: int = 8192):
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.pending_requests: deque = deque()
        self.active_batches: List[List[RequestState]] = []
    
    def add_request(self, request: RequestState):
        """Add a new request to the pending queue."""
        self.pending_requests.append(request)
    
    def get_next_batch(self) -> List[RequestState]:
        """Get the next batch of requests to process."""
        if not self.pending_requests:
            return []
        
        batch = []
        total_tokens = 0
        
        while (len(batch) < self.max_batch_size and 
               total_tokens < self.max_batch_tokens and 
               self.pending_requests):
            
            request = self.pending_requests.popleft()
            request_tokens = len(request.input_ids) + request.max_tokens
            
            if total_tokens + request_tokens <= self.max_batch_tokens:
                batch.append(request)
                total_tokens += request_tokens
            else:
                # Put request back if it doesn't fit
                self.pending_requests.appendleft(request)
                break
        
        return batch

class InferenceEngine:
    """Core inference engine with model loading and generation."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.kv_cache = KVCacheManager()
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the model and tokenizer."""
        console.print(f"[blue]Loading model from {self.model_path}...[/blue]")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            console.print(f"[green]✓ Model loaded successfully on {self.device}[/green]")
            
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}[/red]")
            raise
    
    def generate_batch(self, requests: List[RequestState]) -> List[RequestState]:
        """Generate tokens for a batch of requests."""
        if not requests:
            return []
        
        # Prepare batch inputs
        input_ids_list = []
        attention_masks = []
        
        for request in requests:
            input_ids_list.append(request.input_ids)
        
        # Pad sequences to the same length
        max_length = max(len(ids) for ids in input_ids_list)
        
        for i, input_ids in enumerate(input_ids_list):
            if len(input_ids) < max_length:
                padding = torch.full((max_length - len(input_ids),), 
                                   self.tokenizer.pad_token_id, 
                                   dtype=input_ids.dtype, 
                                   device=input_ids.device)
                input_ids_list[i] = torch.cat([input_ids, padding])
        
        batch_input_ids = torch.stack(input_ids_list)
        attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).long()
        
        # Generate next tokens
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]  # Get last token logits
            
            # Apply temperature and top-p sampling
            for i, request in enumerate(requests):
                token_logits = logits[i]
                
                if request.temperature != 1.0:
                    token_logits = token_logits / request.temperature
                
                # Top-k filtering
                if request.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(token_logits, request.top_k)
                    token_logits = torch.full_like(token_logits, float('-inf'))
                    token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) sampling
                if request.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > request.top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Update request state
                request.generated_tokens.append(next_token)
                request.completion_tokens += 1
                request.input_ids = torch.cat([
                    request.input_ids, 
                    torch.tensor([next_token], device=request.input_ids.device)
                ])
                
                # Check for completion
                if (next_token == self.tokenizer.eos_token_id or 
                    request.completion_tokens >= request.max_tokens):
                    request.finished = True
                    request.finish_reason = "stop" if next_token == self.tokenizer.eos_token_id else "length"
        
        return requests

class InferenceServer:
    """Main inference server with FastAPI integration."""
    
    def __init__(self, 
                 model_path: str,
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 max_batch_size: int = 8,
                 max_batch_tokens: int = 8192,
                 max_concurrent: int = 128):
        
        self.model_path = model_path
        self.host = host
        self.port = port
        self.max_concurrent = max_concurrent
        
        self.engine = InferenceEngine(model_path)
        self.scheduler = DynamicBatchScheduler(max_batch_size, max_batch_tokens)
        self.active_requests: Dict[str, RequestState] = {}
        self.completed_requests: Dict[str, RequestState] = {}
        
        # Initialize FastAPI app
        self.app = FastAPI(title="LLMCtl Inference Server")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/v1/completions", response_model=GenerationResponse)
        async def create_completion(request: GenerationRequest):
            return await self.handle_generation_request(request)
        
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [{
                    "id": self.model_path,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llmctl"
                }]
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "active_requests": len(self.active_requests),
                "pending_requests": len(self.scheduler.pending_requests)
            }
    
    async def handle_generation_request(self, request: GenerationRequest) -> GenerationResponse:
        """Handle a generation request."""
        if len(self.active_requests) >= self.max_concurrent:
            raise HTTPException(status_code=503, detail="Server at capacity")
        
        # Create request state
        request_id = str(uuid.uuid4())
        
        # Tokenize input
        input_ids = self.engine.tokenizer.encode(
            request.prompt, 
            return_tensors="pt", 
            add_special_tokens=True
        ).to(self.engine.device).squeeze(0)
        
        request_state = RequestState(
            request_id=request_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=request.stream,
            input_ids=input_ids,
            generated_tokens=[],
            created_at=time.time(),
            prompt_tokens=len(input_ids)
        )
        
        self.active_requests[request_id] = request_state
        self.scheduler.add_request(request_state)
        
        # Wait for completion
        while not request_state.finished:
            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
        
        # Generate response
        generated_text = self.engine.tokenizer.decode(
            request_state.generated_tokens, 
            skip_special_tokens=True
        )
        
        response = GenerationResponse(
            id=request_id,
            text=generated_text,
            finish_reason=request_state.finish_reason,
            usage={
                "prompt_tokens": request_state.prompt_tokens,
                "completion_tokens": request_state.completion_tokens,
                "total_tokens": request_state.prompt_tokens + request_state.completion_tokens
            }
        )
        
        # Cleanup
        self.completed_requests[request_id] = self.active_requests.pop(request_id)
        self.engine.kv_cache.clear_cache(request_id)
        
        return response
    
    async def process_batches(self):
        """Background task to process batches of requests."""
        while True:
            batch = self.scheduler.get_next_batch()
            if batch:
                try:
                    self.engine.generate_batch(batch)
                except Exception as e:
                    console.print(f"[red]Batch processing error: {e}[/red]")
                    # Mark all requests in batch as failed
                    for request in batch:
                        request.finished = True
                        request.finish_reason = "error"
            
            await asyncio.sleep(0.001)  # Very small delay
    
    async def start_server(self):
        """Start the inference server."""
        console.print(f"[blue]Initializing inference server...[/blue]")
        
        # Load model
        self.engine.load_model()
        
        # Start background batch processor
        asyncio.create_task(self.process_batches())
        
        console.print(f"[green]✓ Server starting on {self.host}:{self.port}[/green]")
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

def create_inference_server(model_path: str, **kwargs) -> InferenceServer:
    """Factory function to create an inference server."""
    return InferenceServer(model_path, **kwargs)