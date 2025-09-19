#!/usr/bin/env python3
"""
Demo script to test the inference server functionality.
"""

import asyncio
import json
import time
from pathlib import Path
import aiohttp

async def test_inference_server():
    """Test the inference server with sample requests."""
    
    print("🧪 Testing LLMCtl Inference Server")
    
    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    await asyncio.sleep(2)
    
    # Test data
    test_requests = [
        {
            "prompt": "The future of AI is",
            "max_tokens": 20,
            "temperature": 0.7
        },
        {
            "prompt": "Machine learning helps",
            "max_tokens": 15,
            "temperature": 0.8
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test health endpoint
            print("\n📡 Testing health endpoint...")
            async with session.get("http://localhost:8080/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    print(f"✅ Health check passed: {health_data}")
                else:
                    print(f"❌ Health check failed: {resp.status}")
                    return
            
            # Test models endpoint
            print("\n📋 Testing models endpoint...")
            async with session.get("http://localhost:8080/v1/models") as resp:
                if resp.status == 200:
                    models_data = await resp.json()
                    print(f"✅ Models endpoint: {models_data}")
                else:
                    print(f"❌ Models endpoint failed: {resp.status}")
            
            # Test inference requests
            print("\n🔮 Testing inference requests...")
            for i, request_data in enumerate(test_requests, 1):
                print(f"\nRequest {i}: {request_data['prompt']}")
                
                start_time = time.time()
                async with session.post(
                    "http://localhost:8080/v1/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    
                    if resp.status == 200:
                        response_data = await resp.json()
                        latency = time.time() - start_time
                        
                        print(f"✅ Response ({latency:.3f}s):")
                        print(f"   Text: {response_data['text']}")
                        print(f"   Tokens: {response_data['usage']['total_tokens']}")
                        print(f"   Finish: {response_data['finish_reason']}")
                    else:
                        error_text = await resp.text()
                        print(f"❌ Request failed: {resp.status} - {error_text}")
        
        except aiohttp.ClientConnectorError:
            print("❌ Could not connect to server. Make sure it's running on localhost:8080")
        except Exception as e:
            print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_inference_server())