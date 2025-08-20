#!/usr/bin/env python3
"""
Enhanced xAI API connection test with L4 integration
Tests the basic API connection and L4-style reasoning capabilities
"""

import requests
import json
import time
from xai_config import (
    XAI_API_KEY, XAI_API_URL, XAI_MODEL, 
    XAI_MAX_TOKENS, XAI_TEMPERATURE, XAI_TOP_P,
    XAI_L4_SYSTEM_PROMPT
)

def test_network_connectivity():
    """Test basic network connectivity to xAI"""
    print("🌐 Testing network connectivity...")
    
    # Test basic DNS resolution
    try:
        import socket
        host = "api.x.ai"
        ip = socket.gethostbyname(host)
        print(f"✅ DNS resolution successful: {host} -> {ip}")
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")
        return False
    
    # Test basic HTTP connectivity
    try:
        response = requests.get("https://api.x.ai", timeout=10)
        print(f"✅ HTTP connectivity successful: Status {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP connectivity failed: {e}")
        return False

def test_basic_connection():
    """Test the xAI API connection with enhanced error handling"""
    print("🔌 Testing xAI API Connection...")
    print(f"Using model: {XAI_MODEL}")
    print(f"API URL: {XAI_API_URL}")
    print(f"API Key: {XAI_API_KEY[:10]}...{XAI_API_KEY[-10:] if len(XAI_API_KEY) > 20 else '***'}")
    print(f"Max Tokens: {XAI_MAX_TOKENS}")
    print(f"Temperature: {XAI_TEMPERATURE}")
    
    # Test network connectivity first
    if not test_network_connectivity():
        print("⚠️  Network connectivity issues detected. Please check your internet connection.")
        return
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        "model": XAI_MODEL,
        "stream": False,
        "temperature": XAI_TEMPERATURE,
        "max_tokens": XAI_MAX_TOKENS,
        "top_p": XAI_TOP_P
    }
    
    try:
        print("\n📤 Sending basic test request...")
        print(f"Request payload: {json.dumps(payload, indent=2)}")
        
        # Add timeout and retry logic
        for attempt in range(3):
            try:
                response = requests.post(
                    XAI_API_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=30
                )
                break
            except requests.exceptions.Timeout:
                if attempt < 2:
                    print(f"⏰ Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Request timed out after 3 attempts")
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            finish_reason = result.get('choices', [{}])[0].get('finish_reason', 'Unknown')
            usage = result.get('usage', {})
            
            print(f"✅ SUCCESS!")
            print(f"🤖 Response: {content}")
            print(f"🔍 Finish Reason: {finish_reason}")
            print(f"📊 Token Usage: {usage}")
            
            # Check if content is empty but tokens were used
            if not content and usage.get('completion_tokens', 0) > 0:
                print("⚠️  WARNING: Content is empty but completion tokens were used!")
                print("💡 This might indicate content filtering or token counting issues.")
            
            return True
            
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

def test_l4_reasoning():
    """Test L4-style reasoning capabilities"""
    print("\n🧠 Testing L4-style reasoning capabilities...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }
    
    # L4-style reasoning prompt
    l4_prompt = """Consider the following philosophical problem:

A consciousness researcher claims that subjective experience cannot be reduced to physical processes, arguing that there's an "explanatory gap" between objective brain states and subjective qualia. A materialist responds that this gap is merely epistemic, not ontological - we just don't understand the mechanisms yet.

Analyze this debate by:
1. Identifying the core philosophical positions
2. Examining the strength of each argument
3. Considering potential resolutions or middle grounds
4. Reflecting on what this tells us about consciousness research

Provide a thoughtful, nuanced analysis that demonstrates deep reasoning."""
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": XAI_L4_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": l4_prompt
            }
        ],
        "model": XAI_MODEL,
        "stream": False,
        "temperature": XAI_TEMPERATURE,
        "max_tokens": XAI_MAX_TOKENS,
        "top_p": XAI_TOP_P
    }
    
    try:
        print("📤 Sending L4 reasoning test...")
        
        response = requests.post(
            XAI_API_URL, 
            headers=headers, 
            json=payload, 
            timeout=60  # Longer timeout for complex reasoning
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            finish_reason = result.get('choices', [{}])[0].get('finish_reason', 'Unknown')
            usage = result.get('usage', {})
            
            print(f"✅ L4 Test SUCCESS!")
            print(f"🔍 Finish Reason: {finish_reason}")
            print(f"📊 Token Usage: {usage}")
            print(f"\n🤖 L4 Response Preview:")
            print(f"{content[:200]}{'...' if len(content) > 200 else ''}")
            
            # Show full response structure
            print(f"\n📋 Full Response Structure:")
            print(json.dumps(result, indent=2))
            
            return True
            
        else:
            print(f"❌ L4 Test ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ L4 Test Exception: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting xAI Connection and L4 Testing Suite...\n")
    
    # Test basic connection first
    basic_success = test_basic_connection()
    
    if basic_success:
        # Test L4 reasoning capabilities
        l4_success = test_l4_reasoning()
        
        print(f"\n📊 Test Summary:")
        print(f"   Basic Connection: {'✅ PASS' if basic_success else '❌ FAIL'}")
        print(f"   L4 Reasoning: {'✅ PASS' if l4_success else '❌ FAIL'}")
        
        if basic_success and l4_success:
            print(f"\n🎉 All tests passed! xAI is ready for L4 integration.")
        else:
            print(f"\n⚠️  Some tests failed. Please check the errors above.")
    else:
        print(f"\n❌ Basic connection failed. Cannot proceed with L4 testing.") 