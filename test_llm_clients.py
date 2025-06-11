#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai():
    """Test OpenAI client initialization and basic query."""
    print("Testing OpenAI client...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        import openai
        print(f"✅ OpenAI library version: {openai.__version__}")
        
        client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        
        # Test a simple query
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenAI query successful: {result}")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import OpenAI: {e}")
        return False
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return False

def test_anthropic():
    """Test Anthropic client initialization and basic query."""
    print("\nTesting Anthropic client...")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        import anthropic
        print(f"✅ Anthropic library version: {anthropic.__version__}")
        
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Anthropic client initialized successfully")
        
        # Test a simple query
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say hello"}]
        )
        
        result = response.content[0].text
        print(f"✅ Anthropic query successful: {result}")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Anthropic: {e}")
        return False
    except Exception as e:
        print(f"❌ Anthropic error: {e}")
        return False

def test_ollama():
    """Test Ollama connection."""
    print("\nTesting Ollama connection...")
    
    try:
        import requests
        
        url = "http://localhost:11434/api/generate"
        data = {
            "model": "llama3.2:latest",
            "prompt": "Say hello",
            "stream": False
        }
        
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()['response']
        print(f"✅ Ollama query successful: {result[:50]}...")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Ollama not running (connection refused)")
        return False
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return False

def main():
    print("LLM Client Test Script")
    print("=" * 50)
    
    # Test environment
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check for .env file
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ Found {env_file}")
    else:
        print(f"❌ No {env_file} file found")
    
    print("\n" + "=" * 50)
    
    # Run tests
    openai_ok = test_openai()
    anthropic_ok = test_anthropic()
    ollama_ok = test_ollama()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"OpenAI:    {'✅ Working' if openai_ok else '❌ Failed'}")
    print(f"Anthropic: {'✅ Working' if anthropic_ok else '❌ Failed'}")
    print(f"Ollama:    {'✅ Working' if ollama_ok else '❌ Failed'}")
    
    if not any([openai_ok, anthropic_ok, ollama_ok]):
        print("\n❌ All LLM clients failed!")
        sys.exit(1)
    else:
        print(f"\n✅ {sum([openai_ok, anthropic_ok, ollama_ok])}/3 clients working")

if __name__ == "__main__":
    main()