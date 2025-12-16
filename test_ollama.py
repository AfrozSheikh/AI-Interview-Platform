import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:1b",
        "prompt": "Hello, are you working? Reply with JSON {\"status\": \"ok\"}.",
        "format": "json",
        "stream": False
    }
    
    print(f"Testing connectivity to {url}...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Success! Response:")
            print(response.json())
        else:
            print(f"Failed with status code {response.status_code}")
            print(response.text)
            with open("ollama_debug.log", "w") as f:
                f.write(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure Ollama is running (ollama serve) and the model is pulled (ollama pull llama3.1:latest).")

if __name__ == "__main__":
    test_ollama()
