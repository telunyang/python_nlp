'''測試 ollama 連線'''
import requests
def test_ollama_connection():
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("Ollama connection successful.")
            return True
        else:
            print(f"Failed to connect to Ollama. Status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return False

if __name__ == "__main__":
    test_ollama_connection()



# '''使用 ollama 官方套件'''
# from ollama import Client
# client = Client(
#   host='http://localhost:11434',
#   headers={'x-some-header': 'some-value'}
# )
# response = client.chat(model='gpt-oss:120b', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])