import requests

def ridinghood(color, text):
    try:
        url = 'http://localhost:3000/api/gemini'
        myobj = {'type': 'generate_all', 'color': color, 'text': text}
        x = requests.post(url, json = myobj)
        print(x.text)
    except Exception as e:
        print("Failed to send request to Ridinghood API")

def main():
    pass

if __name__ == "__main":
    main()