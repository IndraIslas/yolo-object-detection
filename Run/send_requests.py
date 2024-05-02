import requests
import math


SERVER_IP = "localhost"



def closest_color(bgr):
    colors = {
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255),
        "Purple": (128, 0, 128),
        "Strong Pink": (147, 20, 255),
        "Light Pink": (203, 192, 255),
        "Orange": (0, 165, 255),
        "Light Blue": (255, 228, 225)
    }

    def euclidean_distance(bgr1, bgr2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(bgr1, bgr2)))

    closest = None
    min_distance = float('inf')
    for color, value in colors.items():
        distance = euclidean_distance(bgr, value)
        if distance < min_distance:
            min_distance = distance
            closest = color

    return closest

color_to_song = {
    "Red": "4uOKFydzAejjSFqYbv1XPt",
    "Green": "11DjZQEZ69EXLo77HVj6yW",
    "Blue": "4OtIszkveMtijHSIp3EP4d",
    "Yellow": "3AJwUDP919kvQ9QcozQPxg",
    "Purple": "54X78diSLoUDI3joC2bjMz",
    "Strong Pink": "4KROoGIaPaR1pBHPnR3bwC",
    "Light Pink": "1GeNWyZtCHbcp3ZWp8GTOO",
    "Orange": "7sNhXWrg9eW3qRqeuePaIC",
    "Light Blue": "6hHc7Pks7wtBIW8Z6A0iFq"
}

def ridinghood(color, text):
    try:
        url = f'http://{SERVER_IP}:3000/api/gemini'
        myobj = {'type': 'generate_all', 'color': color, 'text': text}
        x = requests.post(url, json = myobj)
        print(x.text)
    except Exception as e:
        print("Failed to send request to Ridinghood API")


def play_pause():
    try:
        url = f'http://{SERVER_IP}:3000/api/gemini'
        myobj = {'type': 'play_pause'}
        x = requests.post(url, json = myobj)
        print(x.text)
    except Exception as e:
        print("Failed to send request Play/Pause request")

def change_song(bgr_color):
    color_name = closest_color(bgr_color)
    song_id = color_to_song[color_name]
    print(f"Color {color_name} detected, changing song to {song_id}")
    try:
        url = f'http://{SERVER_IP}:3000/api/gemini'
        myobj = {'type': 'change_song', 'song_id': song_id}
        x = requests.post(url, json = myobj)
        print(x.text)
    except Exception as e:
        print("Failed to send request Change Song request")

def main():
    pass

if __name__ == "__main":
    main()