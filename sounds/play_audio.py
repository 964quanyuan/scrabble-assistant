import winsound

def greet():
    winsound.PlaySound('sounds/hello_mofo.wav', winsound.SND_FILENAME)

def rick_entry():
    winsound.PlaySound('sounds/basement.wav', winsound.SND_FILENAME)
    winsound.PlaySound('sounds/reality.wav', winsound.SND_FILENAME)

def playsound(path):
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

def ramble(choice):
    playsound(f'sounds/ramble{choice}.wav')