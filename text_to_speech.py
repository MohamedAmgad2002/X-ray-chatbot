import edge_tts

VOICE = 'en-US-AndrewNeural'


def tts(message):

    audio = edge_tts.Communicate(message, VOICE)
    return audio