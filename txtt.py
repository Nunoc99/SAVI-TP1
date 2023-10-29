import pygame
import gtts
import time
import threading
import os
# from playsound import playsound
# import mpg123


def text_to_speech(text, language='en', speed='normal'):
  """Converts text to speech using the gTTS library.

  Args:
    text: The text to convert to speech.
    language: The language of the text.
    speed: The speed of the speech.

  Returns:
    An audio file containing the converted speech.
  """

  # Create a gTTS object.
  tts = gtts.gTTS(text='Hello '+text, lang=language, slow=False if speed == 'normal' else True)

  # Save the audio file.
  audio_file = 'Audio/speech' + text + '.mp3'
  tts.save(audio_file)

  return audio_file

#playsound('speech.mp3')

#time.sleep(0.1)


def txt_speech(audio_file):
  pygame.init()
    
  pygame.mixer.music.load(audio_file)
    
  pygame.mixer.music.play()



if __name__ == "__main__":
    
    name = 'Jose'
    text_to_speech(name)
    txt_speech('Audio/speech' + name + '.mp3')
    for i in range(1000000):
       print(i)


    

