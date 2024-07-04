import os
import pygame

def play_midi(file):
    """ Play a MIDI file """
    # Use 'alsa' as the audio driver
    pygame.mixer.init(driver='alsa')

    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)  # Use pygame's delay for better timing

if __name__ == '__main__':
    midi_file = 'Cids.mid'
    play_midi(midi_file)
