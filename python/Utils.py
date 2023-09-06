import logging
import numpy as np
import soundfile as sf
import sounddevice as sd

def play(data, sample_rate=16000, wait=True):
    """ 
    Play audio data

    Params:
    data: a numpy array containing the audio data
    sample_rate: the sample rate of the audio data
    wait: whether the function returns immediately or waits until the end of the sound
    """
    logging.info('PLAY data = {}'.format(data.shape))
    sd.play(data, sample_rate)
    if wait:
        sd.wait()
    logging.info('PLAYED')

def save(data, file_name, sample_rate=16000):
    """ 
    Save each audio data split (transcripts) into files

    Params:
    data: a numpy array containing the audio data
    audio: list of n elements each being a numpy array of blocksize audio data
    sample_rate: the sample rate of the audio wave
    file_name: the output file name (available extensions: 'AIFF', 'AU', 'AVR', 'CAF', 'FLAC', 'HTK', 'SVX', 'MAT4', 'MAT5', 'MPC2K', 'MP3', 'OGG', 'PAF', 'PVF', 'RAW', 'RF64', 'SD2', 'SDS', 'IRCAM', 'VOC', 'W64', 'WAV', 'NIST', 'WAVEX', 'WVE', 'XI')
    """
    logging.info('SAVE data = {}'.format(data.shape))
    sf.write(file_name, data, sample_rate)
    logging.info('SAVED')

def is_silence(data, silence_threshold):
    """ data is a numpy.ndarray (block_size, 1) dtype=float32 containing audio wave """
    return np.sqrt(np.mean(np.square(data))) < silence_threshold
