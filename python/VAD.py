import logging
from typing import NamedTuple
from faster_whisper.vad import get_speech_timestamps


class VadOptions(NamedTuple):
    """
    VAD options

    Attributes:
      threshold: Speech threshold.
                Silero VAD outputs speech probabilities for each audio chunk,
                probabilities ABOVE this value are considered as SPEECH.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds.
                Chunks longer than max_speech_duration_s will be split at the timestamp
                of the last silence that lasts more than 100ms (if any), to prevent aggressive cutting.
                Otherwise, they will be split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms before separating it
      window_size_samples: Audio chunks of window_size_samples size are fed to the silero VAD model.
                WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
                Values other than these may affect model performance!!
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
    """    
    threshold: float = 0.5 #0.5
    min_speech_duration_ms: int = 250 #250
    max_speech_duration_s: float = float("inf") #float("inf")
    min_silence_duration_ms: int = 500 #2000
    window_size_samples: int = 1024 #1024
    speech_pad_ms: int = 100 #400

class VAD():

    def __init__(self, audio, start, end, sample_rate, min_silence_sec, padding_sec):
        self.start = start
        self.end = end
        self.sample_rate = sample_rate
        self.min_silence_sec = min_silence_sec
        self.padding_sec = padding_sec
        self.speech_chunks = get_speech_timestamps(audio[start:end], VadOptions())
        for i in range(len(self.speech_chunks)):
            ''' add audio_start to make start/end indexs refer to the begining of self.audio (rather than audio_start) '''
            self.speech_chunks[i]['start'] += start
            self.speech_chunks[i]['end'] += start
        logging.info('vad: found {} speech chunks, audio_start={} {} audio_end={}'.format(len(self.speech_chunks), start, self.speech_chunks, end))

    def __len__(self):
        return len(self.speech_chunks)

    def adjust_speech(self):
        assert(len(self))
        speech_start = self.pad_to_end(self.start, self.speech_chunks[0]['start']) 
        speech_end = self.pad_to_start(self.speech_chunks[-1]['end'], self.end)
        logging.info('vad: speech=[{}, {})'.format(speech_start, speech_end))
        return speech_start, speech_end

    def ending_silence(self):
        ending_silence = (self.end - self.speech_chunks[-1]['end']) / self.sample_rate >= self.min_silence_sec
        logging.info('vad: ending_silence={}'.format(ending_silence))
        return ending_silence

    def pad_to_start(self, start, end):
        if start + int(self.padding_sec*self.sample_rate) < end:
            return start + int(self.padding_sec*self.sample_rate)
        return (start+end)//2

    def pad_to_end(self, start, end):
        if end - int(self.padding_sec*self.sample_rate) > start:
            return end - int(self.padding_sec*self.sample_rate)
        return (start+end)//2
