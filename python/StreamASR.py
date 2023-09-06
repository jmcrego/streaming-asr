import time
import logging
import numpy as np
from faster_whisper import WhisperModel

class StreamASR():
    def __init__(self, model_size='tiny', device='auto', compute_type='int8'):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logging.info('StreamASR ready')
        
    def __call__(self, audio, language=None, history=None, beam_size=5, task='transcribe'):
        ''' This functions calls whisper model to transcribe an audio wave. Audio is the audio wave in the form of a list containing block_size floats
        Params:
        language: speech language, 
        history: text context (prompt) for audio
        beam_size: decoding beam_size
        '''
        data = np.asarray(audio, dtype=np.float32)
        tic = time.time()
        segments, info = self.model.transcribe(data, language=language, task=task, beam_size=beam_size, vad_filter=True, word_timestamps=True, initial_prompt=history if history is not None else None)
        logging.info('transcription of {} floats took {:.2f} seconds'.format(len(audio), time.time()-tic))
        hyp = []
        for segment in segments:
            for word in segment.words:
                hyp.append([word.start, word.end, word.word])
                logging.info("\t{}\t{}\t{}".format(word.start,word.end,word.word))
        res = {'language': info.language, 'language_probability': info.language_probability, 'hyp': hyp}
        logging.info(res)
        return res

