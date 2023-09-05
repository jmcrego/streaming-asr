import os
import sys
import time
import copy
import logging
import requests
import numpy as np
import sounddevice as sd
from collections import defaultdict
from Utils import save
from Hyp import Hyp
from VAD import VAD

class StreamMic():

    def __init__(
        self, 
        task='transcribe', 
        beam_size=5, 
        channels=1, 
        block_size=1024, 
        sample_rate=16000, 
        sleep_ms=500, 
        url_api='http://127.0.0.1:5000/transcribe', 
        language=None, 
        silence_ms=500, 
        end_chars='.!?', 
        skip_ini=3, 
        skip_end=2, 
        padding_ms=200):

        self.mic = sd.default.device[0] if channels == 1 else sd.default.device[1]
        self.task = task
        self.beam_size = beam_size
        self.channels = channels
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.sleep_ms = sleep_ms
        self.url_api = url_api
        self.language = language
        self.end_chars = end_chars
        self.skip_ini = skip_ini
        self.skip_end = skip_end
        self.timeout = 10
        self.silence_sec = silence_ms / 1000
        self.padding_sec = padding_ms / 1000
        self.stats = defaultdict(list)

    def __call__(self):
        self.audio = np.empty(0, dtype=np.float32)
        self.transcripts = []

        def callback(indata, frames, time, status):
            """
            This function is employed by sd.InputStream to store the wave continuously read from the mic into self.audio.
            The function is called whenever new block_size floats are available read from the mic.
            Params:
            indata: numpy.ndarray (block_size, 1) dtype=float32 containing audio captured from microphone. 
            frames: indicate the number of floats (same as block_size)
            time: time indication
            status: exit indication
            """
            if status:
                logging.error('callback error: '.format(status))
            self.audio = np.concatenate((self.audio, indata.squeeze()), dtype=np.float32)

        with sd.InputStream(
            device=self.mic, 
            channels=self.channels, 
            callback=callback, 
            blocksize=self.block_size, 
            samplerate=self.sample_rate):
        
            audio_start = 0
            """ infinite loop (stopped using [Ctrl+c]) """
            tic = time.time()
            while True:

                time_spent_ms = int((time.time()-tic)*1000)
                if time_spent_ms < self.sleep_ms:
                    time_sleep_ms = self.sleep_ms-time_spent_ms
                    self.stats['time_SLEEP'].append(time_sleep_ms/1000)
                    sd.sleep(time_sleep_ms)
                else:
                    time_delay_ms = time_spent_ms-self.sleep_ms
                    self.stats['time_DELAY'].append(time_delay_ms/1000)

                tic = time.time()
                audio_start = self.analyse_audio(audio_start, len(self.audio))

    def analyse_audio(self, audio_start, audio_end):
        """ 
        Analyses self.audio[self.audio_start:audio_end] looking for speech transcripts
        Params:
        audio_start : analysis initial point of self.audio
        audio_end : analysis ending point of self.audio
        Returns:
        The new position of audio_start
        """
        logging.info('analyse audio[{}, {}) => {:.2f} sec'.format(audio_start, audio_end, (audio_end-audio_start)/self.sample_rate))

        """ compute speech chunks using VAD """
        tic_VAD = time.time()
        vad = VAD(self.audio, audio_start, audio_end, self.sample_rate, self.silence_sec, self.padding_sec)
        self.stats['time_VAD'].append(time.time() - tic_VAD)

        if len(vad) == 0: 
            return vad.pad_to_end(audio_start,audio_end)

        """ find_speech using VAD results to restrict the ASR request """
        speech_start, speech_end = vad.adjust_speech()

        """ ASR request over self.audio[audio_start, audio_end] """
        tic_request = time.time()
        hyp = self.asr_request(speech_start, speech_end)
        self.stats['time_ASR'].append(time.time() - tic_request)
        self.stats['time_speech'].append((speech_end - speech_start)/self.sample_rate)

        if len(hyp) == 0:
            return audio_start

        """ save hyp as transcript if the audio analysed is ended by a silence """
        if vad.ending_silence():
            self.add_transcript(hyp, 'endsilence')
            return hyp.end

        """ save intermediate_hyp as transcript if contains a punctuation mark """
        hyp_prefix = self.has_endchar(hyp, speech_start)
        if hyp_prefix is not None:
            self.add_transcript(hyp_prefix, 'endchars')
            return hyp_prefix.end

        return audio_start

    def add_transcript(self, hyp, end_by):
        self.transcripts.append({'start': hyp.start, 'end': hyp.end, 'str': str(hyp), 'language': hyp.language})
        print("***: {} ({}) [{}, {}) time=[{:.2f}, {:.2f}] len={:.2f} sec [{}]".format(str(hyp), hyp.language, hyp.start, hyp.end, hyp.start/self.sample_rate, hyp.end/self.sample_rate, (hyp.end-hyp.start)/self.sample_rate, end_by))

    def has_endchar(self, hyp, speech_start):
        """
        This function looks for a previous hyp with an intermediate punctuation mark. Previous hyps are those starting in the same speech_start point as the current one.
        Parmas:
        speech_start: the starting point in self.audio of the current hypothesis
        skip_ini: the number of initial tokens in previous hyp where the punctuation is not searched (this is to skip wrong ASR hyps because of lack of context)
        skip_end: the number of fiinal tokens in previous hyp where the punctuation is not searched (this is to skip wrong ASR hyps because of lack of context)
        Returns:
        hyp: the Hyp hypothesis ended by the punctuation mark
        """
        n = hyp.has_endchars(self.end_chars, self.skip_ini, self.skip_end)
        if n is not None:
            new_hyp = copy.deepcopy(hyp)
            new_hyp.remove_after_token_n(n, self.sample_rate)
            return new_hyp
        return None

    def asr_request(self, start, end):
        """ 
        Performs a request to the asr server using the wave form contained in self.audio[start:end] data
        Params:
        start: initial point of speech in self.audio to transcribe
        end: ending point of speech in self.audio to transcribe 
        Returns:
        hyp: class containing the transcript hypothesis
        """
        history = self.transcripts[-1]['str'] if len(self.transcripts) else None
        try:
            response = requests.post(
                self.url_api, 
                json={
                    "audio": self.audio[start:end].tolist(), 
                    "language": self.language, 
                    "history": history, 
                    "beam_size": self.beam_size, 
                    "task": self.task}, 
                headers={"Content-Type": "application/json"}, 
                timeout=self.timeout)
        except requests.exceptions.Timeout: 
            logging.error("POST Request Error (Timeout):", e)
            raise SystemExit(e)
        except requests.exceptions.ConnectionError as e:
            logging.error("POST Request Error (ConnectionError):", e)
            raise SystemExit(e)
        except requests.exceptions.TooManyRedirects: 
            logging.error("POST Request Error (TooManyRedirects):", e)
            raise SystemExit(e)
        except requests.exceptions.HTTPError as e:
            logging.error("POST Request Error (HTTPError):", e)
            raise SystemExit(e)
        except requests.exceptions.RequestException as e: 
            logging.error("POST Request Error (RequestException):", e)
            raise SystemExit(e)

        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError:
            logging.error("Response body did not contain valid json:", e)
            raise SystemExit(e)

        return Hyp(response_json, start, end)

    def save_transcripts(self, odir):
        if os.path.exists(odir):
            os.remove(odir+'/*.{mp3,txt}')
        else:
            os.mkdir(odir)
        with open(odir + '/transcripts.txt', 'w') as fdesc:
            for i,t in enumerate(self.transcripts):
                start = t['start']
                end = t['end']
                tstart = "{:.2f}".format(start / self.sample_rate)
                tend = "{:.2f}".format(end / self.sample_rate)
                fname = odir + '/audio.' + str(i) + '_' + tstart + '_' + tend + '.mp3'
                save(self.audio[start:end], fname)
                fdesc.write("{}\t{}\t{}\t{}\n".format(i, tstart, tend, t['str']))

    def print_stats(self):
        print("Time stats (seconds)", file=sys.stderr)
        for name, l in self.stats.items():
            print("{}\t{:.2f}\t{}\t{:.2f}".format(name, sum(l), len(l), sum(l)/len(l)), file=sys.stderr)


    




