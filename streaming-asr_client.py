import sys
import logging
import argparse
from python.StreamMic import StreamMic

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script reads audio data from the available microphone and performs ASR.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group_server = parser.add_argument_group("Server")
    group_server.add_argument('--url_api', type=str, help='Address where ASR server is located', default='http://10.25.1.145:5000/transcribe')
    group_server.add_argument('--channels', type=int, help='Channels (1:mono, 2:stereo)', default=1)
    group_server.add_argument('--block_size', type=int, help='Amount of audio data captured', default=1024)
    group_server.add_argument('--sample_rate', type=int, help='Sample rate', default=16000)
    group_server.add_argument('--beam_size', type=int, help='Decoding beam size', default=5)
    group_server.add_argument('--language', type=str, help='Force language when transcribing', default=None)
    group_server.add_argument('--task', type=str, help='Task to perform: transcribe or translate', default='transcribe')

    group_client = parser.add_argument_group("Client")
    group_client.add_argument('--sleep', type=int, help='ASR requests performed every this amount of time (ms)', default=500)
    group_client.add_argument('--padding', type=int, help='Speech intervals are padded by this amount of time (ms) each side', default=200)

    group_client_endsilence = parser.add_argument_group("  ===== [endsilence] transcripts when long silences detected =====")
    group_client_endsilence.add_argument('--silence', type=int, help='Minimum silence (ms) to consider a transcript [endsilence]', default=500)

    group_client_endchars = parser.add_argument_group("  ===== [endchars] transcripts when a token of current hypothesis is ended by (punctuation) chars =====")
    group_client_endchars.add_argument('--endchars', type=str, help='Recognised tokens ended by any of these chars produce a transcript [endchars]', default=',.!?؟،')
    group_client_endchars.add_argument('--skip_ini', type=int, help='Skip these initial tokens when searching [endchars] transcript', default=3)
    group_client_endchars.add_argument('--skip_end', type=int, help='Skip these ending tokens when searching [endchars] transcript', default=3)

    group_other = parser.add_argument_group("Other")
    group_other.add_argument('--odir', type=str, help='Save transcripts in given directory', default=None)
    group_other.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    logging.basicConfig(
        format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', 
        datefmt='%Y-%m-%d_%H:%M:%S', 
        level=getattr(logging, 'WARNING' if not args.debug else 'INFO'), 
        filename=None)    

    m = StreamMic(
        args.task, 
        args.beam_size, 
        args.channels, 
        args.block_size, 
        args.sample_rate, 
        args.sleep, 
        args.url_api, 
        args.language, 
        args.silence, 
        args.endchars, 
        args.skip_ini, 
        args.skip_end, 
        args.padding)
    print('Listening... use [Ctrl+c] to terminate streaming', file=sys.stderr)
    try:
        m()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: recording finished', file=sys.stderr)
        if args.odir is not None:
            m.save_transcripts(args.odir)
    m.print_stats()


        

    




