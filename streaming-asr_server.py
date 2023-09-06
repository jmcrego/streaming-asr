import logging
import argparse
from datetime import datetime
from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
from python.StreamASR import StreamASR

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script launches an ASR (Whisper) server behind a REST api.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_size', type=str, help='Model size (tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2)', default='tiny')
    parser.add_argument('--device', type=str, help='Device: cpu, cuda, auto', default='auto')
    parser.add_argument('--compute_type', type=str, help='Compute type', default='int8')
    parser.add_argument('--port', type=int, help='Port used in local server', default=5000)
    parser.add_argument('--host', type=str, help='Host used (use 0.0.0.0 to allow distant access, otherwise use 127.0.0.1)', default='0.0.0.0')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO', None), filename='./log.{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    asr = StreamASR(args.model_size, args.device, args.compute_type)

    app = Flask(__name__)
    @app.route('/transcribe', methods=['POST'])
    def send_data():
        content = request.json  # Assuming the client sends JSON data
        response = { 
            'transcript': asr( 
                content['audio'],
                content['language'],
                content['history'],
                int(content['beam_size']),
                content['task'])
            }
        return jsonify(response)

    if __name__ == '__main__':
        app.run(host=args.host, port=args.port)



