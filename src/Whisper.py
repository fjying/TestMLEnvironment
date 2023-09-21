#!/usr/bin/env python
# coding: utf-8
import os
import time
import whisper

def main():
    start_time = time.time()

    # Directory path
    directory_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    print(directory_path)

    # Specify Sound File Input
    soundfile_input_whole = 'loki_tony_93secs.wav'

    # Import Whisper Model and Classifer Models
    whisper_model = whisper.load_model(name = os.path.join(directory_path, 'models/Whisper', 'large-v2.pt'))

    # Change path to audio path
    os.chdir(os.path.join(directory_path, 'data', 'avengers'))

    # Translate text into Whisper
    result = whisper_model.transcribe(soundfile_input_whole)
    print('result', result)

    total_run_time = time.time()-start_time
    print('Total Run Time', total_run_time)

if __name__ == '__main__':
    main()