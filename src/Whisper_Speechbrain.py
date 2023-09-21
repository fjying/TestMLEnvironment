import time
import os
import whisper
from pydub import AudioSegment
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition
import pandas as pd

start_time = time.time()
# Set Primary Directory path
directory_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(directory_path)

# Change to Data Directory
os.chdir(os.path.join(directory_path, 'data', 'avengers'))

# Specify Sound File Input
soundfile_input_whole = 'loki_tony_93secs.wav'
soundfile_input_1 = 'tony.wav'
soundfile_input_2 = 'loki.wav'

# Set access token from HuggingFace
access_token = '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'

# Import Whisper Model and Classifer Models
whisper_model = whisper.load_model(name = os.path.join(directory_path, 'models/Whisper', 'large-v2.pt'))
# first time: download the model directly inside docker, need to run it directly inside docker instead of copying and pasting; Coding error occurs when they are copied and pasted
# Move the model from docker to the local folder
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.join(directory_path, 'models', 'EncoderClassifier'))
# verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.join(directory_path, 'models', 'SpeakerRecognition'))

classifier = EncoderClassifier.from_hparams(source=os.path.join(directory_path, 'models', 'EncoderClassifier'), savedir = os.path.join(directory_path, 'models', 'EncoderClassifier'))
verification = SpeakerRecognition.from_hparams(source=os.path.join(directory_path, 'models', 'SpeakerRecognition'), savedir = os.path.join(directory_path, 'models', 'SpeakerRecognition'))

# Translate text into Whisper
result = whisper_model.transcribe(soundfile_input_whole)

# Construct Audio Embedding of Each Person
signal, fs =torchaudio.load(soundfile_input_1)
embedding_tony = classifier.encode_batch(signal)
signal, fs =torchaudio.load(soundfile_input_2)
embedding_loki = classifier.encode_batch(signal)

# Speech Diarization
audio = AudioSegment.from_wav(soundfile_input_whole)
final_text_speechbrain = {}
for segment in result['segments']:
    audio_chunk=audio[segment['start']*1e3:segment['end']*1e3]
    audio_chunk.export('audio_chunk_temp.wav', format="wav")
    tony_score, tony_prediction = verification.verify_files("tony.wav", "audio_chunk_temp.wav")
    loki_score, loki_prediction = verification.verify_files("loki.wav", "audio_chunk_temp.wav")
    if tony_score >= loki_score:
        final_text_speechbrain[(segment['start'], 'tony')] = segment['text']
    else:
        final_text_speechbrain[(segment['start'], 'loki')] = segment['text']

final_text_df = pd.DataFrame()
final_text_df['timestamp'] = [i[0] for i in list(final_text_speechbrain.keys())]
final_text_df['speaker'] = [i[1] for i in list(final_text_speechbrain.keys())]
final_text_df['text'] = final_text_speechbrain.values()

# Change to output path
os.chdir(os.path.join(directory_path, 'output'))
final_text_df.to_csv('final_text.csv', index = False)

total_run_time = time.time()-start_time
print('Total Run Time', total_run_time)