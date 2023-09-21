import os
import whisper


# Directory path
directory_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(directory_path)

# Specify Sound File Input
soundfile_input_whole = 'loki_tony_93secs.wav'
soundfile_input_1 = 'tony.wav'
soundfile_input_2 = 'loki.wav'

# Set access token from HuggingFace
access_token = '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'

# Import Whisper Model and Classifer Models
whisper_model = whisper.load_model(name = os.path.join(directory_path, 'models', 'large-v2.pt'))

# Change path to audio path
os.chdir(os.path.join(directory_path, 'data', 'avengers'))

# Translate text into Whisper
result = whisper_model.transcribe(soundfile_input_whole)
print('after whisper')
