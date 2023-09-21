import whisper
import os
import time
from bisect import bisect_left
from pyannote.audio import Pipeline


start_time = time.time()

# Directory path
directory_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(directory_path)

os.chdir(os.path.join(directory_path, 'data'))

# Load Local Whisper Model
whisper_model = whisper.load_model(name = os.path.join(directory_path, 'models/Whisper','large-v2.pt'))

# Load PyAnnote Offline
diarization_pipeline = Pipeline.from_pretrained(os.path.join(directory_path, 'models/Pyannote', 'Diarization', 'config.yaml'))

# (1) Apply Whisper Model First to translate audio to text
# (2) Use PyAnnotate to Attribute Specific Audio Segment By Time to Each Speaker
def whisper_pyannotate_audio_text(soundfile_input,
                                  whisper_model=whisper_model, pyannote_pipeline=diarization_pipeline):
    # Transcribe audio to text
    result = whisper_model.transcribe(soundfile_input)
    diarization = pyannote_pipeline(soundfile_input)

    timestamp_speaker = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        timestamp_speaker[turn.start] = speaker

    timestamp_list = list(timestamp_speaker.keys())
    speaker_list = list(timestamp_speaker.values())

    # keep correct order
    # insert time zero in the beginning if it is not be recognized
    if timestamp_list[0] > 0:
        updict = {0: speaker_list[0]}
        updict.update(timestamp_speaker)
        timestamp_speaker = updict
    timestamp_list.insert(0, 0)
    speaker_list.insert(0, speaker_list[0])

    # Split text to each segment by speaker using speaker diarization from Pyannotate
    final_text = {}
    for segment in result['segments']:
        timestamp_idx = bisect_left(timestamp_list, segment['start'])
        if timestamp_idx > 0:
            timestamp_idx -= 1
        speaker = timestamp_speaker[timestamp_list[timestamp_idx]]
        final_text[(segment['start'], speaker)] = segment['text']
    return final_text, result, timestamp_list, speaker_list

final_text, whisper_suits, timestamp_list_suits, speaker_list_suits = whisper_pyannotate_audio_text(soundfile_input = 'twospeakers_consecutive.wav')
print(final_text)

total_run_time = time.time()-start_time
print('Total Run Time', total_run_time)