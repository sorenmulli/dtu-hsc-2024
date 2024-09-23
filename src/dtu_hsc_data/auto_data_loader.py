import os   
import torchaudio

def auto_dataloader(task_types, task_levels):
    """
    task_types: 1 - 2
    task_levels: 1 - 7
    """

def process_data_to_torch_Dataset(task_types, task_levels, sample_rate=16000):
    recorded_samples = []
    clean_samples = []
    text_samples = []

    for task_type in task_types:
        for task_level in task_levels:
            current_task_name = 'Task_{task_type}_Level_{task_level}'
            current_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', current_task_name)
            
            recorded_samples += os.listdir(os.path.join(recorded_samples, 'Recorded'))
            clean_samples += os.listdir(os.path.join(current_path, 'Clean'))
            text_samples += os.path.join(current_path, f'Task_{task_type}_Level_{task_level}_text_samples.txt')

def evaluate_torch_to_wav(audio_processed, eval_path):
    audio_processed(torchaudio.save(eval_path, audio_processed))

def process_and_transcribe(model, audio_file):
    # Load and resample the audio file to 16kHz
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    # Convert the audio signal from floating point to 16-bit PCM
    audio_int16 = (audio * np.iinfo(np.int16).max).astype(np.int16)
    # Perform speech-to-text with model
    transcription = model.stt(audio_int16)
    return transcription