from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Union

import numpy as np

from .audio import load_audio


@dataclass
class TaskExample:
    name: str
    recorded_audio_path: Path
    clean_audio_path: Path
    transcription: str

    def get_recorded(self) -> np.ndarray:
        return load_audio(self.recorded_audio_path)

    def get_clean(self) -> np.ndarray:
        return load_audio(self.clean_audio_path)


def _get_level_data(level_path: Path, test_split: bool) -> Generator[TaskExample, None, None]:
    transcription_path = next(level_path.glob("*.txt"))
    for example in transcription_path.read_text().splitlines():
        recorded_name, transcription = example.split("\t")

        recorded_path = level_path / "Recorded" / recorded_name
        if not recorded_path.exists():
            print(f"WARNING: No audio for {recorded_name}, skipping")
            continue

        yield TaskExample(
            name=recorded_name.replace("_recorded", "").replace(".wav", ""),
            recorded_audio_path=recorded_path,
            clean_audio_path=level_path / "Clean" / recorded_name.replace("_recorded", "_clean"),
            transcription=transcription,
        )


def get_task_data(data_path: Union[str, Path], test_split=False) -> dict[str, list[TaskExample]]:
    """
    Get all given challenge data without loading the audio files in a dict mapping level to
    list of examples.
    """
    if test_split:
        data_path = Path(data_path) / "Test"
    task_data = {
        level_path.name: list(_get_level_data(level_path, test_split))
        for level_path in sorted(Path(data_path).glob("Task*Level*"))
    }
    if not task_data:
        raise FileNotFoundError(f"No data found in {data_path}")
    return task_data
