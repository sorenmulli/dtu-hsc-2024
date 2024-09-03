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


def _get_level_data(level_path: Path) -> Generator[TaskExample, None, None]:
    for example in (level_path / f"{level_path.name}_text_samples.txt").read_text().splitlines():
        recorded_name, transcription = example.split("\t")

        yield TaskExample(
            name=recorded_name.replace("_recorded", "").replace(".wav", ""),
            recorded_audio_path=level_path / "Recorded" / recorded_name,
            clean_audio_path=level_path / "Clean" / recorded_name.replace("_recorded", "_clean"),
            transcription=transcription,
        )


def get_task_data(data_path: Union[str, Path]) -> dict[str, list[TaskExample]]:
    """
    Get all given challenge data without loading the audio files in a dict mapping level to
    list of examples.
    """
    task_data = {
        level_path.name: list(_get_level_data(level_path))
        for level_path in sorted(Path(data_path).glob("Task*Level*"))
    }
    if not task_data:
        raise FileNotFoundError(f"No data found in {data_path}")
    return task_data
