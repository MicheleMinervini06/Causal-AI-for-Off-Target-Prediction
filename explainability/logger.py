from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class ExperimentLogger:
    """Persist explainability artifacts in an HDF5 log file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.file_path, mode="a")

    def _write(self, group: str, name: str, values: np.ndarray) -> None:
        grp = self._h5.require_group(group)
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=np.asarray(values))

    def log_embedding(self, name: str, values: np.ndarray) -> None:
        self._write("embeddings", name, values)

    def log_attention(self, name: str, values: np.ndarray) -> None:
        self._write("attention", name, values)

    def log_attribution(self, name: str, values: np.ndarray) -> None:
        self._write("attribution", name, values)

    def close(self) -> None:
        if self._h5:
            self._h5.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()
