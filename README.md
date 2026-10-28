# Voice-based gender classification experiments

Academic experiments from 2023 comparing KNN, SVM, logistic regression, random forest and MLP classifiers using features extracted from audio. This repository accompanies the broader subtitle and dubbing research in [VidSubWizard](https://github.com/toandokhanh/vid-sub-wizard).

The labels describe the male/female categories supplied with the experimental data. They do not establish a speaker's gender identity or support broader demographic claims.

## Repository map

- `KNN/`, `SVM/`, `L/`, `RF/`: model-specific notebooks.
- `classification.py`: shared preprocessing and evaluation experiments; this script runs logistic-regression cross-validation and RF/MLP evaluation, rather than every model listed above.
- `examples/legacy_audio_classification.py`: tutorial-derived reference code, kept separate from test suites.
- `result/` and the tracked text outputs: historical experimental records.

Notebook outputs have been cleared from source control. The original notebooks and outputs were preserved locally during cleanup.

## Reproducing an experiment

The dataset itself is not distributed here. Supply WAV files with names beginning with their dataset label, followed by `_`, as parsed by `custom_database_import`. Use separate training and evaluation directories. Check speaker overlap between splits before reporting generalization to new speakers.

```sh
python3 classification.py -p data/audio -e data/eval -r result
```

`requirements.txt` records part of the original numerical environment. Audio/video conversion with `wav.py` also requires MoviePy and FFmpeg. No environment was installed or model retrained during the 2026 cleanup.

## Historical results and limitations

The original project reported mean accuracy of 0.8715 for SVM and 0.7773 for KNN. These are historical reports, not newly reproduced benchmarks. The repository does not yet document enough dataset provenance and speaker-level split information to make a stronger claim. The CLI now fixes the initial split seed; historical results have not been recomputed.

Several notebooks and the reference example retain material from an audio-classification tutorial attributed to Alberto. See [third-party notices](THIRD_PARTY_NOTICES.md); the original tutorial URL still needs to be recovered before selecting a repository-wide license.
