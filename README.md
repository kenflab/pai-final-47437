# Camera-guided tabletop tidying

PAI final assignment / account ID: 47437

## Method

The robot uses two camera images, robot state and a language instruction.
The participant-trained pi0.5 R07 weights and the original inference class
are reused. Short execution chunks are followed by updated observations.
No saved action trajectory is played back and no new training is performed.

Public task: `libero_goal/put_the_bowl_on_the_plate_table_1`.
The initial state comes from the simulator reset with seed 42. The JSON init_id is a reset trial offset, not an index into saved benchmark init states.
The video is the first successful complete clip eligible for 1-3 minutes
at up to 2x playback. This selected demonstration is not a benchmark.

## Reproduce

Use an allocated Linux GPU, a local LIBERO installation and the original
inference dependencies. `environment_versions.json` records the simulator
environment. Model weights and third-party vendor packages are external
assets and are not included. The participant-trained weights have SHA-256:

`96c0a1836a231a811122da19a355a571f7cf332f1209b4c51944e8ce9cc76a1f`

The runtime directory must contain the original model, tokenizer and
processors under `model_weights/pi05_libero_finetuned_v044/`, and `vendor/`
when the inference dependencies were vendored. Supply local paths below:

```bash
python capture.py --runtime "$PAI_POLICY_RUNTIME" --policy-python "$PAI_POLICY_PYTHON" --libero-repo "$LIBERO_REPO" --output ./recording
```

`policy_impl.py` and its helper modules are the inference source used in
the recording. The original MyPolicy AST and preprocessing are preserved;
the HTTP endpoints are unnecessary for this local pipe-based recorder.

To build the presentation on a computer used for video editing:

```bash
python -m pip install -r requirements_presentation.txt
python -m zipfile -e recording/recording_return.zip ./recording_extracted
python presentation.py --input ./recording_extracted --account-id YOUR_ACCOUNT_ID --output ./submission
```

## Sources

- https://github.com/Lifelong-Robot-Learning/LIBERO
- https://github.com/huggingface/lerobot
- https://huggingface.co/lerobot/pi05_libero_finetuned_v044
- https://github.com/Physical-Intelligence/openpi

External models and dependencies retain their upstream terms. This repository
contains the course recorder, presentation builder and participant inference
source. It contains no private evaluation registry, evaluation infrastructure,
model weights, personal paths or participant logs. Generative AI assisted
the implementation and code organization.

The bundled PAI Report Sans is a renamed report-character subset of Noto Sans JP.
Its source, modifications and SIL Open Font License are in `assets/fonts/`.
