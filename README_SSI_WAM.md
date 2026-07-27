# SSI-WAM Notes

This document tracks the current SSI-WAM implementation plan and the preprocessing utilities added on top of FastWAM. It should be updated whenever new SSI branches, dataset adapters, model heads, or training losses are implemented.

## Current status

The repository currently includes three independent offline label preprocessors
for LIBERO. The ATM-style bbox-label preprocessor is:

```text
scripts/preprocess_libero_bbox.py
```

And a depth-only variant:

```text
scripts/preprocess_libero_depth.py
```

And a trajectory-only variant:

```text
scripts/preprocess_libero_motion.py
```

These tools generate episode-level supervision labels. Labels are saved over
the full demonstration timeline, and the training dataset slices them according
to the policy horizon and video stride. Use
`preprocess_libero_bbox.py` for raw ATM-style boxes and confidence scores,
`preprocess_libero_depth.py` for monocular depth maps, and
`preprocess_libero_motion.py` for point trajectories.

## Depth-only label variant

`scripts/preprocess_libero_depth.py` keeps the FastWAM/LIBERO episode loader,
manifest writing, and depth visualization utilities, but is otherwise
depth-only. It does not compute bbox or trajectory labels.

The depth variant writes:

```text
depth:      [C, T_label, H_label, W_label]
depth_conf: [C, T_label, H_label, W_label]
```

For DA3 and Video Depth Anything, `H_label = W_label = --image-size` (224 by
default): the normalized full-resolution depth map is saved directly without
downsampling to an SSI grid. `T_label` is controlled by `--frame-stride`. The
default backend is Video Depth Anything with the Large relative-depth model:

```bash
--depth-backend video_depth_anything \
--video-depth-anything-repo-dir ./third_party/Video-Depth-Anything \
--video-depth-anything-checkpoint ./third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth \
--video-depth-anything-encoder vitl \
--video-depth-anything-input-size 518 \
--image-size 224
```

DA3 remains available as an optional per-frame backend:

```bash
--depth-backend da3 --da3-model-id depth-anything/DA3MONO-LARGE
```

VDA reads videos resized to 224x224 by default (`--image-size 224`) and saves
224x224 depth labels. `--video-depth-anything-input-size` controls the model's
internal inference size and does not change the saved label resolution.

Depth-only visualizations contain two columns: the resized RGB input and the
final depth label saved to the cache.

## Trajectory-only label variant

`scripts/preprocess_libero_motion.py` keeps the FastWAM/LIBERO episode loader,
manifest writing, and trajectory visualization utilities, but is otherwise
motion-only. It does not compute depth or bbox labels.

The motion variant writes:

```text
motion_points:       [C, T_episode, N, 2]
motion_visibility:   [C, T_episode, N]
motion_point_source: [C, N]
```

The default backend is the SSI/ATM-style CoTracker path:

```bash
--motion-backend cotracker3
```

It uses random query points plus a double grid, filters low-variance random
tracks, repeats the remaining dynamic points with small spatial noise, and
tracks again. CoTracker2 is also supported:

```bash
--motion-backend cotracker2
```

## ATM-style bbox label variant

`scripts/preprocess_libero_bbox.py` keeps the FastWAM/LIBERO episode loader,
manifest writing, and bbox visualization utilities, but is otherwise bbox-only.
It does not compute depth or trajectory labels.

By default it selects one demo for every `(suite, task)` pair and saves both
the Grounded SAM 2 cache and its visualization (`--max-demos-per-task 1` and
`--vis-num-demos-per-task 1`). Use `--max-demos-per-task 0` to process every
matching demo.

The bbox teacher is the local torch-2.7-compatible GroundingDINO copy under:

```text
third_party/grounding_dino_torch27/grounding_dino
```

The default checkpoint is also local:

```text
third_party/grounding_dino_torch27/gdino_checkpoints/groundingdino_swint_ogc.pth
```

The detector follows ATM's bbox-only behavior: it applies the ATM prompt,
filters GroundingDINO boxes by `--box-threshold` / `--text-threshold`, and saves
raw boxes without running SAM2 masks.

Task prompts are resolved from the local copy of ATM's
`get_task_to_prompt_dict()` in:

```text
scripts/atm_bbox_prompts.py
```

Since the FastWAM lerobot metadata stores natural
language tasks, the bbox script maps each episode task back to the ATM prompt
dictionary key. For example:

```text
libero_10 / turn on the stove and put the moka pot on it -> the moka pot.
libero_object / pick up the alphabet soup and place it in the basket -> the basket.
libero_goal / open the middle drawer of the cabinet -> the black drawer.
```

The script also keeps ATM's agent-view prompt rule. By default,
`observation.images.image` receives the prefix:

```text
the robotic arm. <task_prompt>
```

while wrist-view cameras use only `<task_prompt>`. Override this with:

```bash
--agentview-camera-keys observation.images.image
```

The bbox variant writes ragged episode-level bbox arrays:

```text
bbox_xyxy:        [K, 4]
bbox_confidences: [K]
bbox_offsets:     [C, T_label + 1]
bbox_counts:      [C, T_label]
```

For camera `c` and labeled frame `t`:

```python
start = bbox_offsets[c, t]
end = bbox_offsets[c, t + 1]
boxes = bbox_xyxy[start:end]
scores = bbox_confidences[start:end]
```

The bbox coordinates are `xyxy` pixel coordinates over the resized teacher
image controlled by `--image-size`.

If compatibility with ATM's original per-frame hdf5 layout is needed, add:

```bash
--write-frame-hdf5 --frame-hdf5-root ./bbox_data
```

This writes files containing datasets named `bboxes` and `confidences`.

### GroundingDINO CUDA extension for torch 2.7

The ATM repository's original GroundingDINO `_C` extension may be compiled
against a different PyTorch ABI. In `fastwam` with torch 2.7 this can show up
as:

```text
NameError: name '_C' is not defined
undefined symbol: torchInternalAssertFail
```

To avoid modifying the ATM repository, the bbox preprocessor uses a local
torch-2.7-compatible GroundingDINO overlay:

```text
third_party/grounding_dino_torch27/grounding_dino
```

If the extension needs to be rebuilt, run:

```bash
cd third_party/grounding_dino_torch27/grounding_dino
MAX_JOBS=1 CUDA_HOME=/usr/local/cuda \
  conda run -n fastwam python setup.py build_ext --inplace
```

The script uses only the local GroundingDINO code/config/checkpoint for bbox
preprocessing. If CUDA is requested but unavailable, or if the extension cannot
be imported, the script falls back to CPU with an explicit warning.

## Output format

### Depth cache

The depth preprocessor writes:

```text
<output-root>/<suite>/episode_XXXXXX.depth.npz
```

Each file contains:

```text
frame_indices
camera_keys
depth
depth_conf
meta_json
```

### Motion cache

The motion preprocessor writes:

```text
<output-root>/<suite>/episode_XXXXXX.motion.npz
```

Each file contains:

```text
frame_indices
camera_keys
motion_points
motion_visibility
motion_point_source
meta_json
```

### ATM bbox cache

For each processed episode, the bbox variant writes:

```text
<output-root>/<suite>/episode_XXXXXX.bbox.npz
```

Default output root:

```text
./data/libero_mujoco3.3.2_bbox_cache
```

Each `.bbox.npz` file contains:

```text
frame_indices
camera_keys
bbox_xyxy
bbox_confidences
bbox_offsets
bbox_counts
bbox_labels_json
meta_json
```

When `--bbox-backend grounded_sam2` is selected (the default), the file additionally
contains `bbox_masks`.

`bbox_labels_json` stores the raw text labels returned by the ATM/Grounding
DINO path for each camera/frame. `meta_json` records the resolved ATM prompt,
the prompt source, camera order, image size, and bbox coordinate convention.

Each preprocessor writes its global manifest to:

```text
<output-root>/manifest.jsonl
```

## Visualization output

Each preprocessor can save visualizations for the first `N` demos of each task.
This is enabled by default with `N=1`.

The saved video depends on the selected script:

```text
preprocess_libero_depth.py  -> depth_map.mp4
preprocess_libero_bbox.py   -> bbox_map.mp4
preprocess_libero_motion.py -> trajectory.mp4
```

Default visualization directory:

```text
<output-root>/visualizations
```

Example structure:

```text
<output-root>/visualizations/
└── libero_spatial_no_noops_lerobot/
    └── pick_up_the_black_bowl_and_place_it_on_the_plate/
        └── episode_000000/
            ├── task.txt
            ├── camera_00_observation.images.image/
            │   └── <modality>.mp4
            └── camera_01_observation.images.wrist_image/
                └── <modality>.mp4
```

Visualization controls:

```bash
--vis-num-demos-per-task 1
--vis-output-dir <path>
--vis-fps 10
--vis-max-frames 48
```

The motion-only script also accepts `--vis-max-tracks 128`.

Notes:

- `--vis-num-demos-per-task 0` disables visualization.
- `--vis-max-frames 48` uniformly samples at most 48 labeled frames for visualization only.
- `--vis-max-frames 0` renders the complete demo.
- Visualization does not change the saved training labels.

## Example commands

### Smoke test with depth labels

Use this to verify the default depth-only VDA path on one LIBERO episode. It
uses the default 224x224 video input and saves 224x224 labels.

```bash
python scripts/preprocess_libero_depth.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_depth_vda_smoke \
  --suites libero_goal_no_noops_lerobot \
  --max-episodes 1 \
  --device cuda \
  --frame-stride 1 \
  --vis-num-demos-per-task 1 \
  --overwrite
```

For a lightweight CPU smoke test, reduce both the saved video resolution and
VDA's internal inference size explicitly:

```bash
python scripts/preprocess_libero_depth.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_depth_vda_smoke \
  --suites libero_object_no_noops_lerobot \
  --max-episodes 1 \
  --camera-keys observation.images.image \
  --depth-backend video_depth_anything \
  --device cpu \
  --image-size 64 \
  --video-depth-anything-encoder vitl \
  --video-depth-anything-input-size 70 \
  --frame-stride 20 \
  --vis-num-demos-per-task 1 \
  --overwrite
```

### Full preprocessing with depth labels

This command uses the default VDA backend and writes full-resolution 224x224
depth labels to the `.depth.npz` episode caches.

```bash
python scripts/preprocess_libero_depth.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_depth_cache \
  --device cuda
```

### Smoke test with trajectory labels

Use this to verify the motion-only CoTracker path on one LIBERO episode.

```bash
python scripts/preprocess_libero_motion.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_motion_smoke \
  --suites libero_goal_no_noops_lerobot \
  --max-episodes 1 \
  --motion-backend cotracker3 \
  --device cuda \
  --frame-stride 1 \
  --vis-num-demos-per-task 1 \
  --overwrite
```

### Full preprocessing with trajectory labels

This command writes `.motion.npz` episode caches with CoTracker trajectory
labels only.

```bash
python scripts/preprocess_libero_motion.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_motion_cache \
  --motion-backend cotracker3 \
  --device cuda
```

### Smoke test with ATM bbox labels

Use this to verify the local ATM-style bbox path on one LIBERO episode. The
bbox script is bbox-only, so no depth or trajectory flags are needed.

```bash
python scripts/preprocess_libero_bbox.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_bbox_smoke \
  --suites libero_goal_no_noops_lerobot \
  --max-episodes 1 \
  --device cuda \
  --frame-stride 1 \
  --vis-num-demos-per-task 1 \
  --overwrite
```

### Full preprocessing with ATM bbox labels

This command writes `.bbox.npz` episode caches with raw ATM-style bbox labels
only.

```bash
python scripts/preprocess_libero_bbox.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_bbox_cache \
  --device cuda
```

### ATM-compatible per-frame hdf5 bbox export

Add `--write-frame-hdf5` when downstream code expects the original ATM
per-frame files:

```bash
python scripts/preprocess_libero_bbox.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_bbox_cache \
  --device cuda \
  --write-frame-hdf5 \
  --frame-hdf5-root ./bbox_data
```

The hdf5 export path is:

```text
<frame-hdf5-root>/<suite>/<atm-task-key>/bbox/episode_XXXXXX/<camera>_<frame>.hdf5
```

## Training-time slicing

Each cache is episode-level. Training-time slicing is implemented in
`src/fastwam/datasets/auxiliary_labels.py` and is enabled through
`data.train.auxiliary_labels.enabled=true`. The complete tensor, coordinate,
padding, and collator contract is documented in
`docs/libero_auxiliary_data.md`.

For FastWAM LIBERO defaults:

```text
num_frames = 33
action_video_freq_ratio = 4
video_sample_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32]
```

Given a sampled window start index, the loader derives the same raw timeline as
RGB (including episode-end replication):

```python
video_indices = start + np.arange(0, 33, 4)

depth_label = depth[:, video_indices]

traj = motion_points[:, video_indices]
traj_vis = motion_visibility[:, video_indices]
```

For `.bbox.npz`, bbox labels are ragged. First map episode frame indices to
label positions, then slice each selected frame through `bbox_offsets`:

```python
label_pos = {int(frame): i for i, frame in enumerate(frame_indices)}
video_label_pos = [label_pos[int(i)] for i in video_indices]

window_boxes = []
window_scores = []
for cam in range(len(camera_keys)):
    cam_boxes = []
    cam_scores = []
    for t in video_label_pos:
        start = bbox_offsets[cam, t]
        end = bbox_offsets[cam, t + 1]
        cam_boxes.append(bbox_xyxy[start:end])
        cam_scores.append(bbox_confidences[start:end])
    window_boxes.append(cam_boxes)
    window_scores.append(cam_scores)
```

The loader additionally applies the RGB camera composition and final
resize/crop. Boxes are returned as normalized `cx,cy,w,h`; masks remain aligned
one-to-one with boxes. The custom collator keeps per-frame instances ragged and
pads only the trajectory point dimension.

This design keeps preprocessing independent from a fixed action horizon or
video stride.

## Cache validation

`scripts/validate_libero_auxiliary_cache.py` is a read-only validator, not a
preprocessor or visualizer. It first checks that every LIBERO episode has all
three cache files, then deeply inspects a deterministic sample for complete
stride-1 timelines, camera ordering, finite tensors, ragged bbox offsets,
mask/box instance counts, trajectory shapes, and visibility shapes:

```bash
conda run -n fastwam python scripts/validate_libero_auxiliary_cache.py \
  --sample-count 24
```

The current caches pass coverage for all 1712 episodes and deep validation for
24 deterministic random episodes (seed 42).

## Model, training, and inference integration

Four independent training-only experts now live under:

```text
src/fastwam/models/wan22/auxiliary/
├── depth_branch.py
├── bbox_branch.py
├── mask_branch.py
└── trajectory_branch.py
```

Each expert owns its text/timestep projections, Video-DiT blocks, task tokens,
and output head. The common base is only a constructor/interface helper; it
does not create a unified SSI encoder and the branches do not share parameters.

The task outputs and losses are:

- Depth: `[B,T,1,H,W]`, SmoothL1 plus optional gradient loss.
- BBox: `[B,T,Q,C]` logits and normalized `[B,T,Q,4]` `cx,cy,w,h`, with
  classification + Hungarian-matched L1/GIoU loss.
- Mask: independent query-to-mask logits `[B,T,Q,H,W]`, with its own
  BCE/Dice Hungarian matching. It never reads BBox predictions.
- Trajectory: ATM-style repeated first-frame query tokens, deterministic
  double-grid point selection, `[B,N,T,2]` coordinates, `[B,N,T]` visibility,
  and masked coordinate/visibility losses.

During mixed MoT training, Action reads its own tokens and both cameras'
clean first-frame Video K/V. Each auxiliary expert reads its own tokens and
only the configured agentview region of first-frame Video K/V. Auxiliary
targets likewise contain agentview only while remaining aligned to the
two-camera canvas. Auxiliary experts cannot read one another, and Action
cannot read auxiliary tokens or head outputs. This gives every auxiliary loss
a gradient path into the Video world representation used by Action inference
without changing the action input.

`configs/model/fastwam.yaml` contains independent `enabled` flags and branch
settings. The repository default remains the exact baseline:

```yaml
model:
  auxiliary:
    enabled: false
```

For Full training on 4x A800 80GB, use ZeRO-2, gradient checkpointing, and an
effective global batch size of 128:

```bash
bash scripts/train_zero2.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  data.train.auxiliary_labels.enabled=true \
  model.auxiliary.enabled=true \
  model.auxiliary.depth.enabled=true \
  model.auxiliary.bbox.enabled=true \
  model.auxiliary.mask.enabled=true \
  model.auxiliary.trajectory.enabled=true \
  model.mot_checkpoint_mixed_attn=true \
  batch_size=2 \
  gradient_accumulation_steps=16 \
  num_workers=4
```

Here `batch_size` is per GPU, so `2 x 4 GPUs x 16 accumulation steps` keeps
the effective global batch size at 128.

The combined objective logs `loss_video`, `loss_action`, `loss_depth`,
`loss_bbox`, `loss_mask`, `loss_trajectory`, and `loss_total`. Auxiliary
parameters are registered inside `model.dit`, so the existing optimizer,
gradient accumulation, AMP, DDP, and checkpoint code includes them without a
second optimizer.

`infer_action()` still uses only Video prefill plus cached Video K/V and Action
denoising. Its inputs and `[action_horizon, action_dim]` output are unchanged;
it never creates or calls auxiliary tokens/heads. A Full checkpoint can be
loaded with all auxiliary branches disabled because checkpoint loading is
non-strict for optional MoT experts.

### Verification

Run the project-owned tests (the repository's `third_party/` trees contain
unrelated tests with additional simulator dependencies):

```bash
conda run -n fastwam python -m pytest -q tests
```

The suite covers aligned data loading/collation plus tiny real-DiT smoke tests
for every branch, isolated auxiliary-to-Video gradients, Full AMP and DDP
backward, exact baseline behavior when auxiliary computation is off,
checkpoint deployment, and inference-time branch skipping.

## Environment notes

LIBERO videos in the current FastWAM dataset are AV1-encoded mp4 files. A working video decoder is required. If decoding fails, install a suitable backend, for example:

```bash
pip install av "imageio[ffmpeg]"
```

or use a system FFmpeg build with AV1 support, such as `libdav1d` or `libaom`.

## TODO

### Preprocessing

- [ ] Verify the official Depth Anything 3 API path and update the `da3` backend if needed.
- [ ] Add deterministic seeding for random trajectory query sampling.
- [ ] Add quality metrics for label generation, such as empty-box ratio, depth confidence statistics, and trajectory visibility ratio.
- [ ] Add a resumable manifest mode that skips completed episodes while still allowing visualization regeneration.

### Dataset integration

- [x] Load depth, bbox/mask, and motion episode caches through one dataset adapter.
- [x] Preserve ragged bbox/mask instances with a dedicated DataLoader collator.
- [x] Slice episode labels to the Fast-WAM window and `video_sample_indices`.
- [x] Align two-camera labels with the composed RGB canvas and normalize coordinates.
- [x] Validate cache coverage, camera order, episode length, frame indices, and schemas.

### Model integration

- [x] Add four independent Depth/BBox/Mask/Trajectory DiT experts and heads.
- [x] Add depth, DETR-style bbox, query-mask, and ATM-style trajectory losses.
- [x] Route auxiliary gradients through clean-frame Video K/V shared with Action.
- [x] Ensure the action branch does not read auxiliary tokens or predictions.
- [x] Keep deployment action inference unchanged and support Full checkpoints with auxiliaries disabled.

### Experiments

- [ ] Run small LIBERO subset preprocessing with VDA and CoTracker3 teachers.
- [ ] Run small LIBERO subset preprocessing with ATM bbox labels.
- [ ] Compare FastWAM baseline vs SSI auxiliary branch.
- [ ] Add ablations for depth-only, bbox-only, trajectory-only, and combined supervision.
- [ ] Add a detached-video-feature ablation to verify whether SSI helps through representation shaping.
