# SSI-WAM Notes

This document tracks the current SSI-WAM implementation plan and the preprocessing utilities added on top of FastWAM. It should be updated whenever new SSI branches, dataset adapters, model heads, or training losses are implemented.

## Current status

The repository currently includes an offline SSI label preprocessor for LIBERO:

```text
scripts/preprocess_libero_ssi.py
```

It also includes an ATM-style bbox-label variant:

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

The preprocessor generates episode-level SSI supervision labels. It is intentionally a label-stage tool: labels are saved over the full demonstration timeline, and the training dataset should later slice the labels according to the policy horizon, video stride, and SSI loss configuration.

Use `preprocess_libero_ssi.py` when the auxiliary branch should learn role
layout maps (`source`, `target`, `robot`, `other`). Use
`preprocess_libero_bbox.py` when the auxiliary branch should keep the original
ATM bbox labels: raw `[x1, y1, x2, y2]` boxes plus confidence scores. Use
`preprocess_libero_depth.py` when only monocular depth-grid labels are needed.
Use `preprocess_libero_motion.py` when only point trajectory labels are needed.

## Depth-only label variant

`scripts/preprocess_libero_depth.py` keeps the FastWAM/LIBERO episode loader,
manifest writing, and depth visualization utilities, but is otherwise
depth-only. It does not compute role-layout labels, bbox labels, or trajectory
labels.

The depth variant writes:

```text
depth:      [C, T_label, G, G]
depth_conf: [C, T_label, G, G]
```

`G` is controlled by `--grid-size`, and `T_label` is controlled by
`--frame-stride`. The default backend is Depth Anything 3:

```bash
--depth-backend da3 --da3-model-id depth-anything/DA3MONO-LARGE
```

Depth Anything V2 ViT-L is also supported with a local repo and checkpoint:

```bash
--depth-backend da2 --da2-repo-dir ./third_party/Depth_Anything_V2 --da2-checkpoint ./third_party/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth
```

For temporally consistent video depth labels, use Video Depth Anything:

```bash
--depth-backend video_depth_anything --video-depth-anything-repo-dir ./third_party/Video-Depth-Anything --video-depth-anything-checkpoint ./third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth
```

## Trajectory-only label variant

`scripts/preprocess_libero_motion.py` keeps the FastWAM/LIBERO episode loader,
manifest writing, and trajectory visualization utilities, but is otherwise
motion-only. It does not compute depth labels, role-layout labels, or bbox
labels.

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
It does not compute depth labels, role-layout labels, or trajectory labels.

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

The bbox variant does not write `layout`, `layout_conf`, or `role_names`.
Instead it writes ragged episode-level bbox arrays:

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

## Default layout teacher: Grounding DINO-T

`sam3` is still available as an optional layout backend, but the default layout
teacher is now Grounding DINO-T because the official SAM3 weights are gated and
may be unavailable.

Default layout backend:

```text
--layout-backend grounding_dino_t
```

Default public model:

```text
IDEA-Research/grounding-dino-tiny
```

Required packages are already expected in the `fastwam` environment:

```bash
conda run -n fastwam python -m pip install "transformers>=4.49.0" huggingface_hub safetensors
```

Download the public Grounding DINO-T files once:

```bash
conda run -n fastwam python -c "from huggingface_hub import hf_hub_download; files=['config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','vocab.txt','special_tokens_map.json','model.safetensors']; [hf_hub_download('IDEA-Research/grounding-dino-tiny', f) for f in files]"
```

The preprocessing script loads this model with local-cache-only mode by default:

```text
--grounding-dino-local-files-only
```

If the model is not cached and online loading is desired, use:

```text
--no-grounding-dino-local-files-only
```

Grounding DINO-T produces language-grounded boxes. The preprocessor rasterizes
accepted boxes into role-wise SSI grids, so the saved `layout` and `layout_conf`
tensor schema remains identical to the SAM3 path.

## Optional local SAM3 setup

SAM3 is expected to be installed from the local repository path:

```text
third_party/sam3
```

The `fastwam` environment should use this local checkout in editable mode:

```bash
conda run -n fastwam python -m pip install -e third_party/sam3 --no-deps
```

Verify the active import path:

```bash
conda run -n fastwam python -c "import sam3, pathlib; print(pathlib.Path(sam3.__file__).resolve())"
```

The printed path should point to:

```text
<repo-root>/third_party/sam3/sam3/__init__.py
```

SAM3 model weights are local-only by default. Place the official image
checkpoint here:

```text
third_party/sam3/checkpoints/sam3.pt
```

After Hugging Face login and access approval for `facebook/sam3`, download the
checkpoint with:

```bash
conda run -n fastwam python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', local_dir='third_party/sam3/checkpoints')"
```

Then verify:

```bash
ls -lh third_party/sam3/checkpoints/sam3.pt
```

Important: `huggingface-cli login` only configures a token. The account must
also be approved for the gated `facebook/sam3` repository. If the download
fails with `403 Forbidden` or `not in the authorized list`, visit:

```text
https://huggingface.co/facebook/sam3
```

request/accept access, then rerun the download command.

The preprocessing script also accepts:

```bash
export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
```

or:

```bash
--sam3-checkpoint-path /path/to/sam3.pt
```

Hugging Face download is disabled by default because `facebook/sam3` is gated.
Only use `--layout-backend sam3 --sam3-load-from-hf` when the environment is
logged in with an account that has accepted access to the official SAM3 weights.

## SSI label components

The current preprocessor produces three SSI modalities.

### 1. Geometry / depth labels

Depth labels are generated per camera and per selected frame.

Supported backends:

- `da3`: Depth Anything 3 backend.
- `heuristic`: lightweight smoke-test backend based on grayscale intensity.
- `none`: writes zero labels.

Saved fields:

```text
depth:      [C, T_label, G, G]
depth_conf: [C, T_label, G, G]
```

Where:

- `C`: number of cameras.
- `T_label`: number of labeled frames.
- `G`: SSI grid size, controlled by `--grid-size`.

Depth values are normalized to `[0, 1]` before being downsampled to the SSI grid.

### 2. Layout / mask labels

Layout labels are role-based spatial maps derived from language-grounded boxes
or masks.

Current roles:

```text
source
target
robot
other
```

Supported backends:

- `grounding_dino_t`: default Grounding DINO-T box-grounding backend.
- `sam3`: optional SAM 3 / SAM 3.1 concept segmentation backend.
- `heuristic`: lightweight smoke-test backend with synthetic role maps.
- `none`: writes zero labels.

Saved fields:

```text
layout:      [C, T_label, R, G, G]
layout_conf: [C, T_label, R, G, G]
role_names:  [R]
```

The script parses the LIBERO task instruction into role prompts. With the
default Grounding DINO-T backend, it runs text-prompted box grounding for each
role, rasterizes boxes into dense confidence maps, and downsamples them into
role grids. With `sam3`, it instead runs text-prompted segmentation masks before
the same grid conversion.

### 3. Trajectory labels

Trajectory labels follow the ATM-style CoTracker preprocessing pattern.

The current CoTracker path:

1. Samples random query points.
2. Samples an ATM-style double grid.
3. Assigns each query a random anchor time.
4. Runs CoTracker with backward tracking when supported.
5. Filters static random points using trajectory variance.
6. Repeats dynamic points with spatial noise.
7. Re-tracks and concatenates grid tracks with dynamic random tracks.

Supported backends:

- `cotracker3`: CoTracker3 backend.
- `opencv`: lightweight LK optical-flow fallback for smoke tests.
- `none`: writes zero labels.

Saved fields:

```text
motion_points:       [C, T_episode, N, 2]
motion_visibility:   [C, T_episode, N]
motion_point_source: [C, N]
```

Coordinates are normalized `xy` values in `[0, 1]`.

With default CoTracker settings:

```text
N = motion_num_random_points + 2 * motion_grid_size^2
  = 1000 + 2 * 7^2
  = 1098
```

## Output format

### Role-layout SSI cache

For each processed episode, the script writes:

```text
<output-root>/<suite>/episode_XXXXXX.ssi.npz
```

Default output root:

```text
./data/libero_mujoco3.3.2_ssi_cache
```

Each `.npz` file contains:

```text
frame_indices
camera_keys
role_names
depth
depth_conf
layout
layout_conf
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
depth
depth_conf
bbox_xyxy
bbox_confidences
bbox_offsets
bbox_counts
bbox_labels_json
motion_points
motion_visibility
motion_point_source
meta_json
```

`bbox_labels_json` stores the raw text labels returned by the ATM/Grounding
DINO path for each camera/frame. `meta_json` records the resolved ATM prompt,
the prompt source, camera order, image size, and bbox coordinate convention.

The global manifest is written to:

```text
<output-root>/manifest.jsonl
```

## Visualization output

The preprocessor can save visualizations for the first `N` demos of each task. This is enabled by default with `N=1`.

Saved visualization types:

```text
depth_map.mp4
bbox_map.mp4
trajectory.mp4
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
            │   ├── depth_map.mp4
            │   ├── bbox_map.mp4
            │   └── trajectory.mp4
            └── camera_01_observation.images.wrist_image/
                ├── depth_map.mp4
                ├── bbox_map.mp4
                └── trajectory.mp4
```

Visualization controls:

```bash
--vis-num-demos-per-task 1
--vis-output-dir <path>
--vis-fps 10
--vis-max-frames 48
--vis-max-tracks 128
--vis-bbox-threshold 0.20
```

Notes:

- `--vis-num-demos-per-task 0` disables visualization.
- `--vis-max-frames 48` uniformly samples at most 48 labeled frames for visualization only.
- `--vis-max-frames 0` renders the complete demo.
- Visualization does not change the saved training labels.
- `--vis-bbox-threshold` only applies to role-layout SSI visualizations; the bbox variant draws the raw boxes directly.

## Example commands

### Smoke test with SSI teacher models

Use this to verify the real DA3 / Grounding DINO-T / CoTracker3 preprocessing
path on one LIBERO episode.

```bash
python scripts/preprocess_libero_ssi.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_ssi_cache_tmp \
  --suites libero_spatial_no_noops_lerobot \
  --max-episodes 1 \
  --depth-backend da3 \
  --layout-backend grounding_dino_t \
  --motion-backend cotracker3 \
  --device cuda
```

For a quick layout-only validation on CPU, label only one frame and disable
depth/motion:

```bash
python scripts/preprocess_libero_ssi.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root /tmp/fastwam_ssi_grounding_dino_t_smoke \
  --suites libero_spatial_no_noops_lerobot \
  --max-episodes 1 \
  --depth-backend none \
  --layout-backend grounding_dino_t \
  --motion-backend none \
  --device cpu \
  --frame-stride 999 \
  --vis-num-demos-per-task 1 \
  --overwrite
```

### Full preprocessing with SSI teacher models

Uses Grounding DINO-T by default for layout labels.

```bash
python scripts/preprocess_libero_ssi.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_ssi_cache \
  --depth-backend da3 \
  --layout-backend grounding_dino_t \
  --motion-backend cotracker3 \
  --device cuda
```

### Smoke test with depth labels

Use this to verify the depth-only DA3 path on one LIBERO episode.

```bash
python scripts/preprocess_libero_depth.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_depth_smoke \
  --suites libero_goal_no_noops_lerobot \
  --max-episodes 1 \
  --depth-backend da3 \
  --device cuda \
  --frame-stride 1 \
  --vis-num-demos-per-task 1 \
  --overwrite
```

Video Depth Anything is also supported for temporally consistent depth labels.
The default checkpoint is the official Large relative-depth model:
`third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth`.

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
  --vis-num-demos-per-task 0 \
  --overwrite
```

### Full preprocessing with depth labels

This command writes `.depth.npz` episode caches with DA3 depth labels only.

```bash
python scripts/preprocess_libero_depth.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_depth_cache \
  --depth-backend da3 \
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
bbox script is bbox-only, so no depth, role-layout, or trajectory flags are
needed.

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

### Full preprocessing with complete visualization videos

```bash
python scripts/preprocess_libero_ssi.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_ssi_cache \
  --depth-backend da3 \
  --layout-backend grounding_dino_t \
  --motion-backend cotracker3 \
  --device cuda \
  --vis-num-demos-per-task 1 \
  --vis-max-frames 0
```

### Disable visualization

```bash
python scripts/preprocess_libero_ssi.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_ssi_cache \
  --depth-backend da3 \
  --layout-backend grounding_dino_t \
  --motion-backend cotracker3 \
  --device cuda \
  --vis-num-demos-per-task 0
```

### Optional SAM3 layout backend

SAM3 remains available when local or authorized weights are available:

```bash
python scripts/preprocess_libero_ssi.py \
  --data-root ./data/libero_mujoco3.3.2 \
  --output-root ./data/libero_mujoco3.3.2_ssi_cache_sam3 \
  --depth-backend da3 \
  --layout-backend sam3 \
  --sam3-checkpoint-path /path/to/sam3.pt \
  --motion-backend cotracker3 \
  --device cuda
```

## Training-time slicing

The cache is episode-level. Training code should dynamically slice it.

For FastWAM LIBERO defaults:

```text
num_frames = 33
action_video_freq_ratio = 4
video_sample_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32]
```

Given a sampled window start index:

```python
video_indices = start + np.arange(0, 33, 4)

depth_label = depth[:, video_indices]
layout_label = layout[:, video_indices]

traj = motion_points[:, video_indices]
traj_delta = traj - traj[:, :1]
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

If a dense tensor is required by the model, pad each frame to the configured
maximum bbox count and concatenate confidence as the fifth column:

```text
[x1, y1, x2, y2, confidence]
```

This design keeps preprocessing independent from a fixed action horizon or video stride.

## Environment notes

LIBERO videos in the current FastWAM dataset are AV1-encoded mp4 files. A working video decoder is required. If decoding fails, install a suitable backend, for example:

```bash
pip install av "imageio[ffmpeg]"
```

or use a system FFmpeg build with AV1 support, such as `libdav1d` or `libaom`.

## TODO

### Preprocessing

- [ ] Verify the official SAM 3.1 API path and update the `sam3` backend if the installed package exposes a different interface.
- [ ] Verify the official Depth Anything 3 API path and update the `da3` backend if needed.
- [ ] Add deterministic seeding for random trajectory query sampling.
- [ ] Add optional role-prompt metadata files for more reliable LIBERO object/receptacle parsing.
- [ ] Add quality metrics for label generation, such as empty-mask ratio, depth confidence statistics, and trajectory visibility ratio.
- [ ] Add a resumable manifest mode that skips completed episodes while still allowing visualization regeneration.

### Dataset integration

- [ ] Add an SSI-aware dataset wrapper that loads `.ssi.npz` files.
- [ ] Add a bbox-aware dataset wrapper that loads `.bbox.npz` files and pads ragged bbox labels.
- [ ] Implement training-time slicing from episode-level SSI labels to window-level supervision.
- [ ] Align SSI frame indices with FastWAM `video_sample_indices`.
- [ ] Add validation checks for camera order, episode length, and frame index consistency.

### Model integration

- [ ] Add an SSI branch / SSI expert module.
- [ ] Add depth, layout, and trajectory prediction heads.
- [ ] Add SSI auxiliary losses.
- [ ] Ensure the action branch does not read SSI tokens during deployment.
- [ ] Keep deployment path identical to FastWAM action inference except for checkpoint compatibility.

### Experiments

- [ ] Run smoke tests with heuristic labels.
- [ ] Run small LIBERO subset preprocessing with real DA3 / Grounding DINO-T / CoTracker3 teachers.
- [ ] Run small LIBERO subset preprocessing with ATM bbox labels.
- [ ] Optionally compare Grounding DINO-T box grids with SAM3 masks if SAM3 access becomes available.
- [ ] Compare role-layout SSI labels vs raw ATM bbox labels.
- [ ] Compare FastWAM baseline vs SSI auxiliary branch.
- [ ] Add ablations for depth-only, layout-only, trajectory-only, and full SSI supervision.
- [ ] Add a detached-video-feature ablation to verify whether SSI helps through representation shaping.
