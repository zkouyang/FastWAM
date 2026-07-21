# SSI-WAM on LIBERO：基于 Fast-WAM 的最小实现说明

## 1. 目标

在 Fast-WAM 原有 **Video Branch + Action Branch** 基础上，增加四个仅训练期使用的空间辅助分支：

1. **Depth Branch**
2. **BBox Branch**
3. **Mask Branch**
4. **Trajectory Branch**

四个分支共享 Fast-WAM 已有的视觉和语言编码输入，但分别拥有独立的 DiT branch 和任务输出 head。辅助分支只通过联合训练增强 Action Branch 使用的共享世界表示；部署时不显式生成 depth、bbox、mask 或 trajectory。

LIBERO 阶段的目标是以尽量小的代码改动完成：

- 已处理标签的读取与时间对齐；
- 四个辅助分支的独立开关；
- 联合训练和消融实验；
- 保持原 Fast-WAM 推理接口与评测流程不变。

---

## 2. 已知前提

默认 LIBERO demonstration 已具有与 RGB 视频对齐的标签：

- raw depth map；
- bounding box；
- segmentation mask；
- pixel trajectory。

本阶段不实现离线标注，不调用 Video Depth Anything、SAM、Grounding DINO 或 ATM 生成标签。

参考仓库：

- Fast-WAM: `https://github.com/yuantianyuan01/FastWAM`
- ATM: `https://github.com/Large-Trajectory-Model/ATM`

---

## 3. 总体架构

```text
RGB observations ── Fast-WAM VAE / Visual Encoding ── visual tokens
Language instruction ── Fast-WAM Text Encoder ─────── language tokens
                                      │
          ┌──────────────┬────────────┼────────────┬────────────┬──────────────┐
          │              │            │            │            │              │
      Video DiT      Action DiT   Depth DiT     BBox DiT     Mask DiT   Trajectory DiT
          │              │            │            │            │              │
      Video Head     Action Head   Depth Head    BBox Head    Mask Head   ATM Track Head
```

每个辅助 branch 均由两部分组成：

```text
Video-DiT-like branch backbone + task-specific lightweight head/decoder
```

### 关键约束

- 不新增统一的 `SSI Module`、`SSI Encoder` 或串联式 SSI 输入。
- Depth、BBox、Mask、Trajectory 是四个彼此独立的 branch。
- 四个 branch 尽量复用 Video Branch 的模块结构、conditioning 接口和 attention 实现。
- 四个 branch 使用独立参数；可以选择从 Video DiT 初始化，但默认不与 Video Branch 共用完整 branch 权重。
- Action Branch 不读取四个 head 的显式预测结果。
- 推理时不创建辅助 target token，也不调用四个辅助 branch。
- 不改变原 Fast-WAM action generation 和 LIBERO evaluation 接口。

### 辅助监督如何影响 Action Branch

仅把相同 visual/language tensor 分别送入完全独立且无共享参数的网络，不能有效增强 Action Branch。实现时必须保证辅助 loss 的梯度能够更新 Action Branch 所依赖的共享世界表示，例如 Fast-WAM 已有的 clean-frame anchor representation、共享 attention 或对应的共享 world-backbone 层。

具体共享位置应在阅读 Fast-WAM 代码后确定，并遵循其 Video/Action MoT 与 structured attention 设计。四个 branch 的显式输出仍不得作为 Action DiT 输入。

---

## 4. 最小改动原则

1. 不重写 Fast-WAM 的训练器、Video DiT 或 Action DiT。
2. 优先抽取或复用原 Video Branch 的 branch/block 构造函数。
3. 四个新 branch 保持与 Video Branch 一致的：
   - hidden size 和 token layout；
   - timestep/noise embedding 接口（如实际需要）；
   - language cross-attention 接口；
   - clean-frame visual conditioning 接口；
   - DDP、AMP 和 checkpoint 行为。
4. 仅替换每个任务的输入 token adapter 与输出 head。
5. 所有新功能通过配置开关控制；关闭后必须恢复原 Fast-WAM 行为。

---

## 5. 推荐代码结构

应首先映射到 Fast-WAM 现有目录，避免移动原文件。建议新增类似：

```text
src/fastwam/
├── models/
│   ├── fastwam.py                    # 最小修改：注册并调用四个 branch
│   └── auxiliary/
│       ├── __init__.py
│       ├── base_video_dit_branch.py  # 可选：复用 Video Branch 的公共包装
│       ├── depth_branch.py
│       ├── bbox_branch.py
│       ├── mask_branch.py
│       └── trajectory_branch.py
│
├── losses/
│   ├── depth_loss.py
│   ├── bbox_loss.py
│   ├── mask_loss.py
│   └── trajectory_loss.py
│
├── data/
│   ├── libero_dataset.py             # 在原 Dataset 上最小扩展
│   └── auxiliary_labels.py
│
└── training/
    └── auxiliary_loss_manager.py     # 可选
```

`base_video_dit_branch.py` 只用于减少重复代码，不代表重新封装统一 SSI 模块。四个 branch 必须能够单独构建、关闭和加载 checkpoint。

---

## 6. 数据格式

### 6.1 单个训练样本

字段名应适配现有 Fast-WAM Dataset；下面仅表示语义：

```python
sample = {
    # Fast-WAM 原始字段
    "images": ...,             # observation/video frames
    "language": ...,           # task instruction
    "actions": ...,            # action chunk
    "future_frames": ...,      # original video target

    # 新增标签
    "depth": ...,              # raw depth maps, [T, 1, H, W] or [T, H, W]
    "boxes": ...,              # per-frame List[Tensor[Ni, 4]]
    "box_labels": ...,         # per-frame List[Tensor[Ni]]，如数据包含类别
    "masks": ...,              # per-frame List[Tensor[Ni, H, W]]
    "trajectories": ...,       # [N, T_traj, 2]
    "traj_visibility": ...,    # [N, T_traj]
    "traj_query_points": ...,  # [N, 2]
}
```

### 6.2 数据约定

- depth：直接读取并使用原始 depth map label。
- bbox：推荐统一为 normalized `cx, cy, w, h`，范围 `[0, 1]`。
- mask：binary/float mask，范围 `[0, 1]`。
- trajectory：坐标格式和可见性定义直接遵循 ATM 数据与 Track Transformer 实现。
- Depth/BBox/Mask 的时间维应与对应 RGB target frame 对齐。
- Trajectory 的起始帧、query points 和预测 horizon 必须与 ATM 约定一致。

所有索引、resize 和坐标变换集中在 Dataset/Collator 完成，模型内部不修正时间偏移。

---

## 7. 模型接口

下面是伪代码。实际实现应复用 Fast-WAM 当前 forward 和 token flow，而不是强行新增相同名称的方法。

```python
class FastWAMWithSpatialAux(nn.Module):
    def __init__(
        self,
        ...,
        depth_branch=None,
        bbox_branch=None,
        mask_branch=None,
        trajectory_branch=None,
        aux_config=None,
    ):
        super().__init__()
        self.depth_branch = depth_branch
        self.bbox_branch = bbox_branch
        self.mask_branch = mask_branch
        self.trajectory_branch = trajectory_branch
        self.aux_config = aux_config

    def forward(
        self,
        batch,
        compute_video=True,
        compute_auxiliary=True,
    ):
        shared = self.encode_fastwam_context(
            images=batch["images"],
            language=batch["language"],
        )

        outputs = {
            "action": self.forward_action_branch(
                shared=shared,
                actions=batch.get("actions"),
            )
        }

        if self.training and compute_video:
            outputs["video"] = self.forward_video_branch(
                shared=shared,
                future_frames=batch.get("future_frames"),
            )

        if self.training and compute_auxiliary:
            if self.depth_branch is not None:
                outputs["depth"] = self.depth_branch(shared, batch)

            if self.bbox_branch is not None:
                outputs["bbox"] = self.bbox_branch(shared, batch)

            if self.mask_branch is not None:
                outputs["mask"] = self.mask_branch(shared, batch)

            if self.trajectory_branch is not None:
                outputs["trajectory"] = self.trajectory_branch(shared, batch)

        return outputs
```

`shared` 必须包含 Action Branch 实际依赖、且允许辅助 loss 回传更新的共享世界表示。

---

## 8. 四个辅助分支

## 8.1 Depth Branch

### Branch backbone

- 独立 Depth DiT。
- 尽量复制/复用原 Video DiT 的 block、language cross-attention 和 visual conditioning 接口。
- 输入为共享 visual/language representation 和必要的 depth task tokens。

### 简单输出 head

```text
DiT tokens
  → Linear projection
  → spatial reshape
  → 2–3 layers Upsample + Conv
  → one-channel depth map
```

输出：

```python
pred_depth: Tensor[B, T, 1, H_d, W_d]
```

### 标签与 loss

直接使用原始 depth map label。第一版采用简单监督：

```text
L_depth = SmoothL1(pred_depth, depth_label)
          + alpha_grad * GradientLoss(pred_depth, depth_label)  # optional
```

默认先令 `alpha_grad = 0`，完成直接回归 smoke test 后再开启。

---

## 8.2 BBox Branch

### Branch backbone

- 独立 BBox DiT。
- 使用 Video-DiT-like blocks、共享 visual/language conditioning 和 learnable object queries。

### 简单输出 head

```text
BBox query tokens
  ├── Linear/MLP → objectness or class logits
  └── 3-layer MLP + sigmoid → normalized boxes (cx, cy, w, h)
```

输出：

```python
{
    "pred_logits": Tensor[B, T, Q, C],
    "pred_boxes": Tensor[B, T, Q, 4],
}
```

如果标签只有 boxes、没有类别，可将 `C=1`，仅预测 objectness。

### Matching 与 loss

采用简单 DETR 风格 Hungarian matching：

```text
matching cost = classification/objectness cost + L1 box cost + GIoU cost
```

```text
L_bbox = L_cls + beta_l1 * L_box_l1 + beta_giou * L_giou
```

建议初始值：`beta_l1=5.0`，`beta_giou=2.0`。

---

## 8.3 Mask Branch

### Branch backbone

- 独立 Mask DiT，不与 BBox Branch 共用 branch 或 decoder。
- 尽量复用 Video DiT block 与 visual/language conditioning。
- 使用独立 mask queries 或 dense mask tokens。

### 简单输出 head

推荐轻量 query-mask head：

```text
DiT image tokens → lightweight upsampling feature map
DiT mask queries → MLP projection
query embedding × feature map → mask logits
```

该结构比完整 SAM Mask Decoder 更容易接入；可参考 SAM 的 query-to-mask 思路，但不复制完整模型。

输出：

```python
pred_masks: Tensor[B, T, Q, H_m, W_m]
```

### Loss

若 mask 与对象实例一一对应，可独立使用 Hungarian matching 或复用标签中固定的对象顺序，但不得依赖 BBox Branch 的预测结果。

```text
L_mask = BCEWithLogitsLoss + beta_dice * DiceLoss
```

建议初始值：`beta_dice=1.0`。

---

## 8.4 Trajectory Branch

### Branch backbone 与 head

- 独立 Trajectory DiT branch。
- visual/language conditioning 接口尽量与 Video Branch 一致。
- query-point embedding、track token 组织、temporal prediction head 和 visibility head 直接参考 ATM 的 Track Transformer 代码。
- 优先复用或最小适配 ATM Track Transformer，不自行重新设计复杂 trajectory decoder。

输出格式遵循 ATM：

```python
{
    "pred_coords": Tensor[B, N, T_traj, 2],
    "pred_visibility": Tensor[B, N, T_traj],
}
```

### Loss

优先复用 ATM 代码中的 trajectory training loss 和可见性处理。若需要最小替代版本：

```text
L_traj = masked coordinate loss + beta_vis * visibility BCE
```

不要在第一版额外加入 smoothness、cycle consistency 等新目标。

---

## 9. 总损失

Fast-WAM 原目标：

```text
L_base = L_action + lambda_video * L_video
```

加入四个独立辅助目标：

```text
L_total = L_action
        + lambda_video * L_video
        + lambda_depth * L_depth
        + lambda_bbox * L_bbox
        + lambda_mask * L_mask
        + lambda_traj * L_traj
```

第一版建议：

```yaml
loss_weights:
  action: 1.0
  video: 1.0
  depth: 0.1
  bbox: 0.1
  mask: 0.1
  trajectory: 0.1
```

要求：

- 单独记录六项 loss 和 total loss；
- 支持单独关闭任意辅助 loss/branch；
- 检查 NaN/Inf；
- 至少验证每个辅助 loss 能对共享世界表示产生非零梯度；
- 辅助权重不得改变 baseline 配置的默认行为。

---

## 10. 配置建议

```yaml
model:
  enable_depth_branch: true
  enable_bbox_branch: true
  enable_mask_branch: true
  enable_trajectory_branch: true

  depth_branch:
    init_from_video_dit: false
    hidden_dim: null          # 默认跟随 Video DiT
    output_size: [128, 128]

  bbox_branch:
    init_from_video_dit: false
    num_queries: 16
    num_classes: 1

  mask_branch:
    init_from_video_dit: false
    num_queries: 16
    output_size: [128, 128]

  trajectory_branch:
    use_atm_track_transformer: true
    num_points: 64
    horizon: 16

training:
  compute_auxiliary: true
  loss_weights:
    video: 1.0
    depth: 0.1
    bbox: 0.1
    mask: 0.1
    trajectory: 0.1

inference:
  use_video_world_encoder: true   # 保留 Fast-WAM 当前帧单次 Video DiT 前向
  generate_future_video: false    # 不进行未来视频生成/去噪
  compute_auxiliary: false
```

`hidden_dim`、`num_points`、`horizon` 和分辨率应从实际 Fast-WAM、LIBERO 标签与 ATM 配置读取，不要硬编码以上示例。

---

## 11. 训练流程

```python
outputs = model(
    batch,
    compute_video=True,
    compute_auxiliary=True,
)

losses = {
    "action": compute_action_loss(outputs["action"], batch),
    "video": compute_video_loss(outputs["video"], batch),
}

if "depth" in outputs:
    losses["depth"] = compute_depth_loss(outputs["depth"], batch["depth"])

if "bbox" in outputs:
    losses["bbox"] = compute_bbox_loss(outputs["bbox"], batch)

if "mask" in outputs:
    losses["mask"] = compute_mask_loss(outputs["mask"], batch["masks"])

if "trajectory" in outputs:
    losses["trajectory"] = compute_atm_trajectory_loss(
        outputs["trajectory"],
        batch,
    )

total_loss = sum(loss_weights[name] * loss for name, loss in losses.items())
```

保持原 Fast-WAM optimizer、scheduler、AMP、DDP、gradient accumulation 和 checkpoint 流程。

---

## 12. 推理流程

推理只执行原 Fast-WAM action 路径：

```python
with torch.no_grad():
    actions = model.generate_actions(images, language)
```

推理时不得：

- 创建 future video target/noisy tokens；
- 创建 depth、bbox、mask 或 trajectory task tokens；
- 调用四个辅助 branch/head；
- 将辅助预测拼接到 Action Branch；
- 改变原 action output 格式。

---

## 13. LIBERO 最小实验矩阵

| ID | Video | Depth | BBox | Mask | Trajectory | 目的 |
|---|---:|---:|---:|---:|---:|---|
| B0 | ✓ | ✗ | ✗ | ✗ | ✗ | 原 Fast-WAM baseline |
| D | ✓ | ✓ | ✗ | ✗ | ✗ | Depth 单分支 |
| B | ✓ | ✗ | ✓ | ✗ | ✗ | BBox 单分支 |
| M | ✓ | ✗ | ✗ | ✓ | ✗ | Mask 单分支 |
| T | ✓ | ✗ | ✗ | ✗ | ✓ | Trajectory 单分支 |
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | 四分支联合训练 |

先在一个 LIBERO suite 完成 smoke test，再运行：

- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long

保持原 Fast-WAM 的 seed、rollout 数量、action horizon、observation preprocessing 和 evaluation protocol。

---

## 14. 分阶段实现

### Phase 1：代码与数据定位

- 定位 Fast-WAM Dataset、Video DiT、Action DiT、structured attention、loss 和 inference entry。
- 定位 ATM Track Transformer、query point 输入、trajectory target 和 loss。
- 明确四类标签的文件路径、shape、时间轴和 resize 规则。

### Phase 2：数据接入

- Dataset 能加载 depth/bbox/mask/trajectory。
- 随机样本可视化确认与 RGB 对齐。
- DataLoader 和 Collator 能处理每帧可变数量 bbox/mask。

### Phase 3：单分支 smoke test

依次完成：

1. Depth only
2. BBox only
3. Mask only
4. Trajectory only

每个分支必须满足：

- 单 batch forward/backward 成功；
- loss 为有限值并有下降趋势；
- branch/head 参数存在非零梯度；
- 共享世界表示存在来自该辅助 loss 的非零梯度；
- 关闭 branch 后恢复 baseline 行为。

### Phase 4：Full 联合训练

- 四个分支同时工作；
- AMP、DDP、checkpoint save/resume 正常；
- 无明显显存泄漏；
- 每项 loss 可追踪。

### Phase 5：LIBERO 评测

输出：

- 每个 suite 成功率与平均成功率；
- 相对 B0 的变化；
- 分项 loss 曲线；
- 训练速度和显存占用；
- 推理 latency 与 B0 对比。

---

## 15. 验收标准

### 数据

- [ ] 四类标签与 RGB 帧对齐
- [ ] bbox/trajectory 坐标约定明确
- [ ] mask 和 bbox 的实例对应关系明确
- [ ] 时间 horizon 正确

### 模型

- [ ] 四个 branch 可独立开关
- [ ] 四个 branch 均复用 Video-DiT-like 接口
- [ ] BBox 与 Mask branch 完全独立
- [ ] Trajectory branch 正确适配 ATM Track Transformer
- [ ] Action Branch 不读取显式辅助预测
- [ ] 辅助 loss 能更新 Action Branch 依赖的共享世界表示

### 训练

- [ ] 六项 loss 命名和日志一致
- [ ] 每个辅助 loss 可独立 backward
- [ ] Full loss 可联合 backward
- [ ] AMP/DDP/checkpoint 正常
- [ ] baseline 配置可复现原 Fast-WAM

### 推理

- [ ] 不调用四个辅助 branch/head
- [ ] action 输出接口与原 Fast-WAM 一致
- [ ] 推理 latency 基本不增加
- [ ] Full checkpoint 可在关闭辅助 branch 时部署

---

## 16. Codex 首轮任务

先完成代码分析，不要立即实现全部 branch：

1. 阅读 Fast-WAM，定位 Dataset、Video DiT、Action DiT、shared attention/token flow、training loss、inference 和 LIBERO evaluation。
2. 阅读 ATM，定位 Track Transformer、trajectory input/output、query point 组织和 loss。
3. 给出最小修改文件列表。
4. 给出四类标签接入 Dataset 的具体位置和预期 shape。
5. 给出四个 branch 复用 Video Branch 代码的方式，明确哪些模块复用实现、哪些参数独立。
6. 明确辅助 loss 回传到 Action Branch 所依赖共享表示的具体路径。
7. 明确推理时完全跳过四个 branch 的代码路径。
8. 得到确认后，先实现 Dataset 接入和 Depth-only smoke test。
