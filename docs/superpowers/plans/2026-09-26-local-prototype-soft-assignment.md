# Local Prototype Soft Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在保持旧实验可重建的前提下，交付可用于NWPU和BigEarthNet工程短测的`soft_prototype`分组、软反馈、训练入口及诊断记录。

**Architecture:** 复用原固定区域注意力生成当前图片的区域原型；共享LN/线性投影后做局部cosine软分配。S用于区域竞争和反馈，W用于注意力加权区域汇总；保留Swin、K16、48+32边和现有GAT。新增配置以版本化字典存入checkpoint，历史配置和参数名保持原样。

**Tech Stack:** Python、PyTorch/torchvision、timm、PyG、NumPy；现有unittest测试由pytest运行。纯身份模块不依赖Torch；本地CPU验证与Featurize CUDA验证分别记录。

**Spec:** [已确认设计](../specs/2026-09-26-local-prototype-soft-assignment-design.md)。用户2026-09-26回复“ok”确认设计；本计划供实施前审阅，复选框均为未完成工作。

## Global Constraints

- 只实现`local_prototype_cosine_v1`：N49、D768、K16，投影128，局部切比雪夫半径1，温度0.1，单次分配，LN eps=1e-5，L2 eps=1e-6；不新增搜索开关。
- 新模式只支持`edge_type=sparse_hybrid`和`integration=token_feedback`。旧模式、旧默认值、旧state_dict键与训练配方不变。
- 不采用09-25旧草案的90%/10%、区域编号MLP、零末层初始化或额外冻结训练。新模块静态新增参数99,840。
- 不在分配路径使用argmax、detach或no_grad。诊断张量必须detach；离散选边沿用旧规则。
- 空间边是锚点邻接先验，不声称学到区域实际接壤、语义分割或后续逐token细化。
- CPU测试只证明相应工程性质；合成数据、smoke分数、历史异协议分数不进入正式收益表。
- 当前工作目录为`/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2`，已核实是独立worktree。保留全部既有未提交文件；没有自动commit、push、reset、stash步骤。任何Git发布另按用户明确授权执行。
- 当前阶段不启动付费GPU任务，不改变数据划分，不覆写旧checkpoint，不通过test选配置。用户继续自行复制服务器单行命令。
- 本计划完成标准为本地实现及证据、短测交付准备；正式训练/多seed/论文消融仍是后续事项。

## Review Focus

1. 新增层消耗随机数改变公共层初值或后续数据顺序：相同seed公共state_dict与CPU RNG尾状态逐项相同，新增参数随seed可重现且不同seed可变化（Task 2、3）。
2. 老权重失效，或新权重丢配置后静默回退旧分组：旧配置原样可用，新配置缺字段/冲突/跨架构续训显式失败（Task 1、4、5）。
3. 只有骨干直通分类获得梯度，被误判分配可学习：用只依赖S的非对称损失，分别证明投影、原注意力和骨干的有限非零梯度（Task 2、3）。
4. 局部mask归一化轴错误、零范数或跨batch混合：S行和1、W列和1、mask外0；零/相反/相同特征有限；另一图片不影响本图（Task 2）。
5. 诊断改变训练RNG/模式或持有计算图；refit误用val探针：有无诊断的训练轨迹一致、异常也恢复状态、refit只取训练池且不读取test（Task 4、5）。

---

## 文件职责与接口总表

路径均相对上述worktree；下表为原计划职责。实现已完成，checkpoint新增覆盖集中在test_soft_gavit.py和test_soft_entrypoints.py，旧test_checkpoint_roundtrip.py原样回归。

| 文件 | 职责 |
|---|---|
| `grouping_identity.py`（新） | 无Torch的唯一版本配置、严格checkpoint检查、GAViT元数据补全 |
| `models/soft_region_grouping.py`（新） | SoftPrototypeGrouping；继承旧注意力层与硬锚点buffer，独立新投影及局部分配 |
| `models/gavit.py` | 显式新分组构造、S矩阵反馈、组合校验；不重写旧分支 |
| `utils.py` | 通用metadata解析保留新配置并拒绝缺失/冲突；旧调用保持行为 |
| `scene_classification/modeling.py`、`runner.py`、`run_scene.py` | 单标签入口、校准/refit身份、按metadata重建、探针调用 |
| `soft_assignment_diagnostics.py`（新） | 有限固定探针、无副作用捕获、每轮统计和必要tensor快照；两个入口共用 |
| `train_bigearth.py`、`test_bigearth.py` | 新模式CLI、身份、重建及训练诊断；不改默认实验配方 |
| `test_gavit.py` | 明确拒绝从旧NWPU入口载入新模式权重，指向scene入口；旧CLI不新增新模式 |
| `tests/test_grouping_identity.py`（新） | 无Torch配置边界与独立字典测试 |
| `tests/test_soft_region_grouping.py`（新） | 分配数学、数值、梯度与初始化 |
| `tests/test_soft_gavit.py`（新） | one-hot等价、真实GAT、公共参数、图构造与重载 |
| `tests/test_soft_assignment_diagnostics.py`（新） | 探针统计、RNG/模式恢复、异常清理与无test使用 |
| `tests/test_scene_runtime.py`、`test_checkpoint_roundtrip.py` | 扩展现有真实CPU smoke及兼容测试 |
| `tests/test_soft_entrypoints.py`（新） | CLI与BigEarth小数据端到端，不只检查源码字符串 |
| `docs/run_soft_assignment_smoke.md`（新） | 已验证版本的交付/服务器短测步骤和完成条件 |
| `results/soft_assignment_local_validation.md`（新） | 命令、环境、成功/失败/未测的证据边界 |

## Task 0: 恢复独立本地测试环境并保留基线

**Files:** Create `results/evidence/soft_assignment_local_20260926/`；不改业务源码。

**Interfaces:** 之后命令使用`/private/tmp/gavit-soft-qa-20260926/bin/python`。现有临时环境已实查缺少torch/pytest/numpy，不将历史89 passed当当前基线。

- [x] **Step 1: 建独立环境并安装固定测试依赖。** 使用已有Python 3.12解释器，不修改用户基础环境；网络受限时通过正常权限机制处理，不绕过。Mac CPU环境与服务器Python3.11/CUDA12.1并非完全同构，必须分别记录。以下版本用于本地兼容测试，不升级服务器环境。

```bash
/private/tmp/gavit-aid-qa-20260919/bin/python -m venv /private/tmp/gavit-soft-qa-20260926
/private/tmp/gavit-soft-qa-20260926/bin/python -m pip install torch==2.2.2 torchvision==0.17.2 timm==1.0.27 torch-geometric==2.8.0 numpy==1.26.4 pillow==10.3.0 pytest==8.3.5 scikit-learn==1.6.1 rasterio==1.4.4 pandas==2.2.3
/private/tmp/gavit-soft-qa-20260926/bin/python -m pip freeze
```

若平台无对应wheel或版本不可获得，先报告具体安装错误，依据可用兼容组合修订环境记录，不能谎称服务器同版本已验证。不得因为环境失败把Tensor测试替换成静态检查后放行。

- [x] **Step 2: 跑修改前现有基线并保存完整退出证据。**

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_experiment_identity.py tests/test_checkpoint_roundtrip.py tests/test_swin_features.py tests/test_graph_construction.py tests/test_sparse_hybrid_model.py tests/test_sparse_hybrid_static_contract.py tests/test_training_state_helpers.py tests/test_scene_data_metrics.py tests/test_scene_runtime.py -p no:cacheprovider -q --tb=short
```

预期exit0；若失败先判断环境或既有问题并记录，不能把新功能引入前的失败归咎于新模块。保存基线所覆盖源码SHA256及测试环境。2026-09-26计划阶段只已跑19项identity+3项static通过，见`results/evidence/soft_assignment_plan_20260926/baseline.json`。

## Task 1: 固定版本身份与严格配置检查

**Files:** Create `grouping_identity.py`, `tests/test_grouping_identity.py`；Modify `utils.py::resolve_arch_config`。

**Interfaces:**
- `soft_assignment_identity() -> dict`：每次返回新字典。
- `validate_soft_assignment(config: dict) -> dict`：必须恰好匹配首版全部字段，严格区分bool/int，拒绝未知/缺失/非有限/错误值。
- `complete_grouping_architecture(arch: dict, *, checkpoint: bool = False) -> dict`：复制输入；新训练缺配置时加入默认；checkpoint新模式缺配置报错；旧模式不得携带非空新配置。
- `architecture['soft_assignment']`仅新模式存在，内含下列全部字段。GAViT新增可选关键字`soft_assignment=None`，旧调用不用传。

- [x] **Step 1: 写RED测试。** 下列pytest测试在新文件内完整定义；补充参数化逐键删除/篡改，而不是只测试version。

```python
import copy
import pytest
from grouping_identity import soft_assignment_identity, complete_grouping_architecture

def test_old_architecture_is_unchanged():
    old = {'grouping': 'attentive_spatial', 'num_regions': 16,
           'edge_type': 'sparse_hybrid', 'integration': 'token_feedback'}
    assert complete_grouping_architecture(old, checkpoint=True) == old
    assert 'soft_assignment' not in old

def test_new_checkpoint_cannot_invent_missing_configuration():
    arch = {'grouping': 'soft_prototype', 'num_regions': 16,
            'edge_type': 'sparse_hybrid', 'integration': 'token_feedback'}
    with pytest.raises(ValueError, match='soft_assignment'):
        complete_grouping_architecture(arch, checkpoint=True)
    full = complete_grouping_architecture(arch)
    assert full['soft_assignment'] == soft_assignment_identity()
    for key in soft_assignment_identity():
        broken = copy.deepcopy(full)
        del broken['soft_assignment'][key]
        with pytest.raises(ValueError):
            complete_grouping_architecture(broken, checkpoint=True)
```

- [x] **Step 2: 跑RED。** `/private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_grouping_identity.py -q`；预期缺模块失败。
- [x] **Step 3: 实现唯一配置与校验。** 禁止散落两套默认。字典保存于内部常量，通过`dict(...)`返回；类型与值都比较，float必须有限。三种不支持组合K9/knn/fusion构造前拒绝。旧分组名称校验仍由现有模型承担。

```python
_SOFT = {
    'assignment_version': 'local_prototype_cosine_v1',
    'assignment_dim': 128, 'assignment_radius': 1,
    'assignment_metric': 'chebyshev', 'assignment_temperature': 0.1,
    'assignment_iterations': 1,
    'assignment_projection': 'shared_ln_linear_no_bias',
    'assignment_init': 'xavier_uniform',
    'assignment_norm_eps': 1e-6, 'assignment_ln_eps': 1e-5,
    'prototype_pooling': 'fixed_region_attention',
    'region_pooling': 'attentive_log_membership',
    'feedback_mapping': 'soft_membership',
    'graph_node_geometry': 'anchor_index_prior',
}
```

`resolve_arch_config`先保留现有旧字段/CLI冲突规则。若metadata含新grouping或soft_assignment，调用严格完整性检查，保留完整嵌套配置；metadata=None且CLI选新模式直接报错（新模式测试必须有sidecar）。训练端通过`complete_grouping_architecture`补配置，不走无sidecar测试回退。不能把soft_assignment加入全体旧defaults导致老权重缺字段。

- [x] **Step 4: 跑GREEN及回归。**

```bash
/private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_grouping_identity.py tests/test_experiment_identity.py tests/test_checkpoint_roundtrip.py tests/test_training_state_helpers.py -p no:cacheprovider -q
```

额外断言：修改返回字典不污染之后调用；新字段传给旧分组失败；None/NaN/True/额外key失败；旧checkpoint无新字段仍重建；CLI新旧grouping冲突失败。完成后检查diff，不自动提交。

## Task 2: 可微局部原型分组及数值验证

**Files:** Create `models/soft_region_grouping.py`, `tests/test_soft_region_grouping.py`。只读取`models/region_grouping.py`，不改原实现。

**Interfaces:**
- `SoftPrototypeGrouping(num_regions=16, grid_size=7, feat_dim=768, soft_assignment=None)`继承`AttentiveSpatialGrouping`，保留`assignment/membership/attn.*`名称。
- `forward(tokens: Tensor[B,49,768]) -> (regions: Tensor[B,16,768], membership: Tensor[B,49,16])`。
- `_assignment(tokens, scores) -> (log_s, w0, prototype_norms, normalized_prototypes)`；供内部实现和梯度测试使用。
- `_pool(tokens, scores, log_s) -> (regions, w)`；one-hot测试明确调用此边界。
- `assignment_observer = None`可选回调：仅诊断时传递本次S/W/W0/原型范数/原型cosine/输入范数；禁止模块保存带图张量。

- [x] **Step 1: 写归一化、独立batch、无旁路梯度RED测试。**

```python
import torch
from models.soft_region_grouping import SoftPrototypeGrouping

def test_local_probabilities_and_assignment_path_gradients():
    torch.manual_seed(42)
    layer = SoftPrototypeGrouping()
    x = torch.randn(2, 49, 768, requires_grad=True)
    scores = layer.attn(x)
    log_s, w0, norms, prototypes = layer._assignment(x, scores)
    regions, w = layer._pool(x, scores, log_s)
    s = log_s.exp()
    torch.testing.assert_close(s.sum(-1), torch.ones(2, 49))
    torch.testing.assert_close(w.sum(1), torch.ones(2, 16))
    assert not s.masked_select(~layer.candidate_mask[None]).any()
    assert regions.shape == (2, 16, 768)
    cost = torch.randn_like(s)  # 非对称；sum(S)恒定不可作为梯度测试
    grads = torch.autograd.grad((s * cost).sum(),
        (x, layer.assignment_projection[1].weight, layer.attn[0].weight))
    for grad in grads:
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
```

同时写测试：变动batch1不改变batch0；相同输入eval重现；修改候选原型对应token使预选S变化（确定性固定输入，不比较所有token）；S范围/总和及W范围，K/N/D/dtype校验。不把候选以外的变化要求为全局完全无影响，因为共享注意力参数和局部原型是机制的一部分。

- [x] **Step 2: 跑RED。**

```bash
/private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_soft_region_grouping.py -q
```

- [x] **Step 3: 实现投影与候选buffer。** 先`super().__init__`产生与旧层完全一样的attn和buffer；在独立CPU RNG作用域内新建并Xavier初始化投影。基于`torch.initial_seed()`派生seed但只`torch.random.default_generator.manual_seed(...)`，不要在`fork_rng(devices=[])`内调用会影响全CUDA RNG的`torch.manual_seed`。所有层构造在CPU后统一`.to(device)`。

```python
with torch.random.fork_rng(devices=[]):
    seed = (torch.initial_seed() + 0x534F4654) % (2**63 - 1)
    torch.random.default_generator.manual_seed(seed)
    self.assignment_projection = nn.Sequential(
        nn.LayerNorm(768, eps=1e-5), nn.Linear(768, 128, bias=False))
    nn.init.xavier_uniform_(self.assignment_projection[1].weight)
region_ids = torch.arange(16)
distance = torch.maximum(
    (self.assignment[:, None] // 4 - region_ids[None] // 4).abs(),
    (self.assignment[:, None] % 4 - region_ids[None] % 4).abs())
self.register_buffer('candidate_mask', distance <= 1)
```

- [x] **Step 4: 实现计算，使用log域防止先log(0)造成梯度NaN。** scores来自旧attn；masked softmax保证每区域有原token且每token有局部候选。非有限tokens/scores/投影输出显式ValueError，不用nan_to_num掩盖。

```python
w0 = scores.expand(-1, -1, 16).masked_fill(
    self.membership[None] == 0, -torch.inf).softmax(dim=1)
z = F.normalize(self.assignment_projection(tokens), dim=-1, eps=1e-6)
raw_prototypes = torch.einsum('bnk,bnd->bkd', w0, z)
prototypes = F.normalize(raw_prototypes, dim=-1, eps=1e-6)
logits = torch.einsum('bnd,bkd->bnk', z, prototypes) / 0.1
log_s = logits.masked_fill(~self.candidate_mask[None], -torch.inf).log_softmax(-1)
w = (log_s + scores).softmax(dim=1)
regions = torch.einsum('bnk,bnd->bkd', w, tokens)
```

`prototype_norms=raw_prototypes.norm(dim=-1)`，不是单位化后的恒1。内部log_s的mask外-inf是合法表示，不用全finite断言误报。对外S/W/R必须有限。observer仅在非None时以detach张量调用；正常forward不保留诊断缓存。

- [x] **Step 5: 添加one-hot和退化输入测试再跑GREEN。** 用`torch.where(one_hot.bool(), 0., -torch.inf)`构造log_membership，`_pool`与原AttentiveSpatialGrouping复制公共权重后输出一致。零/全相同/正负相反输入都有限，逐个backward检查有限（退化输入允许零梯度）。验证原区域token计数为4/2/1、候选非空、所有16区域均有有效pool项；NaN/inf/整数输入/空batch/错误维度应清晰失败。验证同seed投影一致、不同seed改变；构造旧/新分组后`torch.get_rng_state()`相同。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_soft_region_grouping.py -p no:cacheprovider -q
```

## Task 3: 接入GAViT软反馈并证明旧路径兼容

**Files:** Modify `models/gavit.py`；Create `tests/test_soft_gavit.py`；Extend `tests/test_checkpoint_roundtrip.py`。

**Interfaces:** `GAViT(..., soft_assignment: dict | None = None)`；保存`self.grouping`作明确dispatch，新grouping调用Task 2，其余调用顺序保持不变。Task 1生成的完整architecture可直接`GAViT(**arch, pretrained=False)`。

- [x] **Step 1: 写公共初值/整模型反馈RED测试。** 新测试使用无下载、输出49×768的确定性小backbone；带空间变化且有可学习参数，不能复制同一个token49次用于梯度测试。

```python
class TokenBackbone(torch.nn.Module):
    hidden_dim = 768
    def __init__(self, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(3, 768)
    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
        return self.proj(x.flatten(2).transpose(1, 2))

def test_shared_initialization_and_rng_are_identical():
    from unittest.mock import patch
    from models.gavit import GAViT
    with patch('models.gavit.SwinBackbone', TokenBackbone):
        torch.manual_seed(42)
        old = GAViT(pretrained=False, edge_type='sparse_hybrid')
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        new = GAViT(pretrained=False, edge_type='sparse_hybrid', grouping='soft_prototype')
        assert torch.equal(old_rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    assert sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()) == 99840
```

- [x] **Step 2: 跑RED。**

```bash
/private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_soft_gavit.py -q
```

- [x] **Step 3: 加显式构造和反馈分支。** 新组合在backbone构造前验证。旧gather实现按原代码保留；不能仅凭assignments.ndim猜分组。

```python
if self.grouping == 'soft_prototype':
    # 先检查floating、[B,49,16]、finite、非负、行和及候选合法性。
    token_feedback = torch.einsum('bnk,bkd->bnd', assignments, region_feedback)
else:
    idx = assignments.unsqueeze(-1).expand(-1, -1, region_feedback.shape[-1])
    token_feedback = torch.gather(region_feedback, 1, idx)
updated_tokens = tokens + token_feedback
```

保持旧融合、旧knn/hybrid、graph builder、GAT等所有分支；代码使用原forward中反馈变量实际名称，勿重复投影。

- [x] **Step 4: 完成可解释兼容测试。** 新模型eval时仅在测试用例替换`_assignment`的log_s为原one-hot（原型辅助结果按真实helper取得），验证R、图边、token_feedback及最终logits与旧模型一致。新模型原始soft模式用实际PyG前向/backward、BCE和CE两种loss验证有限；独立仅S损失证明backbone.proj非零梯度。挂现有`capture_graph_once`检查每图48+32、无跨batch、无重复、邻居→query；已有graph builder测试继续验证top2方向与排除关系。旧/新checkpoint严格重载eval logits，dtype/shape损坏不能静默gather。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_soft_gavit.py tests/test_sparse_hybrid_model.py tests/test_checkpoint_roundtrip.py tests/test_graph_construction.py -p no:cacheprovider -q
```

## Task 4: 无副作用诊断与单标签完整生命周期

**Files:** Create `soft_assignment_diagnostics.py`, `tests/test_soft_assignment_diagnostics.py`；Modify `scene_classification/modeling.py`, `scene_classification/runner.py`, `run_scene.py`, `tests/test_scene_runtime.py`。

**Interfaces:**
- `architecture(model, num_classes, grouping=None) -> dict`：swin仅接受None；gavit None沿用attentive_spatial，新模式通过Task 1补全配置。
- `build_model(model, num_classes, pretrained_path=None, *, architecture_config=None)`：None沿用旧构造；非None须完整且等于支持的架构，直接用其新配置建模。
- `RunConfig.grouping: str | None = None`；train增加`--grouping {attentive_spatial,soft_prototype}`，evaluate不新增覆盖架构开关，以sidecar为准。
- `SoftAssignmentProbe(model, inputs, sample_ids, output_dir, split, *, resume=False)`：只接受soft模型、固定无增强输入与train/val/pool；`record(epoch, *, save_tensors=False) -> dict`；test直接拒绝。内部状态只有CPU detach的初始S和探针输入，不保存autograd图。resume=True须从持久化epoch0快照读取初始S并比对probe_manifest；缺失或身份不匹配即失败，不能重新定义初始S。
- 捕获字段：S、W、W0、原型范数、prototype cosine、token L2范数、feedback_proj输出；只在`record`期间注册，finally解除observer和hook。

- [x] **Step 1: 写诊断无副作用RED测试。** 执行同一随机训练step前插入或不插入probe.record，参数/optimizer状态应一致。包括模型内部某子层单独eval的情况，逐module恢复原training标志；不能简单最后model.train()覆盖原状态。测试record中故意observer异常也还原RNG/mode/hooks；split=test应在forward前失败。

```python
import torch

def test_probe_preserves_rng_modes_and_rejects_test(tmp_path):
    from unittest.mock import patch
    from models.gavit import GAViT
    from test_soft_gavit import TokenBackbone
    from soft_assignment_diagnostics import SoftAssignmentProbe
    with patch('models.gavit.SwinBackbone', TokenBackbone):
        model = GAViT(pretrained=False, grouping='soft_prototype', edge_type='sparse_hybrid')
    model.train()
    model.classifier.eval()
    images = torch.randn(2, 3, 16, 16)
    rng = torch.get_rng_state().clone()
    modes = [m.training for m in model.modules()]
    probe = SoftAssignmentProbe(model, images, ['a', 'b'], tmp_path, 'val')
    result = probe.record(0, save_tensors=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert result['epoch'] == 0 and result['samples'] == 2
    assert model.region_grouping.assignment_observer is None
```

同时测试初始变化0、相同模型第二次变化0、更新投影后变化非零、统计可从保存S独立重算。保存CPU tensor的所有`requires_grad=False/grad_fn=None`；hook异常之后再次正常forward不受影响。

- [x] **Step 2: 跑RED并实现诊断边界。**

```bash
/private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_soft_assignment_diagnostics.py -q
```

用`try/finally`保存/恢复每module模式、CPU与实际使用CUDA设备RNG状态。record内部`model.eval()`+`torch.no_grad()`仅用于探针推理，不修改训练forward。每条记录含epoch、split、sample_ids、熵`-sum(S*log(S.clamp_min(1e-30)))`、每区域mass=S.sum(token)、argmax占用（仅诊断）、低原型范数计数（阈值1e-6）、非对角prototype cosine、相对epoch0的mean_abs_delta_S、反馈与token的L2范数及比值。零token范数用显式`ratio=null`并记录zero_input，不写Infinity。

输出`assignment_diagnostics.jsonl`每轮flush；`probe_manifest.json`记录顺序/路径/输入SHA/无增强变换；epoch0、best与final保存最多8图的S/W/W0/原型/图边`.pt`用于后续图稿。各文件初次创建拒绝覆盖；更新best快照用明确best命名替换而非伪造epoch。原始tensor快照不作为模型checkpoint的一部分。单轮探针墙钟独立计时，不混入纯训练吞吐。record已存在epoch拒绝追加；resume发现诊断最后epoch与训练state不一致时明确报错，不截断历史或冒充已恢复。

- [x] **Step 3: 单标签架构传递RED测试。** 扩展现有synthetic真实CPU smoke加入gavit/soft_prototype；用`RunConfig(..., grouping='soft_prototype')`，检查best.meta.json完整配置、实际80边、梯度、checkpoint重建误差≤1e-5、val evaluate成功、test evaluate仍被smoke规则拒绝。逐键删除soft_assignment字段后checkpoint验证应失败；旧swin/gavit仍跑原测试。校准source身份包含grouping配置，fixed→soft、soft→fixed或不同温度复用refit均失败。

- [x] **Step 4: 实现入口/runner贯通。** `_checkpoint_metadata`按metadata中grouping得到支持的expected architecture，并严格比较全部字段；`_restore`将该完整架构传给build_model。训练identity、config、metadata、refit validate_calibration一致使用一次解析后的arch。旧`scene-v1`格式保留；新配置可辨识，旧记录不被自动增加字段。`_code_identity`补入`grouping_identity.py`和`soft_assignment_diagnostics.py`，模型目录的现有glob须覆盖新文件。

```python
resolved_arch = architecture(cfg.model, len(names), grouping=cfg.grouping)
model = build_model(cfg.model, len(names), cfg.pretrained_path,
                    architecture_config=resolved_arch).to(device)
```

仅新模式创建probe：calibrate/smoke按清单val顺序前8图，refit从pool前8图且不建val loader；沿用eval transform、单独装入而不迭代训练loader。记录epoch0、每epoch、best/final；不读取test。旧模式不创建probe且无额外forward/RNG消耗。`run.json`显式列出probe split/数量/文件/耗时。

- [x] **Step 5: 跑GREEN/CLI help。**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_scene_runtime.py tests/test_scene_data_metrics.py tests/test_soft_assignment_diagnostics.py -p no:cacheprovider -q
/private/tmp/gavit-soft-qa-20260926/bin/python run_scene.py train --help
```

本地formal CUDA限制继续生效；不为了测试成功移除真实数据、clean Git、batch32和pretrained检查。对refit架构传递用现有纯身份函数+小模型重建测试；本地不能冒充完成真实30轮calibration→refit。

## Task 5: BigEarth接入及旧NWPU入口清晰拒绝

**Files:** Modify `train_bigearth.py`, `test_bigearth.py`, `test_gavit.py`；Create `tests/test_soft_entrypoints.py`；Extend `tests/test_checkpoint_roundtrip.py`。

**Interfaces:** train/test BigEarth支持`--grouping soft_prototype`；train另外显式传`--edge_type sparse_hybrid --integration token_feedback`。不新增温度等调参CLI。SoftAssignmentProbe直接复用Task 4。

- [x] **Step 1: 写入口/重载RED测试。** 以subprocess运行两个入口`--help`验证接受新grouping；旧`train_gavit.py`不给新choice。合成BigEarth fixture在tmp中用rasterio写B02/B03/B04 TIFF和合法19维标签CSV，复用dataset现有目录/字段格式；跑原入口的1 epoch `run_stage=smoke`，正式模型在CPU、极少样本，pretrained构造用测试进程内替身注入本地初始化，避免网络。父进程超时/退出/缺指标都算失败。不得只mock train/evaluate整个函数。

Fixture严格使用`patch_path,label_0,...,label_18`列；patch_path为临时绝对目录，内含同目录basename前缀的`_B02.tif/_B03.tif/_B04.tif`。每split至少2图，各类至少1正/1负（用交错标签），避免无正例AP被误当流程正常。测试工厂在独立子进程用runpy执行原入口；明确patch timm.create_model的pretrained=False，并把loader workers设0以免macOS spawn重入脚本，该测试边界必须记录，真实workers4在服务器验收。

测试至少包括：Swin旧模式smoke、GAViT固定模式旧checkpoint、GAViT soft模式前后向与best/state/meta；`test_bigearth`从新sidecar完整恢复（使用临时test.csv只作合成工程测试）；缺sidecar或soft字段失败；冲突grouping失败；`test_gavit`遇soft checkpoint应提示使用run_scene而非静默绕过验证。日志和temporary路径与正式checkpoint完全隔离。

- [x] **Step 2: 跑RED。**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_soft_entrypoints.py -q
```

- [x] **Step 3: 接入元数据与恢复。** train先构造当前ARCHITECTURE，再调用Task 1完整化，然后只用完整架构建GAViT。不要训练用一个配置/metadata写另一个；resume沿用现有完整architecture及training identity匹配。test的`resolve_arch_config`返回含soft_assignment的完整arch，使用`GAViT(**arch, pretrained=False, freeze_backbone=False)`，避免原逐参数调用丢失新配置。读取sidecar后必须先校验新架构，再读数据/加载权重执行推理。对于旧test_gavit，检查metadata.architecture.grouping新值即ValueError明确指向scene入口，防止CLI虽不接受但metadata绕过限制。

```python
ARCHITECTURE = complete_grouping_architecture(ARCHITECTURE)
# 仅args.model == 'gavit'调用；Swin身份不添加无关字段。
model = GAViT(**ARCHITECTURE, pretrained=True,
              pretrained_path=args.pretrained_path, freeze_backbone=False).to(DEVICE)
```

- [x] **Step 4: 接入新模式每轮探针并测试resume。** 从固定val清单前8图用val_tf单独加载，路径/输入SHA保存；不遍历随机训练loader、不触碰test。默认把diagnostics放在新checkpoint旁的专属同stem目录。新训练拒覆写；resume先核验同一checkpoint/run tag/架构与探针身份再追加，不能重置epoch0或把上次诊断当新初始S。如果恢复所需epoch0探针状态缺失，明确拒绝新模式resume并给出缺文件，而不是伪造可比delta。

已有save_epoch_artifacts返回的improved用于best快照，最终轮保存final；例外处理恢复模型mode。加epoch/timing/诊断文件路径记录，仅新模式启用，不改变旧train pin_memory=True、val/test False、workers4、TF32/AMP默认和原损失/优化器/scheduler/阈值。针对新smoke用不同输出目录，不调用硬编码seed45正式队列。

- [x] **Step 5: 跑GREEN及既有相关回归。**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests/test_soft_entrypoints.py tests/test_checkpoint_roundtrip.py tests/test_bigearth_seed45.py tests/test_sparse_hybrid_static_contract.py tests/test_training_state_helpers.py -p no:cacheprovider -q
```

未覆盖的真实原图解码/目标GPU资源仍留服务器短测，不用合成RGB TIFF通过冒充完整BigEarthNet已验证。

## Task 6: 汇总回归、交付短测说明并记录证据

**Files:** Create `docs/run_soft_assignment_smoke.md`, `results/soft_assignment_local_validation.md`；Update本设计状态、`research_diary.md`、`docs/thesis_schedule.md`。不把未测结果填入comparison/ablation指标单元格。

**Interfaces:** 本地完整实现交付；服务器正式操作另给经核实的绝对路径单行命令，不能把计划中的假定目录当已部署版本。

- [x] **Step 1: 跑一次完整tests目录并保存log/exit/env/git/file hashes。**

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-soft-qa-20260926/bin/python -m pytest tests -p no:cacheprovider -q --tb=short
git diff --check
```

只统计实际执行通过项；missing-dependency skip、外部fixture缺失、Tensor未跑均单列。不用全tests通过取代实际Torch前后向证据。完整通过后仅在新修改/失败/未解风险时重跑相关测试。

- [x] **Step 2: 逐条核对交付物。** 需有新模型CPU真前后向、assignment-only梯度、旧模型公共初值与one-hot等价、metadata/严格重载、scene真实synthetic smoke、BigEarth入口工程smoke、graph80边与diagnostics无副作用证据。整项缺一就记录未通过范围，不把计划勾成完成。特别复核双重归一化、mask外-inf只在内部、独立RNG不影响CUDA，以及refit没有test/val探针泄漏。
- [x] **Step 3: 写具体服务器短测步骤。** 顺序固定：读取服务器Git/数据现状→独立发布目录及固定commit（发布需另获授权）→NWPU prepare验证5040/1260/25200→soft scene 8train/2val每类1 epoch→BigEarth96train/64val1 epoch。NWPU本地原图及下载尚未确认，当前不编造已存在路径或下载链接。保持AID和BE既有成果；BE split复制临时小样本，冻结正式CSV不变。两次短测须记录完整启动命令、seed42、pretrained SHA、首batch、实际GPU/显存/耗时、图边、checkpoint hash/reload diff、退出码。GPU未实测时不报价总时长。
- [x] **Step 4: 记录下一批的研究问题，不自动启动。** NWPU三模型seed42使用同一manifest、augmentation、CE、batch32、AdamW3e-4/wd1e-4/Cosine30，先val OA校准；候选有效性与资源明确后再冻结refit/test及BigEarth完整30轮。已有同协议checkpoint先审计是否可复用，不重复付费。正式多seed和HGTNet/DFAGCN仍独立待完成，不因本地通过而标论文完成。
- [x] **Step 5: 形成审阅diff和最终说明。** 清楚列出做了什么、实际测试、未测CUDA/性能及后续用户只需的服务器动作。不自动Git提交，不把新图稿设计或新性能写成已完成。

## 实施顺序、完成边界与本次自检

推荐在当前任务由助手直接顺序实施Task 0→6。任务共用架构身份、模型返回值和runner，逐任务交接容易产生接口漂移；直接实施更适合此次紧密关联的小范围改造。最后按所选执行技能做整体代码审阅，不默认并行开发。

本计划已对照09-26设计自检：计算/梯度/数值在Task 2；初始化/软反馈/图在Task 3；身份/旧权重/CLI在Task 1/4/5；固定探针与每轮记录在Task 4/5；双数据集短测及论文边界在Task 6。新增嵌套字典只是metadata组织方式，所有设计字段均保留；温度/参数量/边数和首批三模型未变。

2026-09-26本地实施状态：Task 0至6完成，独立CPU环境已恢复，代码和入口已实现；最终完整190项测试通过、exit0。一次独立审阅及其受控中断快照修复已归档。新checkpoint覆盖放在独立soft测试文件，既有roundtrip测试保持原样。当前HEAD未变，本次仍为未提交工作文件，不能拿基线commit当新发布版本。GPU短测、精度和正式多seed不在本次完成范围；见[本地验收](../../../results/soft_assignment_local_validation.md)及[服务器短测说明](../../run_soft_assignment_smoke.md)。
