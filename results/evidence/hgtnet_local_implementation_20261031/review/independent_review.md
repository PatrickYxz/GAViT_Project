# Independent HGTNet integration review

Reviewed the current config, attention, graph, model, pretraining, checkpoint and
local smoke/CLI modules against the approved local design and implementation
plan. This review did not modify implementation or preexisting user files.

**Result:** no unresolved Critical, Important or Minor implementation findings in
the scoped local candidate. This is not acceptance of exact author reproduction,
GPU execution, dataset accuracy, a formal trainer or cross-version portability.

- The visual path contains real torchvision attention blocks in 2/2/18 counts,
  alternating shifts, channel widths 128/256/512 and all three specified spatial
  downsamplings. DA combines the two gates by addition. The new stage-3 Conv2
  and graph transition padding are explicit independent choices.
- Graph groups use six and two actual dynamic blocks, the current projected
  features, detached Euclidean kNN, neighbor-minus-query messages, explicit
  max/mean choice, excluded self edges and strict k=20. The 49/25-node geometry
  remains legal without clipping. Nodes are restored to BCHW before transition2.
- The two 512-channel global means are summed before the 45/30-class linear head.
  Configuration identity protects mutations and the ordinary constructor has no
  download or implicit pretrained load.
- Pretrained mapping covers the specified first three torchvision Swin-B
  feature stages, two patch merges and stem. New modules and skipped source
  stages are explicit. File loading requires a full measured SHA, the official
  prefix, complete official source schema and all finite source values before
  mutation; schema-only results cannot apply.
- Checkpoint loading validates bytes, safe tensor deserialization, exact schema,
  config/source identity, all keys/shapes/dtypes/layouts and finite values before
  strict copying. Output creation is exclusive, including symlink destinations.
  The report clearly distinguishes saving from loading and carries no optimizer
  resume or accuracy claim.
- The synthetic smoke checks actual B1/B2 traces and all trainable gradients,
  requires positive gradient norms and a changed representative in each of five
  groups, and checks updated parameters for finite values. Completion is false
  until a distinct process validates checkpoint outputs. This code review alone
  is not evidence that the full smoke executed; parent-owned runtime logs are
  required for that claim.

Independent targeted probe: `integration_probe.py`, exit 0, actual small CPU
tensors. It verified integrated graph layout restoration, exact equality of
trace/ordinary paths, digests of actual indices, sample isolation, finite
backpropagation, a checkpoint roundtrip under actual six-file source identities,
and refusal to overwrite the checkpoint. Results are in `integration_probe.log`.
This probe deliberately does not substitute for the full candidate's numerical
CPU run.

The source identity binds six local model-defining files. External library
versions are recorded separately in the smoke runtime; numerical verification
must be repeated before carrying results across a changed Torch/torchvision
runtime. This does not invalidate the current pinned-environment local scope.
