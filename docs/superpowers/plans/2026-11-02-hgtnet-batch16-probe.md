# HGTNet batch16 capacity probe plan

**Goal:** Verify paper batch16 under the full independent HGTNet on the current3090 before any long formal run. This continues the accepted smoke→resource→formal preparation workflow.

**Protocol:** Frozen model/config/Swin-B/manifest/seed42/FP32, no AMP/TF32; deterministic16 training images (one first image for classes0–15), no val/test. Same input shape224/ImageNet normalize as accepted smoke, not a frozen formal augmentation recipe. Adam lr1e-5/weight_decay5e-5, local explicit betas0.9/0.999 andeps1e-8. One warmup+three timed optimizer steps reusing the16 inputs on one disposable model. No epoch, accuracy, checkpoint or continuation of learned smoke weights.

**Measurement:** Synchronize CUDA boundaries; report each measured step, median/mean and compute-images/s. Input decoding/copy occurs before timed steps, so this is not dataloader/epoch throughput. Memory scope includes current model/input/optimizer and diagnostics, excluding duplicate reload model. Existing batch2 history is a separate observed pipeline rate, not a batch16 prediction.

**Files/interfaces:** batch_probe.py entry/preflight/data/status; batch_probe_engine.py run_batch_probe(model,images,labels,*,device,batch_size16,warmup_steps1,measured_steps3,on_progress); registered builder runner_member; profile3090/currentUUID/newoutput/scratch. Core model, data, smoke engine, common CUDA memory fix and precision gates remain unchanged. Existing450 recovered images are verified and reused when present, otherwise recover only16 train images from fixed ZIP.

**Local test runtime:** newly isolated /private/tmp/gavit-hgtnet-probe-cpu-20261102 with exactTorch2.2.2/vision0.17.2/NumPy1.26.4/Pillow10.3.0; Python3.12.13 local vs3.11.8 target recorded. No change to old environments or server.

- [x] Entry RED→GREEN: selection16/trainonly, inputbytechecks/cache/ZIP fallback, preflight ML isolation/newpaths/currentUUID and failure after observed update.
- [x] Engine RED→GREEN with real CPU tensors:4 updates, finite gradients and group updates, input preservation, measurement scopes, badinputs/failure propagation.
- [x] Builder registered runner RED→GREEN; default smoke behavior intact, no arbitrary entrypoints.
- [x] Independent review;333 HGT/DF related numerical tests in restored runtime;104 preservation/identity checks pass.
- [x] Build fresh61006B pyz and actual stdlib inspect/status; recordidentity, history arithmetic, diary and copy-paste instructions. Target batch16 result remains pending, no remote launch.

User history receipt is archived separately; no repeat of the accepted360/90 smoke. Original paper crop/flip/scheduler/etc unknowns remain disclosed, and this probe cannot choose model/seed/shape from a score. No Git publication or destructive cleanup.
