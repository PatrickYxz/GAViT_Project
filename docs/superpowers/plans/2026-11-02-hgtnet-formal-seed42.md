# HGTNet formal seed42 Implementation Plan

> For agentic workers: execute the independent data, engine and entry tasks in parallel, then independent review and numerical verification. Continuation of the user-authorized external-baseline reproduction; prepare the reviewable local package and do not remotely launch GPU work.

**Goal:** Deliver an offline3090 run/inspect/status/resume package for frozen calibration→fresh-refit→final-test, following successful real smoke and batch16 checks.

**Architecture:** Keep old numerical/source/config/pyzs immutable. Add formal_protocol (stdlib), formal_data (stdlib selection/recovery plus lazy datasets), formal_checkpoints (safe atomic two-slot epoch recovery/model reload), formal_engine (numerical stages), formal_entry (stdlib preflight/status and lazy orchestration). Extend registered builder for formal runner/config only.

**Tech Stack:** Python3.11.8/Torch2.2.2+cu121/vision0.17.2/NumPy1.26.4/Pillow10.3.0 on3090; localPython3.12.13 pinnedCPU environment for fixtures.

**Spec:** docs/external_baselines_20261027/hgtnet_formal_preparation_20261102.md and this frozen decision record. No further user approval loop: unknown details below are disclosed independent engineering choices, not author-confirmed defaults.

## Global constraints and frozen decisions

- Source HGTNet74619858params, configcanonical af6ef410...ac25, initialcandidate68c6b09e...1a893, manifestce0cfef2...840e, seed42; current3090 physicalUUID selected by inspect; FP32/noAMP/noTF32/8CPUthreads.
- Real counts train5040,val1260,pool6300,test25200,45classes. Calibration100epochs,batch16,no drop; Adam1e-5 WD5e-5/betas(.9,.999)/eps1e-8; constantLR/no warmup or scheduler, CE/no smoothing, no actual gradient clipping. Whole method, no single-module causal comparison.
- Engineering augmentation fixedbeforetest: RGB Resize(256,256) bilinear→RandomCrop224→RandomHorizontalFlip(.5)→ImageNet normalize for train/pool; evalResize(256,256)→CenterCrop224→normalize. Random crop/flip are paper-mentioned; precisechoices unconfirmed. workers0, no pinnedmemory, evalbatch16. Per-epoch Torch seed42+epoch and shuffle generator seed42+epoch permit deterministic epoch restart.
- Validation primary correct-count/OA; strict greater count updates best, exact tie keeps earliestepoch. Retain best+finalcalibration and all epoch metrics. Fresh refit from equal initial state SHA, exactly selectedbestepoch on6300pool. Test firstdecoded/evaluated only after finalrefit and numeric finalcheckpoint reload; save OA/macroF1/perclass/cm/predictions/orderpaths. Never select hyperparameters/seed using test.
- Epoch recovery two alternating snapshot slots plus per-slot SHA receipts; each payload contains binding/model/Adam state, phase/epoch/generation/histories/selection/initialstate. Resume picks highest valid compatible completed epoch, restarts incompleteepoch deterministically. Preserve completed best/final model receipts. No external logging/dependency install or automatic GPU launch.
- work6GiB/scratch2GiB conservative separate budgets, same-volume add; fresh output/scratch, resume only matching owned binding. Candidate/manifest/recipe/source full hashes rechecked beforetest and completion; previous450imagecache cannot satisfy full6300 pool. Dataset per-access hashes exact bytes decoded. Archive stage onlypool first, test recovery deferred.

## Interfaces

protocol: RECIPE dict, recipe_sha256(recipe), load_recipe(path), validate_binding(binding)→JSON-normalized binding. Real fixed exactRECIPE/counts/45classes/deviceCUDA viaengine; cpu_fixture explicitly separate withsmallpositivecounts/maxepochs/batch, no realmetric claims.

data: prepare_inputs(manifest_path,expected_sha256=MANIFEST_SHA)→{class_names,counts,selected:{train,val,pool,test}}; validate_selected(root,records); recover_selected(archive,records,destination,expected_sha256); make_dataset(root,records,training=False). Test records supported only in newformalmodule, oldsmoke helpers unchanged. Test recovery receipt counts truthful; existingdestination onlyreused after full matching byte/receipt checks.

checkpoints: atomic_json(path,value); state_sha256(model); save_snapshot(output,payload)→receipt; load_snapshot(output,binding)→(payload,receipt); apply_model_state(model,state); save_model(model,path,*,binding,metadata)→receipt; load_model(model,path,*,expected_sha256,binding)→receipt. SafeTorch weights_only, finite/dense/dtype/shape/key checks and fullSHA before copy.

engine: run_formal(model_factory,loader_factory,output,*,binding,device,on_progress=None,resume=False)→summary. model_factory() fresh CPUmodel fromseed42+verifiedcandidate everycall; loader_factory(split,epoch) accepts train/val/pool/test and returns fixedcount DataLoader; only test calledafterbefore_test callback. Engine owns initializationSHA, phase/epoch/eval metrics/checkpoints/history/test predictions/numerical reload. Progress events include stage,phase,epoch,optimizer_steps,training_started,formal_training_started and firstsuccessfulupdate proof; callbackstagebefore_test performsentry inputchecks. Engine stages canfixture2epochs onCPU only; no CLIoverride for real100/batch16.

entry: parser(), inspect_resources(args), run_formal_entry(args,resume=False), status(output), main(argv). Newrun createsoutput/scratch onlyafterstdlibready/currentUUID, runtime COMMON gate andbinding. _run_model(args,torch,runtime,inputs,binding,progress,resume) buildsdata closure/freshfactory thenengine. Resume verifiesbindingownership andsource/data flags, preservesattemptlogs andnewruntime; no source edit. Wrapper completes onlyafterall stages/source/inputcheck, summary/exit0/marker; preserves first step andactualcounts onfailure. status alwaysstdlib.

## Task1 data (agent)

- [x] RED tests test_hgtnet_formal_data.py: fullmanifest/countorder/disjointness; train/eval transforms; peraccesshash; testZIP truthful receipt; nooverwrite/tamper; noMLinspect.
- [x] Implement formal_data.py using frozen helper manifest verification, custom truthful recovery, lazyTorch/PIL transforms, bytehashanddecode samebytes.
- [x] GREEN targeted unittest, evidence formal_prepare/data/; no protectedoldsource edits.

## Task2 checkpoints/protocol (root)

- [x] RED tests test_hgtnet_formal_checkpoints.py for roundtripoptimizer/model, badSHA/binding/state, newestcorrupt→priorvalidslot fallback, atomicownedpaths andequalinitstate.
- [x] Implement protocol/recipeJSON and checkpoints API above; GREEN CPUfixtures andsourceidentity receipts. No learnedsmoke initialization.

## Task3 engine (agent)

- [x] RED tests test_hgtnet_formal_engine.py: earliesttie, fresh-refit identity, exact refitbudget, untouchedtestuntilrefit/reload, allmetricoutputs, epochresume equivalence, failedforward/postupdate state, validationcount/order checks.
- [x] Implement formal_engine.py withabovecheckpoint APIs; preserve actual updateevents and phase clocks/throughput; report firstepoch-based fullpipeline projection as estimate, not billedtime.
- [x] GREEN realCPU small45class tensor fixtures (notGPU/NWPU); evidence formal_prepare/engine/.

## Task4 entry (agent)

- [x] RED tests test_hgtnet_formal_entry.py: stdlibpreflight, fullcounts, missingimages ZIPpending, version/UUID/capacity/newpaths/resumeownership, failureafterfirstupdate, inputmutation beforetest, actualstatus.
- [x] Implement formal_entry.py consumingaboveinterfaces; pretrainingreports source hashes/params; firststeprecord, per-attemptlog, sourcebinding excludingmutableGPUUUID; weights/config/manifest/recipe immutable.
- [x] GREEN orchestration tests; evidence formal_prepare/entry/.

## Task5 package/integration (root then freshreview)

- [x] Restricted formalrunner + extra recipe payload TDD. Preserve olddefault APIs/3oldHGTpackages/52historicalpaths/core/config/candidate bytes.
- [x] Archive returnedbatch16status, validate allidentities/timing/peaks; realbatch16 resourcegate accepted, no repeatGPU.
- [x] Run fullHGT/DFrelatedsuite and one full-modelCPU tinyJPEG stagedrun+checkpointreload; verifyprototype labels/modes/pathsexplicit. Freshreview source/data/selection/resume/bindings andfailureedges; repairimportantissues.
- [x] Buildnew hgtnet_nwpu_formal_3090_20261103_v1.pyz, actualstdlibMLguard inspect/status, payloadhash checks, README/diary/formalrecipe/resultrecord updated. Target formalrunpending; provideoneinspectcommand, generatedrun/currentUUID, readstatus/resumeguide. NoGitpublication.

## Review focus

- Interrupted snapshot/receipt pair must fall back toothercompleted slot, never silentlyrestart at0 orinfercompletedtraining fromPID.
- Validationtie cannot change refitbudget; test cannot be decoded/evaluated before selectedbudget, fresh initialization andfinalreload.
- Snapshotbinding/source/recipe changes or imagebytes mutated reject beforecontinuedtraining/test; different3090UUID mayresume only sameexactsoftware/profile withnewsessionreceipt.
- Newrun/scratch/oldsmoke checkpoints neveroverwritten; twoslots/atomictemps budget preventsper-epoch checkpoint disk growth.
- CPUfixture cannot label its metrics realNWPU; computeprobe79.46images/s/3.97h areconditional, fullpipeline actualepochs separatelyobserved.

## Final verification 2026-11-03

386 related tests pass; full_model_cpu_fixture_03 completed4 updates, equal initialization SHA and numeric reload0; entry14/data13/checkpoints7/engine15/delivery4 new tests. Additional review fixes freeze restoredAdam policy/moments, exact failure counters, partialpool/atomicreceipt and committedbest prediction packet. RealGPU formal task remains pending; old artifacts preserved, no Git publication.
