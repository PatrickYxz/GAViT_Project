# Real formal startup / first epoch return

State: **SERVER_REPORTED_FORMAL_STARTED_EPOCH1_COMPLETE_EPOCH2_TRAIN_COMPLETE**. Evidence is user-pasted terminal output, not a live remote inspection.

- [Byte-exact original](reported_output.txt):9556B, SHA3184766b0014b350f5424fe1415d0363f95506613fc65e770749ee6da9c43bef. Contains exact v1 command/currentUUID, CUDA report and firststep/epoch events.
- [Local22-check acceptance](validation.json) and [independent22-check review](review/validation.json) pass. Actual delivered localbundleSHA b2bc9077...572e1 andenginebytes match. This verifies local code semantics; remote runtime/snapshot files were not retrieved.
- First actualcuda:0 step passed all5group/451tensor gradient checks andrepresentativeparameterupdates; PID2661/attempt1,formal_training_startedtrue. This exceeds a PID-only startup claim; no separate remoteGPUprocessquery/peakmemory wasreturned.
- Epoch1 complete315updates/5040train, online trainOA47.9167% (arithmetic2415/5040),trainloss2.1766647956. ValidationOA77.3015873% (arithmetic974/1260),macroF176.8132144% serverreported only,not independentlyrecomputedfrompredictions. Normal code writes bestmodel/epochsnapshot/history beforeepoch_complete.
- Epoch1 train250.583607628s/20.11304749images/s,val8.641767578s,wall326.696851827s. Other67.471476621s covers combineddatasetsetup/checks/checkpoint/historyIO/synchronization; exact bottleneck unprofiled. Epoch2 train236.359086964s/21.32348735images/s,global630updates; noepoch2val/snapshotcomplete/wall event inthispaste.
- Reportedfirstepochremainingcalibration32342.988330873s=8.98416h;100epochs atfirstwallrate9.07491h. This was the estimate at that event, withno server timestamp to inferpresent remainingtime; excludes freshrefit/test andnot guaranteed/billedtime. Earlier3.97h fromfixedinputbatch16compute was never fullpipelineETA; useactualcompleteepochstatistics goingforward.
- Nofinaltest/complete marker/run.exit0,finalbest/refitbudget/peakresource/currentremotestate confirmed. Keep existing frozenjob running; no rerun/recipe/source changes fromthisreceipt.

[Machine progress](../../../results/hgtnet_nwpu_formal_progress_20261103.json) / [readable progress](../../../results/hgtnet_nwpu_formal_progress_20261103.md). NoGPU/localnumericaltraining, Gitpublication, externalmessage orformalperformance-tableedit wasmadehere.
