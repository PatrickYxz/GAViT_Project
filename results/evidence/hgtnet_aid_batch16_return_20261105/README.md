# 实际RTX3060 AID batch16完成回传

2026-11-05。来源为用户粘贴的run&&status输出；[转录](reported_output.txt)保留字段内容，status JSON的空白布局作了规范化，不是服务器run.log原件。[status](status_reported.json)、[四步](progress_reported.json)、[10项核对](validation.json)和[工程记录](../../hgtnet_aid_batch16_probe_3060_20261105.json)已保存。未直连服务器或下载原产物。

用户回传state complete、exit0、completion_evidence_valid/source_identity_matches true；同一RTX3060 UUID GPU-df386e6c-dbb3-0952-8190-3265a7bd8ae8，16张原train图复用4次，共64次呈现、4次实际更新、1warmup+3measured。首步五组451个梯度张量无缺失/非有限值、组L2正、代表参数已更新；四步postcheck均通过。原manifest/config/初始化身份一致，包SHA仍fd54b4c1a21db1731d1addb0808c49104c11113b610883b79b2c6af456fc80f5。

峰值allocated4731960320B=4.406981468200684GiB，reserved5213519872B=4.85546875GiB。三步0.4692300269998668、0.46906283299995266、0.47697647499990126秒，中位0.4692300269998668秒，计算34.09841459273138图/秒；总wall13.486246307999863秒。该计时不含完整loader、读图、增强、传输、callback或参数诊断，warmup排除；峰值从设备放置后reset，含常驻模型/输入、更新与诊断，不代表更早初始化或正式全部阶段峰值。

容量工程门禁通过，可继续实现独立正式AID包。没有新增epoch/验证/test/checkpoint/accuracy或正式训练；不将短探针换算成完整正式ETA、费用或性能。完整pool5000与磁盘/hostRAM、实际正式首batch仍须由新包实际门禁核验。
