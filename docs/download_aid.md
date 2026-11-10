# AID服务器直接下载

日期：2026-09-20。用户在Featurize未找到AID；代码版本2ba112a已由服务器输出确认。

下载源为[Vision-Flan保存的AID镜像](https://huggingface.co/datasets/Vision-Flan/vision-flan/commit/a7fc1f3307439d00e39acb9a944356b346531ae9)，固定revision a7fc1f3307439d00e39acb9a944356b346531ae9。不是AID作者自己的下载站；原数据身份参照[AID官方页](https://captain-whu.github.io/AID/)。

本地HTTP探测返回可下载ZIP，2628474154字节。仅读ZIP中央目录和三张图：恰30个类别目录、10000个图像条目，三张均600×600 RGB。[检查记录](../results/evidence/aid_source_probe_20260920/probe.json)。没有下载完整包、没有独立计算完整SHA256或逐图解码；发布方LFS SHA256为`cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7`，需服务器下载后核对。

## 下载与校验

压缩包保留在持久化work目录，重启/换实例后可复用。网络可用性只在本地验证，Featurize能否访问仍以执行结果为准；若网络失败保留输出，不关闭TLS校验。

```bash
wget -c --timeout=30 --tries=3 -O /home/featurize/work/AID_visionflan_20260920.zip https://huggingface.co/datasets/Vision-Flan/vision-flan/resolve/a7fc1f3307439d00e39acb9a944356b346531ae9/shard_05/raw_dataset/AID.zip
```

```bash
sha256sum /home/featurize/work/AID_visionflan_20260920.zip
```

校验值一致后解压。包内目录是AID/AID/类别，根目录有两层AID。使用-n保留已存在文件，随后prepare检查文件内容。

```bash
unzip -n -q /home/featurize/work/AID_visionflan_20260920.zip -d /home/featurize/data
```

```bash
python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py prepare --dataset AID --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_protocol_20260920_01.json
```

完成需MANIFEST COMPLETE并核对train4000/val1000/test5000。若原清单已存在，不覆盖；先核对其来源。后续所有smoke的--data-root改为/home/featurize/data/AID/AID；模型、配置和划分规则不改。真实数据预检、预训练文件及GPU短测仍待回传。

## 未采用的来源

SkyScript作者的benchmark/aid.zip可下载，1549747311字节，但中央目录列出23886个图片条目，分在aid/train、aid/val、aid/test。没有进一步核对这些条目间的重复关系，不对其正确性作判断；本任务没有采用它或临时改当前数据协议。

## 429限流反馈（2026-09-20）

用户报告下载返回429，尚未提供发生错误的URL、响应头及正文，不能认定具体为匿名IP、账号或CDN限制。先暂停反复尝试并保留已有ZIP，取得失败最后10行再定位。

Hugging Face[官方限流文档](https://huggingface.co/docs/hub/rate-limits)说明Hub 429、匿名请求按IP计限及RateLimit中的剩余秒数；官方建议认证及按重置时间退避。本次不能仅凭状态码保证等待5分钟或登录必定解决。

对同revision的hf-mirror.com URL进行本地只读HEAD检查：HTTP308跳回原huggingface.co/resolve地址，再跳文件CDN，最终本地200。它不是本次已验证的独立备用下载通道，不能据此要求用户盲目换域名。没有向第三方镜像传递凭据，没有下载完整数据集或修改服务器环境。

### 用户回传代理路径后的下一步

日志显示`Connecting to 172.16.0.13:5848`及`Proxy request sent ... 429`。只能确认wget走了代理，不能确定429由代理生成还是上游返回。下一次仅添加wget的`--no-proxy`，重试限制为1，隔离代理路径这个变量；不全局unset代理、不关闭TLS校验、保留同文件续传。结果未回传前不能称修复成功。

```bash
wget --no-proxy -c --timeout=30 --tries=1 -O /home/featurize/work/AID_visionflan_20260920.zip https://huggingface.co/datasets/Vision-Flan/vision-flan/resolve/a7fc1f3307439d00e39acb9a944356b346531ae9/shard_05/raw_dataset/AID.zip
```

## 直连失败后：备用镜像（2026-09-20）

用户回传：原HF直连解析到108.160.172.208和2a03:2880:f10c:283:face:b00c:0:25de，IPv4超时、IPv6不可达；与此前代理429是两条失败路径。仅从该日志不能确认具体DNS服务或注入方。权重ls片段显示约109M文件存在，尚未验证SHA及加载。

本地对hf-mirror.net同revision同文件做Range GET：直接HTTP206，Content-Range bytes 0-255/2628474154，没有观察到重定向。256字节与原HF文件逐字节相同；[证据](../results/evidence/aid_source_probe_20260920/mirror_net_probe.json)。仅证明小段文件可达，不代表完整SHA一致或服务器已下载成功。第三方镜像不需要账号凭据，下载后继续用原发布方SHA256核验。

```bash
wget --no-proxy -c --timeout=30 --tries=1 -O /home/featurize/work/AID_visionflan_20260920.zip https://hf-mirror.net/datasets/Vision-Flan/vision-flan/resolve/a7fc1f3307439d00e39acb9a944356b346531ae9/shard_05/raw_dataset/AID.zip
```

另用浏览器核对AI Studio 52025：页面有AID_dataset.zip（2504.01M，内部AID/类别），但提示数据集尚未迁移；点击下载要求百度登录，未登录或绕过认证，也没有取得直链。另一个hf-mirror.com入口的本地GET和HEAD都会308回HF；不能把两个域名混写。

### 服务器下载完成回传

2026-09-20用户日志显示.net直连HTTP200，100%及`2628474154/2628474154`完整保存，3m50s；Ctrl+C出现在保存提示之后。下载已完成，完整SHA256、解压及prepare仍待回传，不再继续换源。按前述sha256sum核对一致后，再执行unzip和prepare。
