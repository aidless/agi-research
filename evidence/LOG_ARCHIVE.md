# experiments_log 归档说明（2026-10-04）

`experiments_log/` 原先混放两类内容：实验记录（markdown / JSON / 脚本 / 图）与运行时日志
（`.log` `.err` `.done`）。前者是可复现性材料，后者是 stdout/stderr 与空标记文件，
共 1518 个、1.92 MB，其中 741 个是零字节。

运行时日志已压缩为单个归档文件，工作树不再保留散文件：

- 归档：`evidence/experiments_log_20260801.tar.gz`
- SHA-256：`c51a677064ac4fce3e1436f364e4f92fccc43456060635e4e2ba3ff3a34a471d`
- 文件数：1518（非空 777 / 零字节 741）
- 压缩命令（确定性，可复算）：

```bash
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 2026-10-04' \
    -cf - -C <checkout>/experiments_log . | gzip -9 -n > experiments_log_20260801.tar.gz
sha256sum experiments_log_20260801.tar.gz
```

校验方式：

```bash
sha256sum -c experiments_log_20260801.tar.gz.sha256
tar -tzf evidence/experiments_log_20260801.tar.gz | head
```

## 保留了什么

`experiments_log/` 下的实验记录**未做任何删改**：

| 类型 | 数量 | 说明 |
|---|---|---|
| `.md` | 142 | 带日期的实验记录（试了什么、结果、为何中止） |
| `.json` | 26 | 结果与摘要 |
| `.py` / `.ps1` | 26 | 运行脚本 |
| `.png` / `.txt` / `.gitignore` | 6 | 图表、说明、目录忽略规则 |

也就是说：可复现性材料原样保留，只有机器产生的 stdout/stderr 被打包。
