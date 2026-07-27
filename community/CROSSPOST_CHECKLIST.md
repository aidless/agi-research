# Cross-Post Checklist - 中文平台 v3 (2026-07-27)

> 单源真相：跟踪所有中文社区帖子的发布状态。
> 每次跨贴都先更新这个文件，再去对应平台发。

## 当前批次：v3 (2026-07-27)

**内容更新点（v2 → v3）**：
1. ✅ ENWI Prediction 2 100-epoch 复验（1.9x 差，比 30-epoch 的 3.5x 略好但仍负）
2. ✅ Phase 1.5 5-seed sweep（DEC-0011，delta_avg=+21.5±67.1, p>0.05）
3. ✅ H1 区分 monitor-prediction 层 vs policy-action 层
4. ✅ 加入诚实的负结果区块（同行评议友好）

## 平台清单

| 平台 | 草稿 | 状态 | 发布链接 | 发布时间 |
|------|------|------|----------|----------|
| CSDN 博客 | `csdn_announcement_v3.md` | ✅ 草稿就绪 | ⏳ 待发布 | - |
| OSCHINA 项目 | `oschina_announcement_v3.md` | ✅ 草稿就绪 | ⏳ 待发布 | - |
| 知乎（已完成 v1）| - | ✅ 已发布 2026-07-26 | https://www.zhihu.com/pin/2064649194275714554 | 2026-07-26 |

## 跨贴策略

### 为什么三平台不同内容？

| 平台 | 受众 | 内容重点 |
|------|------|----------|
| 知乎 | 科技爱好者 / 学生 | 概念 + 个人故事，观点型 |
| CSDN | 开发者 / 工程师 | 代码 + 数字，技术型 |
| OSCHINA | 开源贡献者 | 项目 + 框架 + 协作，开源型 |

### 共用 attribution block

所有平台都使用：

```
Liu Zewen (2026). Archimedes: A Self-Improving AGI Substrate.
Independent 5-year research program, AGI-2026-001.
github.com/aidless/agi-research
```

## 发布步骤

### CSDN

1. 打开 https://editor.csdn.net/md/ （或 https://blog.csdn.net/）
2. 复制 `csdn_announcement_v3.md` 的 Markdown 内容
3. 标题：直接复制第一行（**独立 5 年 AGI 研究计划 v3...**）
4. 标签：从 `#标签` 区块复制（5-10 个）
5. 分类：选 人工智能 > AGI
6. 发布模式：公开博客
7. 发布后回这里记录链接

### OSCHINA

1. 打开 https://my.oschina.net/u/0/question/create 或项目创建页
2. 类型：项目资讯（不是问答）
3. 标题加 `[AGI]` 前缀（按 OSCHINA 格式规范）
4. 复制 `oschina_announcement_v3.md` 内容
5. 添加项目链接：https://github.com/aidless/agi-research
6. 发布后回这里记录链接

## 发布后动作（24h 内）

- [ ] 回复所有评论（CSDN / OSCHINA 评论区）
- [ ] 记录每个平台的阅读量、点赞数
- [ ] 截图保存原始帖子的 URL（防 404）
- [ ] 更新 PROGRESS.md 中的 pending 列表
- [ ] 在 .experience_log/ 记录发布心得（什么内容效果好）

## IP 保护 checklist

每次发布前确认：

- [ ] ✅ Attribution block 完整（作者、项目、license）
- [ ] ✅ GitHub commit history 公开（时间戳证据）
- [ ] ✅ LICENSE 文件包含 (c) 2026 刘泽文
- [ ] ✅ 没有泄露私密数据（F:\TMLR\ 内部材料引用但不上传）
- [ ] ✅ 没有承诺未验证的结果（所有数字都在 PROGRESS.md 有据）

## 数据快照（v3 引用时的截止数据）

```
Total commits: 73
ENWI components ported: 4/4
H1 ablation seeds: 5/5 (frozen > joint, delta=0.724)
Slot-Monitor AUROC: 0.989 (vs raw 0.796, +0.193)
Slot WM next-step err: 0.000007
ENWI Prediction 2 (100 epoch): composable 1.9x 差
TTC BoN+Monitor (Phase 2.7): best gated -26.6 vs best ungated
```

## 历史版本

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1 | 2026-07-25 | 初始草稿（Twitter / Discord / Reddit / Email）|
| v2 | 2026-07-27 | CSDN + OSCHINA 草稿，含 ENWI port 结果（30-epoch smoke）|
| v3 | 2026-07-27 | CSDN + OSCHINA 更新，含 100-epoch 复验 + 5-seed DEC-0011 + 诚实负结果 |

---

*生成：2026-07-27，Codex + 刘泽文*
