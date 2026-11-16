# 数据与教学材料

## 随仓库提供

- `openrn.json`：30 项 Open RN 公开教材改编练习，逐题保留作者、原始链接和许可。
- `mdrpi_specialty.json`：8 项项目编写的 MDRPI 专项练习，包含等价答案、常见误解和分层提示。
- `modes/teaching.py`：3 个明确标识为虚构的练习情景。
- 评分点、内容复核和目录分类仅保留上述样例对应部分。AI 复核记录是内容开发记录，不是独立临床专家验证。

原始 Open RN 内容按 CC BY 4.0 使用；项目提供的教学改编按 CC BY-SA 4.0 使用。此许可针对教学材料，不覆盖外链文献的全文。外链 PDF、原文网页与模型权重不打包再分发。

## 独立准备的运行资料

正式研究病例、实验生成答案、研究评分、用户上传文件、会话与学习记录、模型权重和知识库原文均不在本仓库内。运行后形成的数据默认位于 `data/interactive_modes/`，已经被 `.gitignore` 排除。

## 建立自己的知识库

将合法持有的 PDF 放入 `data/source/PDFs/`，调整 `config.yaml`。索引建设保留来源核对和审核准入，不能把未经审核的分块直接当成已审核证据。

```bash
python scripts/01_scan_documents.py --help
python scripts/02_parse_documents.py --help
python scripts/03_build_chunks.py --help
python scripts/04_audit_chunks.py --help
python scripts/04_review_manifest.py --help
python scripts/05_build_vector_db.py --help
```

依次完成题录、解析、分块、内容检查、来源审核和索引建设。每个命令的 `--help` 说明输入输出；完成审核后再生成可检索的发布索引。不要将缺少授权的全文和索引提交到公共仓库。

## 开启命题材料

出题材料登记位置是 `data/question_generation/development_materials.json`，加载契约位于 `modes/question_generation_materials.py`。空仓库不包含该文件，因此工作台显示空列表。

登记材料需要提供独立开发病例、教学依据、来源引用、人群与器械适用范围，以及与审核内容绑定的文件和条目指纹。当前兼容格式依赖 `data/teaching_bank/expansion_250.json` 中的已审核开发内容和本地合法 PDF；文件名是既有契约名称，并不要求恰好 250 项。

代码核验材料登记、内容审核和原文指纹后才送入命题流程。材料缺失或不匹配时停止该次命题，避免以未经核验的资料代替。开发材料与正式研究病例应分别保管。

## 添加教学题目

参考公开样例的题干、答案、评分点、可接受表达、误解和提示结构。题干或答案变化后，应同步更新对应评分点及审核指纹；不要只修改题干而沿用旧审核结论。
