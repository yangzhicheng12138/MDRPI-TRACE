"""Session-owned file artifacts and CPU-only Docker tools."""
from contextlib import nullcontext
import csv
import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile
from decimal import Decimal
from xml.etree import ElementTree
from uuid import uuid4

from .tasks import CURRENT_TURN, check_cancelled

IMAGE = "mdrpi-assistant-tools:20261003"
WORKER = Path(__file__).resolve().parents[1] / "tools/worker.py"


class WorkerFailure(ValueError):
    pass


def output_request_text(question):
    """Keep direct output instructions; omit quoted examples and prohibitions."""
    direct = re.sub(r'```.*?```|`[^`]*`|“[^”]*”|‘[^’]*’|"[^"\n]*"', "", question, flags=re.S)
    action = r"生成|创建|制作|导出|下载|写成|保存为|另存为|输出|提供|给我|做成|\b(?:create|generate|download|export|save(?:\s+as)?|provide)\b"
    clauses = []
    for clause in re.split(r"[，,。；;\n]", direct):
        accepted = False
        for match in re.finditer(action, clause, re.I):
            prefix = clause[:match.start()]
            if re.search(r"(?:不要|不用|不必|无需|别|禁止|不允许|不需要|请勿|勿).{0,12}$|"
                         r"\b(?:do not|don't|never|without)\b.{0,20}$|"
                         r"(?:解释|说明|介绍|翻译|讲解|如何|怎么|怎样|为什么|举例).{0,20}$|"
                         r"\b(?:explain|describe|translate|how to|why)\b.{0,30}$", prefix, re.I):
                continue
            clauses.append(clause)
            accepted = True
            break
        if not accepted and clauses and re.match(r"\s*(?:格式|文件名|命名|DOCX|Word|XLSX|Excel|CSV|TXT|Markdown|HTML|PNG|PDF|PPTX?)", clause, re.I):
            clauses[-1] += "，" + clause
    return "\n".join(clauses)


def check_output_size(output):
    total, entries = 0, 0
    for path in output.rglob("*"):
        entries += 1
        try:
            if path.is_file() and not path.is_symlink(): total += path.stat().st_size
        except FileNotFoundError:
            continue  # A running script may rename/delete its temporary output.
        if entries > 256 or total > 64 * 1024 * 1024:
            raise WorkerFailure("Generated output exceeds the file/size limit")


def requested_file_formats(question):
    """Find formats after an output verb; input filenames are not outputs."""
    formats = {"word": "docx", "docx": "docx", "excel": "xlsx", "xlsx": "xlsx", "csv": "csv",
               "txt": "txt", "文本文件": "txt", "markdown": "md", "md": "md", "html": "html",
               "png": "png", "图片": "png", "图表": "png", "柱状图": "png", "折线图": "png", "饼图": "png",
               "pdf": "pdf", "pptx": "pptx", "ppt": "pptx"}
    token = r"DOCX|Word|XLSX|Excel|CSV|TXT|Markdown|(?<![A-Za-z])MD(?![A-Za-z])|HTML|PNG|PDF|PPTX?|文本文件|图片|图表|柱状图|折线图|饼图"
    result = []
    for clause in output_request_text(question).splitlines():
        output = re.search(r"生成|创建|制作|导出|下载|写成|保存为|另存为|输出|提供|给我|做成|"
                           r"\b(?:create|generate|download|export|save(?:\s+as)?|provide)\b", clause, re.I)
        if not output:
            continue
        for match in re.finditer(token, clause[output.end():output.end() + 80], re.I):
            format_ = formats.get(match[0].lower(), "")
            if format_ and format_ not in result:
                result.append(format_)
        if ("png" not in result and re.search(r"(?:图片|图表|柱状图|折线图|饼图).{0,15}(?:下载|导出)|"
                                             r"\b(?:chart|plot|image)\b.{0,20}\b(?:download|export)\b", clause, re.I)):
            result.append("png")
    return result


def requested_file_format(question):
    formats = requested_file_formats(question)
    return formats[-1] if formats else ""


def requested_output_name(question):
    match = re.search(r"(?:文件名(?:为|是|叫|设为)|命名为|(?:file(?:name)?\s*(?:is|:|=)|named)\s*)"
                      r"\s*[《“\"']?([^\n，,。；;》”\"']{1,100})", question, re.I)
    return match[1].strip() if match else ""


def preserve_table_requested(question):
    return bool(re.search(r"(?:保留|保持|完整导出|全部导出|不删除|不要删除).{0,16}(?:原|所有|全部|行|列|数据|表格)|"
                          r"(?:原始|原有|原来的?).{0,10}(?:行|数据|表格).{0,10}(?:保留|完整导出|全部导出)|"
                          r"(?:retain|keep|preserve).{0,20}(?:original|all|rows|columns)", question, re.I))


def complete_explicit_file_plan(plan, state, question):
    """Compile unambiguous common tasks when the model omits their required tool.

    Uses actual metadata, never guessed file IDs/columns. Complex/ambiguous tasks
    still use the model and observation loop.
    """
    from .tool_runtime import from_calls
    if plan.get("planner") in {"literal_document", "explicit_title_revision"}:
        return plan
    # Keep non-output analysis/retrieval plans; they may not acquire write intent
    # from a request to explain or translate how a file is made.
    if not output_request_text(question) and re.search(r"生成|创建|导出|下载|\b(?:create|generate|export|download)\b", question, re.I):
        existing = plan.get("calls", [])
        safe = [c for c in existing if c.get("name") not in {"create_document", "export_table", "revise_document_title", "python"}]
        return {**plan, **from_calls({"calls": safe, "reason": plan.get("reason", "")})} if "calls" in plan else plan
    complex_task = re.search(r"相关|回归|预测|异常|标准差|方差|中位|标准化|排序|筛选|条件|自定义|\b(?:median|correlation|forecast|regression|normalize|filter|sort|custom)\b", question, re.I)
    if complex_task and set(plan.get("tools", [])) & {"analyze_table", "python"} and not requested_file_format(question):
        return plan
    if re.search(r"不要执行|不用执行|仅(?:解释|说明|写代码)|只(?:解释|说明|写代码)|翻译|虚构|\b(?:without executing|explain only|translate|fictional)\b", question, re.I):
        return plan
    files = [f for f in state.get("attachments", []) if f.get("active", True)]
    tables = [f for f in files if f.get("kind") == "table"]
    mentioned = [f for f in tables if f["name"] in question]
    table_file = mentioned[0] if len(mentioned) == 1 else tables[0] if len(tables) == 1 else None
    if table_file and len(table_file.get("sheets", [])) > 1:
        return plan  # First-sheet metadata cannot authorize columns/statistics of another sheet.
    calls = []
    format_ = requested_file_format(question)
    formats = requested_file_formats(question)
    preserve_rows = preserve_table_requested(question)
    output_name = requested_output_name(question)
    if table_file and format_ in {"docx", "xlsx", "csv"} and not complex_task and preserve_rows:
        total = bool(re.search(r"总计|合计|求和|总和|\b(?:sum|total)\b", question, re.I))
        numeric = table_file.get("numeric_columns", [])
        selected = [c for c in numeric if c in question] or numeric
        if not total or selected:
            calls = [{"name": "export_table", "args": {"file_id": table_file["file_id"], "sheet": "",
                      "format": target_format, "total_columns": selected if total else [], "name": output_name or "table_export"}}
                     for target_format in formats if target_format in {"docx", "xlsx", "csv"}]
    # An explicit output format takes priority over an aggregate-only fallback.
    # Preserve a valid model plan; never reinterpret a DOCX request as just CSV.
    if not calls and format_ in {"docx", "xlsx", "csv", "txt", "md", "html"}:
        existing = plan.get("calls", [])
        if any(c.get("name") == "create_document" and c.get("args", {}).get("format") == format_ for c in existing):
            revised = [{**c, "args": {**c["args"], "name": output_name}} if output_name and c.get("name") == "create_document" else c for c in existing]
            present = {c.get("args", {}).get("format") for c in revised if c.get("name") == "create_document"}
            revised.extend({"name": "create_document", "args": {"format": target_format, "name": output_name or "report",
                            "content": "", "instruction": question[:1200]}}
                           for target_format in formats if target_format not in present
                           and target_format in {"docx", "xlsx", "csv", "txt", "md", "html"})
            return {**plan, **from_calls({"calls": revised, "reason": plan.get("reason", "")})}
        if "python" in plan.get("tools", []) and plan.get("planner") != "failed":
            return plan  # Its actual output is checked by the completion contract.
        calls = [c for c in existing if c.get("name") not in {"create_document", "python", "export_table"}
                 and (complex_task or c.get("name") != "analyze_table")]
        if table_file and not any(c.get("name") == "files" for c in calls):
            calls.append({"name": "files", "args": {"query": question[:1000]}})
        calls.extend({"name": "create_document", "args": {"format": target_format, "name": output_name or "report",
                      "content": "", "instruction": question[:1200]}}
                     for target_format in formats if target_format in {"docx", "xlsx", "csv", "txt", "md", "html"})
    prior_document = (re.search(r"Word|DOCX|文档|报告|Excel", question, re.I) and
                      re.search(r"刚才|前面|上述|之前|上一步|\b(?:last|previous|above)\b", question, re.I) and
                      state.get("tool_results", {}).get("tool_observations"))
    if prior_document and not calls:
        calls = [{"name":"create_document", "args":{"format":"xlsx" if re.search(r"Excel|XLSX",question,re.I) else "docx",
                 "name":"report", "content":"", "instruction":question}}]
    if not calls and table_file and re.search(r"统计|合计|总计|求和|平均|均值|汇总|柱状图|折线图|饼图|(?:分析|绘制|画).{0,12}(?:表|图|数据)|\b(?:analy[sz]e|sum|mean|average|chart|plot|statistics|group)\b", question, re.I):
        numeric = table_file.get("numeric_columns", [])
        selected = [c for c in numeric if c in question] or numeric
        grouped = [c for c in table_file.get("columns", [])
                   if re.search(r"(?:按|根据|\bby\s+)\s*" + re.escape(c), question, re.I)]
        if len(grouped) <= 1 and selected:
            group = grouped[0] if grouped else ""
            chart = "bar" if re.search(r"柱状|\bbar\b|(?:画|绘制).{0,8}图|\bchart\b", question, re.I) else ""
            if re.search(r"折线|\bline\b", question, re.I): chart = "line"
            if re.search(r"饼图|\bpie\b", question, re.I): chart = "pie"
            operations = []
            if re.search(r"合计|总计|总和|求和|\b(?:sum|total)\b", question, re.I): operations.append("sum")
            if re.search(r"平均|均值|\b(?:mean|average)\b", question, re.I): operations.append("mean")
            if group: operations.append("group_mean" if operations == ["mean"] else "group_sum")
            if not operations: operations = ["describe"]
            for operation in operations[:3]:
                calls.append({"name": "analyze_table", "args": {"file_id": table_file["file_id"], "sheet": "",
                    "columns": selected, "operation": operation, "group_by": group if operation.startswith("group_") else "",
                    "chart": chart if operation.startswith("group_") or (not group and operation != "describe") else "", "name": "analysis_" + operation}})
    if not calls and re.search(r"(?:生成|创建|制作|导出|下载|写成|保存).{0,30}(?:Word|DOCX|文档|报告|Excel)|\b(?:create|generate|download|export|save)\b.{0,40}\b(?:docx|word|document|report|excel)\b", question, re.I):
        format_ = "xlsx" if re.search(r"Excel|XLSX", question, re.I) else "docx"
        calls = [{"name": "create_document", "args": {"format": format_, "name": "report", "content": "", "instruction": question}}]
    if not calls and files and re.search(r"(?:根据|读取|概括|总结|文件上|扫描).{0,30}(?:文件|上传|扫描|内容|销售额|平均数)|\b(?:uploaded file|scan|document says|read the file)\b", question, re.I):
        calls = [{"name": "files", "args": {"query": question}}]
    if not calls:
        return plan
    if calls[0]["name"] == "analyze_table":
        if not complex_task and all(c["args"]["operation"] != "describe" for c in calls):
            calls.extend(c for c in plan.get("calls", []) if c["name"] not in {"files", "analyze_table", "python"})
        else:
            if set(plan.get("tools", [])) & {"analyze_table", "python"}: return plan
            calls.extend(c for c in plan.get("calls", []) if c["name"] != "files")
    elif plan.get("tools") and not prior_document and not format_:
        return plan
    repaired = from_calls({"calls": calls, "reason": "An explicit file task requires actual registered file tools."})
    return {**plan, **repaired, "routing_override": "explicit_file_task"}


def render_artifact_links(answer, artifacts):
    """Resolve model-written download links against real session file manifests."""
    known = {a["artifact_id"]: a for a in artifacts}
    def link(match):
        label, url = match.group(1), match.group(2)
        identity = re.search(r"/artifacts/([a-f0-9]{32})", url)
        if identity and identity[1] in known:
            artifact = known[identity[1]]
            return f"[{artifact.get('name') or label}]({artifact['url']})"
        if re.search(r"sandbox:|file:|/mnt/data/|(?:/api/sessions/|https?://api\.sessions/).*/artifacts/", url):
            return label
        return match.group(0)
    parts = re.split(r"(```[\s\S]*?```|`[^`\n]*`)", answer)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", link, parts[i])
        for artifact_id, artifact in known.items():
            url = artifact["url"]
            parts[i] = re.sub(r"https?://[^\s)]+/artifacts/" + artifact_id + r"(?:\?[^\s)]*)?", url, parts[i])
    return "".join(parts)


def aliases(attachments):
    result, used = {}, set()
    for file in attachments:
        if not file.get("active", True):
            continue
        name = Path(file["name"].replace("\\", "/")).name.replace("\n", "_")[:180]
        if name in used:
            name = file["file_id"][:8] + "_" + name
        used.add(name); result[file["file_id"]] = name
    return result


def attached_inputs(root, state, generated=()):
    names = aliases(state.get("attachments", []))
    inputs = {}
    for file in state.get("attachments", []):
        if file["file_id"] in names:
            path = Path(root) / "files" / state["session_id"] / file["file_id"] / ("original" + Path(file["name"]).suffix.lower())
            inputs[names[file["file_id"]]] = path
    for artifact in [*state.get("generated_files", []), *generated]:
        inputs["generated_" + artifact["artifact_id"][:8] + "_" + artifact["name"]] = artifact_path(root, state["session_id"], artifact["artifact_id"])
    return inputs


def run_worker(task, args, inputs, output, *, timeout=60):
    """Execute a single worker in its own container, including per-turn cancellation."""
    check_cancelled()
    output = Path(output).resolve();output.mkdir(parents=True, exist_ok=True)
    container = "mdrpi-tool-" + uuid4().hex
    with tempfile.TemporaryDirectory(prefix="mdrpi-tool-input-") as folder, tempfile.TemporaryFile() as stdout_sink, tempfile.TemporaryFile() as stderr_sink:
        for name, source in inputs.items():
            if not source.is_file():
                raise WorkerFailure("A selected input file is unavailable")
            shutil.copyfile(source, Path(folder) / Path(name).name)
        cmd = ["docker", "run", "--rm", "-i", "--name", container, "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "128",
               "--cpus", "2", "--memory", "1g", "--memory-swap", "1g", "--user", f"{os.getuid()}:{os.getgid()}",
               "--tmpfs", "/tmp:rw,size=128m", "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp/matplotlib",
               "-e", "CUDA_VISIBLE_DEVICES=", "-v", f"{folder}:/input:ro", "-v", f"{output}:/output:rw",
               "-v", f"{WORKER}:/app/worker.py:ro", IMAGE]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stdout_sink, stderr=stderr_sink, text=True)
        def stop():
            try:
                subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=8)
            finally:
                if process.poll() is None: process.terminate()
        control = CURRENT_TURN.get()
        started = time.monotonic()
        try:
            with control.interrupt_with(stop) if control else nullcontext():
                process.stdin.write(json.dumps({"task": task, **args}, ensure_ascii=False));process.stdin.close();process.stdin = None
                while True:
                    check_cancelled()
                    if time.monotonic() - started > timeout:
                        raise WorkerFailure(f"Tool execution exceeded {timeout} seconds")
                    if os.fstat(stdout_sink.fileno()).st_size > 8 * 1024 * 1024 or os.fstat(stderr_sink.fileno()).st_size > 2 * 1024 * 1024:
                        raise WorkerFailure("Tool output is too large")
                    check_output_size(output)
                    try:
                        process.wait(timeout=.3)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            check_cancelled()
            check_output_size(output)
            stdout_sink.seek(0); stderr_sink.seek(0)
            stdout = stdout_sink.read(8 * 1024 * 1024 + 1).decode("utf-8", errors="replace")
            stderr = stderr_sink.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
            if process.returncode:
                raise WorkerFailure("Tool container failed: " + stderr[-1500:])
            if len(stdout) > 8 * 1024 * 1024:
                raise WorkerFailure("Tool output is too large")
            try:
                result = json.loads(stdout)
            except ValueError as exc:
                raise WorkerFailure("Tool returned no valid result: " + stdout[-500:]) from exc
            if not result.get("ok"):
                raise WorkerFailure(result.get("error", "Tool execution failed"))
            data = result["data"]
            if task == "python" and not data.get("ok"):
                raise WorkerFailure(data.get("stdout", "")[-1500:] + "\n" + data.get("error", "Python execution failed"))
            return data
        finally:
            if process.poll() is None:
                stop();process.wait(timeout=10)
            subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=8)


def _matrix_sha256(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _xml_text(element):
    parts = []
    for node in element.iter():
        name = node.tag.rsplit("}", 1)[-1]
        if name == "t":
            parts.append(node.text or "")
        elif name in {"br", "cr"}:
            parts.append("\n")
        elif name == "tab":
            parts.append("\t")
    return "".join(parts)


def inspect_artifact(path):
    """Read actual output bytes; no model completion claim is trusted here."""
    path = Path(path)
    format_ = path.suffix.lower().lstrip(".")
    observed = {"schema_version": "file_validation_v1", "valid": False, "format": format_,
                "format_verified": False,
                "measurement": "actual saved file integrity and measured content; not all natural-language requirements"}
    try:
        if not path.stat().st_size:
            raise ValueError("文件为空")
        if format_ in {"xlsx", "docx", "pptx"}:
            with zipfile.ZipFile(path) as archive:
                if sum(i.file_size for i in archive.infolist()) > 64 * 1024 * 1024:
                    raise ValueError("文档解压后超过校验限制")
                for package in ("[Content_Types].xml", "_rels/.rels"):
                    ElementTree.fromstring(archive.read(package))
                if archive.testzip() is not None:
                    raise ValueError("文档压缩包校验失败")
                if format_ == "docx":
                    root = ElementTree.fromstring(archive.read("word/document.xml"))
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    body = root.find("w:body", ns)
                    if body is None:
                        raise ValueError("DOCX 缺少正文")
                    paragraphs = [_xml_text(p) for p in body.findall("w:p", ns)]
                    tables = body.findall("w:tbl", ns)
                    observed.update(paragraph_count=len(paragraphs), table_count=len(tables),
                                    characters=sum(len(t) for t in paragraphs), content_preview="\n".join(paragraphs)[:900])
                    observed["paragraphs_sha256"] = _matrix_sha256(paragraphs)
                    if len(tables) == 1:
                        matrix = [["\n".join(_xml_text(p) for p in cell.findall("w:p", ns))
                                   for cell in row.findall("w:tc", ns)] for row in tables[0].findall("w:tr", ns)]
                        observed.update(matrix_sha256=_matrix_sha256(matrix), rows=max(0, len(matrix) - 1),
                                        columns=len(matrix[0]) if matrix else 0)
                    if not any(paragraphs) and not tables:
                        raise ValueError("DOCX 没有正文或表格")
                elif format_ == "pptx":
                    ElementTree.fromstring(archive.read("ppt/presentation.xml"))
                    slides = [name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
                    if not slides:
                        raise ValueError("PPTX 没有幻灯片")
                    texts = [_xml_text(ElementTree.fromstring(archive.read(name))) for name in slides]
                    observed.update(slide_count=len(slides), characters=sum(len(text) for text in texts),
                                    content_preview="\n".join(texts)[:900])
                else:
                    if "xl/workbook.xml" not in archive.namelist():
                        raise ValueError("XLSX 缺少工作簿")
                    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    shared = []
                    if "xl/sharedStrings.xml" in archive.namelist():
                        shared = [_xml_text(node) for node in ElementTree.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", ns)]
                    sheets = [name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)]
                    if not sheets:
                        raise ValueError("XLSX 没有工作表")
                    observed["sheet_count"] = len(sheets)
                    for name in sheets:
                        matrix = []
                        root = ElementTree.fromstring(archive.read(name))
                        for row in root.findall("s:sheetData/s:row", ns):
                            values = []
                            for cell in row.findall("s:c", ns):
                                reference = re.match(r"([A-Z]+)", cell.get("r", ""))
                                if not reference:
                                    raise ValueError("XLSX 单元格位置无效")
                                column = 0
                                for letter in reference[1]:
                                    column = column * 26 + ord(letter) - 64
                                if column > 1000:
                                    raise ValueError("XLSX 列数超过校验限制")
                                values.extend([""] * max(0, column - len(values)))
                                value = cell.find("s:v", ns)
                                raw = value.text or "" if value is not None else ""
                                kind = cell.get("t")
                                text = (shared[int(raw)] if kind == "s" else _xml_text(cell) if kind == "inlineStr" else
                                        "True" if raw == "1" else "False") if kind in {"s", "inlineStr", "b"} else raw
                                if raw and kind not in {"s", "inlineStr", "b", "str", "e"}:
                                    text = format(Decimal(raw).normalize(), "f")
                                values[column - 1] = text
                            matrix.append(values)
                        width = max((len(row) for row in matrix), default=0)
                        matrix = [row + [""] * (width - len(row)) for row in matrix]
                        if len(sheets) == 1:
                            observed.update(matrix_sha256=_matrix_sha256(matrix), rows=max(0, len(matrix) - 1), columns=width,
                                            content_preview=matrix[:13])
                    if not matrix:
                        raise ValueError("XLSX 没有单元格数据")
        elif format_ in {"csv", "txt", "md", "html"}:
            raw_bytes = path.read_bytes()
            text = raw_bytes.decode("utf-8-sig")
            if "\x00" in text or not text.strip():
                raise ValueError("文件不是有效文本")
            observed.update(characters=len(text), content_preview=text[:900])
            observed["content_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
            if format_ == "csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    matrix = list(csv.reader(handle))
                observed.update(matrix_sha256=_matrix_sha256(matrix), rows=max(0, len(matrix) - 1),
                                columns=len(matrix[0]) if matrix else 0)
        elif format_ in {"png", "jpg", "jpeg", "webp"}:
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                actual = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}.get(image.format, "")
                if actual != ("jpeg" if format_ == "jpg" else format_):
                    raise ValueError("图片实际格式与扩展名不一致")
                observed.update(width=image.width, height=image.height)
        elif format_ == "pdf":
            import fitz
            with fitz.open(path) as document:
                if document.is_encrypted or document.page_count < 1 or document.page_count > 1000:
                    raise ValueError("PDF 加密、为空或超过页数校验限制")
                texts = [page.get_text() for page in document]
                observed.update(page_count=document.page_count, characters=sum(len(text) for text in texts),
                                content_preview="\n".join(texts)[:900])
        else:
            raise ValueError("当前不支持独立校验此输出格式，无法确认文件任务已完成")
        observed["format_verified"] = True
        observed["valid"] = True
    except (OSError, ValueError, KeyError, IndexError, ImportError, RuntimeError, csv.Error, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        observed["error"] = str(exc)[:240]
    return observed


def collect_artifacts(root, session_id, output, *, task="", expected=None):
    root, output = Path(root), Path(output).resolve()
    result, total = [], 0
    for path in sorted(output.rglob("*")):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(output):
            continue
        size = path.stat().st_size
        if len(result) >= 20 or size > 20 * 1024 * 1024 or total + size > 60 * 1024 * 1024:
            continue
        artifact_id = uuid4().hex
        storage_name = "content" + path.suffix.lower()
        target = root / "artifacts" / session_id / artifact_id / storage_name
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(path, target)
        artifact = {"artifact_id": artifact_id, "name": path.name, "size": size,
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "url": f"/api/sessions/{session_id}/artifacts/{artifact_id}",
            "input_path": "/input/generated_" + artifact_id[:8] + "_" + path.name}
        artifact["validation"] = inspect_artifact(target)
        if task == "create_document":
            checks = {"expected_content_sha256": "content_sha256", "expected_paragraphs_sha256": "paragraphs_sha256",
                      "expected_matrix_sha256": "matrix_sha256"}
            measured = [(expected or {}).get(key) == artifact["validation"].get(field)
                        for key, field in checks.items() if (expected or {}).get(key)]
            verified = (bool(measured) and all(measured)
                        and artifact["validation"]["format"] == (expected or {}).get("format"))
            artifact["validation"]["draft_content_preserved"] = verified
            if not verified:
                artifact["validation"].update(valid=False, error="实际文件正文与提交的完整草稿不一致")
        if task == "export_table":
            verified = (artifact["validation"].get("matrix_sha256") == (expected or {}).get("expected_matrix_sha256")
                        and artifact["validation"]["format"] == (expected or {}).get("format"))
            artifact["validation"].update(preserved_original_values=verified,
                                          preservation_boundary=(expected or {}).get("preservation_boundary"),
                                          original_rows=(expected or {}).get("rows_loaded"))
            if not verified:
                artifact["validation"].update(valid=False, error="实际导出内容与完整表格/总计矩阵不一致")
        pending = target.parent / "metadata.tmp"
        pending.write_text(json.dumps({**artifact, "storage_name": storage_name}, ensure_ascii=False))
        pending.replace(target.parent / "metadata.json")
        result.append(artifact)
        total += size
    return result


def artifact_path(root, session_id, artifact_id):
    import re
    if not re.fullmatch(r"[a-f0-9]{32}", artifact_id):
        raise KeyError("Generated file not found")
    folder = Path(root) / "artifacts" / session_id / artifact_id
    try:
        record = json.loads((folder / "metadata.json").read_text())
        path = folder / record["storage_name"]
        if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            raise KeyError("Generated file not found")
        return path
    except (OSError, ValueError, KeyError) as exc:
        raise KeyError("Generated file not found") from exc


def list_artifacts(root, session_id):
    records = []
    for path in sorted((Path(root) / "artifacts" / session_id).glob("*/metadata.json"), key=lambda p:p.stat().st_mtime):
        try:
            record = json.loads(path.read_text())
            artifact_path(root, session_id, record["artifact_id"])
        except (OSError, ValueError, KeyError):
            continue
        record.pop("storage_name", None)
        records.append(record)
    return records


def invoke(ctx, task, args, *, timeout=60):
    from .tool_runtime import Result
    if ctx.root is None:
        raise WorkerFailure("No workspace is attached to this conversation")
    if ctx.deadline is not None:
        timeout = min(timeout, max(1, int(ctx.deadline - time.monotonic())))
    with tempfile.TemporaryDirectory(prefix="mdrpi-tool-output-") as output:
        data = run_worker(task, args, attached_inputs(ctx.root, ctx.state, ctx.runtime.get("artifacts", [])), output, timeout=timeout)
        artifacts = collect_artifacts(ctx.root, ctx.state["session_id"], output, task=task, expected=data)
    invalid = [a for a in artifacts if not a["validation"]["valid"]]
    status = "invalid_output" if invalid else "available"
    return Result(task, args, status=status, data={"workspace_result": {"task": task, **data}}, artifacts=artifacts,
                  events=[{"provider": task, "status": status, "artifacts": len(artifacts),
                           "validation_errors": [a["validation"].get("error", "校验未通过") for a in invalid]}])


def table(ctx, args):
    names = aliases(ctx.state.get("attachments", []))
    if args["file_id"] not in names:
        raise WorkerFailure("Select an active uploaded table by its file_id")
    if Path(names[args["file_id"]]).suffix.lower() not in {".csv", ".xlsx"}:
        raise WorkerFailure("Table analysis accepts CSV and XLSX files")
    ctx.emit("status", "正在分析表格并生成结果文件…")
    return invoke(ctx, "analyze_table", {**args, "file": names[args["file_id"]]})


def export_table(ctx, args):
    names = aliases(ctx.state.get("attachments", []))
    if args["file_id"] not in names or Path(names[args["file_id"]]).suffix.lower() not in {".csv", ".xlsx"}:
        raise WorkerFailure("请明确选择一个可用的 CSV 或 XLSX 表格。")
    ctx.emit("status", "正在保留原行列并生成完整表格文件…")
    return invoke(ctx, "export_table", {**args, "file": names[args["file_id"]]})


def python(ctx, args):
    ctx.emit("status", "正在执行数据处理代码…")
    code = "\n".join(args["code_lines"])
    if len(code) > 20000:
        raise WorkerFailure("Python code exceeds 20000 characters")
    compile(code, "generated.py", "exec")
    result = invoke(ctx, "python", {"code": code})
    if not result.artifacts and not result.data["workspace_result"].get("stdout", "").strip():
        result.status = "empty_output"
        result.data["workspace_result"]["warning"] = "Execution returned no printed result or output file. This alone does not fulfill an analysis/download request. Check code and print/write the requested result."
    return result


def document(ctx, args):
    content = args["content"]
    if args["instruction"].strip():
        ctx.emit("status", "正在编写文档内容…")
        remaining = min(120, max(1, ctx.deadline - time.monotonic())) if ctx.deadline else 120
        content = ctx.backend.draft_document(ctx.state, args["instruction"], ctx.runtime,
                                              ctx.log, ctx.emit, remaining_seconds=remaining, output_format=args["format"])
    elif not content.strip():
        raise WorkerFailure("Provide literal document content or a writing instruction")
    if args["instruction"].strip():
        # A chat answer describing manual creation is not a document body.
        # Reject it rather than saving a file and claiming the request is met.
        if re.search(r"(?:无法|不能|没有能力|不支持).{0,20}(?:生成|创建|提供|导出|下载).{0,20}(?:文件|文档|Word|Excel)|"
                     r"(?:请|你可以).{0,20}(?:复制|粘贴|保存为|另存为).{0,40}(?:Word|Excel|\.docx|\.xlsx)|"
                     r"(?:I (?:cannot|can't|am unable to)|unable to).{0,40}(?:create|generate|attach|download).{0,30}(?:file|document)|"
                     r"sandbox:/|/mnt/data/", content, re.I):
            raise WorkerFailure("文档草稿包含聊天操作说明或未生成声明，未将其作为完成的文档保存；请重试或提供正文。")
        if args["format"] in {"xlsx", "csv"} and (content.lstrip().startswith("```") or re.search(r"(?m)^\s*\|.+\|\s*$", content)):
            raise WorkerFailure("表格草稿不是原始 CSV 行，未生成可确认的表格文件。")
    ctx.emit("status", "正在生成可下载文档…")
    return invoke(ctx, "create_document", {**args, "content": content})


def revise_document_title(ctx, args):
    from .artifacts import selected_artifact
    selected = selected_artifact(ctx.root, ctx.state["session_id"], args["artifact_id"])
    if args["artifact_id"] != (ctx.state.get("revision_target") or {}).get("artifact_id"):
        raise WorkerFailure("请先明确选择要修改的生成文件。")
    if not selected["name"].lower().endswith(".docx") or not args["new_title"].strip():
        raise WorkerFailure("标题修改需要Word文件和非空的新标题。")
    ctx.emit("status", "正在修改文档标题并保留其他内容…")
    name = "generated_" + selected["artifact_id"][:8] + "_" + selected["name"]
    return invoke(ctx, "revise_document_title", {"file": name, "new_title": args["new_title"],
        "name": Path(selected["name"]).stem + "_v" + str(selected.get("version", 1) + 1)})


def extra_tools():
    from .tool_runtime import Tool, parameters, text
    return [
        Tool("revise_document_title", "Change only the title of an explicitly selected generated DOCX; preserve its other paragraphs, tables and formatting. Requires the selected revision_target artifact_id.",
             parameters(artifact_id=text(32), new_title=text(80)), "always", revise_document_title),
        Tool("analyze_table", "Compute statistics of an uploaded CSV/XLSX. Choose file_id/column names from files metadata. Generates CSV/XLSX and optional chart.",
            parameters(file_id=text(80), sheet=text(100), columns={"type": "array", "items": text(100), "maxItems": 20},
                operation={"type": "string", "enum": ["describe", "sum", "mean", "min", "max", "count", "group_sum", "group_mean", "group_count"]},
                group_by=text(100), chart={"type": "string", "enum": ["", "bar", "line", "pie"]}, name=text(100)), "files", table),
        Tool("export_table", "Export an entire CSV/XLSX's cell values and row/column order to DOCX/XLSX/CSV, optionally appending totals. It preserves original rows; not workbook formulas/styles. Refuses truncated input. Choose actual file_id and columns.",
             parameters(file_id=text(80), sheet=text(100), format={"type": "string", "enum": ["docx", "xlsx", "csv"]},
                        total_columns={"type": "array", "items": text(100), "maxItems": 20}, name=text(100)), "files", export_table),
        Tool("python", "Run Python in an isolated CPU container. Supply code_lines, ONE source line per array element with indentation preserved. pandas/numpy/openpyxl/matplotlib/Pillow are installed. Read supplied /input paths; write downloadable files under /output; print results. No network.",
             parameters(code_lines={"type": "array", "items": text(2000), "minItems": 1, "maxItems": 160}), "always", python),
        Tool("create_document", "Create a downloadable DOCX/XLSX/CSV/TXT/MD/HTML. For writing, supply instruction: it drafts the full document from actual results. For exact literal content, leave instruction empty. XLSX literal content is CSV rows.",
             parameters(format={"type": "string", "enum": ["docx", "xlsx", "csv", "txt", "md", "html"]},
                        name=text(100), content=text(16000), instruction=text(1200)), "always", document),
    ]
