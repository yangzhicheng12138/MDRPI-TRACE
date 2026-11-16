"""Session documents: open-source parsers and CPU BM25, separate from the trusted KB."""
from __future__ import annotations

import io
import json
import re
import uuid
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.table import Table
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

MAX_BYTES = 20 * 1024 * 1024
EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".xlsx", ".png", ".jpg", ".jpeg", ".webp"}


def parse_file(name, data):
    chunks, warnings, _ = parse_document(name, data)
    return chunks, warnings


def worker_parse(task, name, data):
    from .workspace import run_worker
    with tempfile.TemporaryDirectory(prefix="mdrpi-parse-") as folder:
        path = Path(folder) / Path(name).name
        path.write_bytes(data)
        return run_worker(task, {"file": path.name}, {path.name: path}, Path(folder) / "output", timeout=120)


def teaching_upload_texts(name, data):
    """Complete bounded table scan, or each currently extractable text chunk.

    Returned text is only for immediate privacy inspection; it must never be
    written to metadata or diagnostic logs. Opaque images remain outside this
    text check, as described by the document parser's warnings.
    """
    ext = Path(name).suffix.lower()
    if not data or len(data) > MAX_BYTES:
        raise ValueError("文件为空或超过 20 MB。")
    if ext in {".csv", ".xlsx"}:
        result = worker_parse("teaching_table_texts", name, data)
        if result.get("scan_complete") is not True:
            raise ValueError("教学表格完整隐私核查未完成，尚未保存；请拆分文件或另存为脱敏文本。")
        text = "\n".join(result["texts"])
        if len(text) > 1_000_000:
            raise ValueError("教学表格可核查文字超过 100 万字符，尚未保存；请拆分后上传。")
        return [text[start:start + 1600] for start in range(0, len(text), 1400)]
    chunks, warnings, _ = parse_document(name, data)
    if any("OCR 未完成" in warning for warning in warnings):
        raise ValueError("教学附件部分页面未能提取文字，完整可读文字核查未完成，尚未保存；请先转换为可读脱敏文本后上传。")
    return [chunk["text"] for chunk in chunks]


def parse_document(name, data):
    ext = Path(name).suffix.lower()
    if ext not in EXTENSIONS:
        raise ValueError("支持 PDF、DOCX、PPTX、TXT、Markdown、CSV、XLSX 和文字图片；旧版 .doc/.xls 请另存为新格式。")
    if not data or len(data) > MAX_BYTES:
        raise ValueError("文件为空或超过 20 MB。")
    warnings, pages, metadata = [], [], {"kind": "document"}
    try:
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("PDF 已加密，请先解密后上传。")
            if len(reader.pages) > 300:
                raise ValueError("单个 PDF 最多 300 页，请拆分文件。")
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                pages.append((f"第 {i} 页", text))
            empty = [i for i, (_, text) in enumerate(pages, 1) if not text.strip()]
            if empty:
                try:
                    parsed = worker_parse("ocr", name, data)
                    pages = [(p["location"], p["text"]) for p in parsed["pages"]]
                    warnings.extend(parsed["warnings"])
                    metadata["ocr_pages"] = [p for p in parsed["pages"] if p["method"] == "ocr"]
                except ValueError as exc:
                    warnings.append(f"{len(empty)} 页未提取到文字，OCR 未完成：{exc}")
                    if len(empty) == len(pages):
                        raise ValueError(warnings[-1]) from exc
        elif ext == ".docx":
            doc = Document(io.BytesIO(data))
            for i, block in enumerate(doc.iter_inner_content(), 1):
                text = "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows) if isinstance(block, Table) else block.text
                pages.append((f"段落/表格 {i}", text))
        elif ext == ".pptx":
            from lxml import etree
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                slides = sorted((p for p in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", p)),
                                key=lambda p: int(re.search(r"slide(\d+)", p)[1]))
                if len(slides) > 300 or sum(archive.getinfo(p).file_size for p in slides) > MAX_BYTES:
                    raise ValueError("PPTX 超过 300 页或文字结构过大，请拆分文件。")
                parser = etree.XMLParser(resolve_entities=False, no_network=True)
                for i, part in enumerate(slides, 1):
                    tree = etree.fromstring(archive.read(part), parser)
                    text = "\n".join(tree.xpath("//*[local-name()='t']/text()"))
                    pages.append((f"幻灯片 {i}", text))
            warnings.append("PPTX 读取文字和表格文本，嵌入图片未进行 OCR。")
        elif ext in {".csv", ".xlsx"}:
            parsed = worker_parse("table_info", name, data)
            metadata.update({"kind": "table", **{k: parsed[k] for k in ["columns", "rows_loaded", "truncated", "sheets", "numeric_columns"]}})
            metadata["preview"] = parsed["preview"][:3000]
            pages = [("表格预览（前 12 行；统计请用表格工具）", parsed["text"])]
            warnings.append(f"预览前 12 行，统计工具当前最多处理 50000 行；已读取 {parsed['rows_loaded']} 行。")
            if parsed["truncated"]:
                warnings.append("表格超过 50000 行，分析将只处理前 50000 行。")
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            parsed = worker_parse("ocr", name, data)
            pages = [(p["location"], p["text"]) for p in parsed["pages"]]
            warnings.extend(parsed["warnings"])
            metadata.update(kind="ocr", ocr_pages=parsed["pages"])
            warnings.append("图片通过 OCR 读取文字；不包含照片场景或物体识别。")
        else:
            encodings = ["utf-16"] if data.startswith((b"\xff\xfe", b"\xfe\xff")) else ["utf-8-sig", "gb18030"]
            for encoding in encodings:
                try:
                    text = data.decode(encoding)
                    break
                except UnicodeError:
                    continue
            else:
                raise ValueError("无法识别文本编码，请另存为 UTF-8。")
            if "\x00" in text:
                raise ValueError("文件不是可读取的文本。")
            pages = [("正文", text)]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"无法解析 {name}：{type(exc).__name__}。请检查文件是否完整。") from exc
    if sum(len(t) for _, t in pages) > 1_000_000:
        raise ValueError("文件文字量超过 100 万字符，请拆分文件。")
    chunks = []
    for location, text in pages:
        text = text.strip()
        # Preserve PDF page and DOCX table boundaries. Overlap protects sentence context.
        for start in range(0, len(text), 1400):
            chunks.append({"location": location, "text": text[start:start + 1600]})
    if not chunks:
        raise ValueError("没有提取到文字，请检查原文件或扫描清晰度。")
    return chunks, warnings, metadata


def store_file(root, name, data):
    name = Path(name.replace("\\", "/")).name
    chunks, warnings, metadata = parse_document(name, data)
    file_id = uuid.uuid4().hex
    folder = Path(root) / file_id
    folder.mkdir(parents=True)
    (folder / ("original" + Path(name).suffix.lower())).write_bytes(data)
    (folder / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (folder / "parse_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    public_metadata = {k: v for k, v in metadata.items() if k != "ocr_pages"}
    return {"file_id": file_id, "name": name, "size": len(data), "chunks": len(chunks),
            "characters": sum(len(c["text"]) for c in chunks), "warnings": warnings, "active": True, **public_metadata}


def tokens(text):
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    # Chinese bigrams supplement characters; no extra embedding model/GPU needed.
    words += [s[i:i + 2] for s in re.findall(r"[\u4e00-\u9fff]+", text) for i in range(len(s) - 1)]
    return words or ["_"]


def select_sources(root, attachments, question, limit=6):
    candidates = []
    for file in attachments:
        if not file.get("active", True):
            continue
        chunks = json.loads((Path(root) / file["file_id"] / "chunks.json").read_text())
        for i, chunk in enumerate(chunks):
            candidates.append({"doc_id": "upload_" + file["file_id"],
                "evidence_id": f"upload_{file['file_id']}_{i}", "file_id": file["file_id"],
                "source_type": "user_upload", "title": file["name"], **chunk})
    if not candidates:
        return [], []
    scores = BM25Okapi([tokens(c["title"] + " " + c["text"]) for c in candidates]).get_scores(tokens(question))
    ranked = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    # Small files are passed in full. For large files include a representative first
    # excerpt of every file, then fill by question relevance. This is NOT whole-file reading.
    selected, represented = [], set()
    for i in ranked:
        file_id = candidates[i]["file_id"]
        if file_id not in represented and len(selected) < limit:
            selected.append(i)
            represented.add(file_id)
    selected += [i for i in ranked if i not in selected][:max(0, limit - len(selected))]
    return [candidates[i] for i in selected], [{"provider": "upload", "status": "excerpt_selection",
        "selected_chunks": len(selected), "total_chunks": len(candidates),
        "partial": len(selected) < len(candidates)}]
