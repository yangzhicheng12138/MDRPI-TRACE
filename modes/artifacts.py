"""Bounded previews of immutable files. The download always remains the original."""
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import zipfile

from .workspace import artifact_path, list_artifacts

TEXT_LIMIT = 30000
ROWS, COLUMNS = 100, 30
XML_LIMIT = 8 * 1024 * 1024


def selected_artifact(root, sid, artifact_id):
    item = next((row for row in list_artifacts(root, sid) if row["artifact_id"] == artifact_id), None)
    if item is None:
        raise KeyError("未找到本会话生成的文件。")
    return item


def xml_part(archive, part):
    from lxml import etree
    if archive.getinfo(part).file_size > XML_LIMIT:
        raise ValueError("文件文字结构超过预览容量，请下载原文件。")
    return etree.fromstring(archive.read(part), etree.XMLParser(resolve_entities=False, no_network=True))


def xlsx_sheets(raw):
    """Read raw values/formulas using the existing XML parser; no host Excel dependency."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(part.file_size for part in archive.infolist()) > 40 * 1024 * 1024:
            raise ValueError("表格结构超过预览容量，请下载原文件。")
        workbook = xml_part(archive, "xl/workbook.xml")
        links = xml_part(archive, "xl/_rels/workbook.xml.rels")
        targets = {r.get("Id"): r.get("Target") for r in links if r.get("TargetMode") != "External"}
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = xml_part(archive, "xl/sharedStrings.xml")
            shared = ["".join(si.xpath(".//*[local-name()='t']/text()")) for si in tree]
        descriptors = workbook.xpath("//*[local-name()='sheet']")
        sheets, truncated = [], len(descriptors) > 3
        for descriptor in descriptors[:3]:
            relation = descriptor.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = targets.get(relation, "")
            part = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "xl/" + target)
            if not part.startswith("xl/") or part not in archive.namelist():
                raise ValueError("工作表引用无效，请下载原文件核对。")
            tree = xml_part(archive, part)
            cells, width = {}, 0
            for row in tree.xpath("//*[local-name()='sheetData']/*[local-name()='row']"):
                number = int(row.get("r", "0"))
                if number < 1:
                    continue
                if number > ROWS + 1:
                    truncated = True
                    continue
                values = {}
                for cell in row:
                    coordinate = re.match(r"([A-Z]+)[0-9]+$", cell.get("r", ""))
                    if not coordinate:
                        continue
                    column = 0
                    for letter in coordinate[1]:
                        column = column * 26 + ord(letter) - 64
                    if column > COLUMNS:
                        truncated = True
                        continue
                    formula = cell.xpath("./*[local-name()='f']/text()")
                    value = "".join(cell.xpath("./*[local-name()='v']/text()"))
                    if formula:
                        value = "=" + formula[0]
                    elif cell.get("t") == "s":
                        idx = int(value) if value.isdigit() else -1
                        value = shared[idx] if 0 <= idx < len(shared) else ""
                    elif cell.get("t") == "inlineStr":
                        value = "".join(cell.xpath(".//*[local-name()='t']/text()"))
                    elif cell.get("t") == "b":
                        value = "TRUE" if value == "1" else "FALSE"
                    truncated |= len(value) > 1000
                    values[column - 1] = value[:1000]
                    width = max(width, column)
                cells[number - 1] = values
            rows = [[cells.get(i, {}).get(j, "") for j in range(width)] for i in range(max(cells, default=-1) + 1)]
            sheets.append({"name": descriptor.get("name", "工作表"), "rows": rows})
        return sheets, truncated


def preview(root, sid, artifact_id):
    item = selected_artifact(root, sid, artifact_id)
    path = artifact_path(root, sid, artifact_id)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("文件内容与保存记录不一致，请重新生成。")
    ext = path.suffix.lower()
    result = {"artifact": item, "kind": "unsupported", "truncated": False,
              "original_sha256": item["sha256"], "note": "下载为完整原文件。", "limits": {"characters": TEXT_LIMIT, "rows": ROWS, "columns": COLUMNS}}
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        result.update(kind="image", url=item["url"], note="预览和下载使用同一张原始图片。")
    elif ext == ".csv":
        # Bound decoding and rows independently; a huge quoted field cannot grow the response.
        data = raw[:512000].decode("utf-8-sig", errors="replace")
        reader = csv.reader(io.StringIO(data))
        csv.field_size_limit(512000)
        rows = []
        width_truncated = False
        for row in reader:
            width_truncated |= len(row) > COLUMNS or any(len(cell) > 1000 for cell in row[:COLUMNS])
            rows.append([cell[:1000] for cell in row[:COLUMNS]])
            if len(rows) >= ROWS + 2:
                break
        result.update(kind="table", sheets=[{"name": "CSV", "rows": rows[:ROWS + 1]}],
                      truncated=len(rows) > ROWS + 1 or len(raw) > 512000 or width_truncated,
                      note="最多预览100行、30列；下载为完整CSV。")
    elif ext == ".docx":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            tree = xml_part(archive, "word/document.xml")
            paragraphs = ["".join(p.xpath(".//*[local-name()='t']/text()")) for p in tree.xpath("//*[local-name()='p']")]
        text = "\n".join(paragraphs)
        result.update(kind="text", text=text[:TEXT_LIMIT], truncated=len(text) > TEXT_LIMIT,
                      note="文档文本预览；字体、表格和页面排版以原Word文件为准。")
    elif ext == ".xlsx":
        sheets, truncated = xlsx_sheets(raw)
        result.update(kind="table", sheets=sheets, truncated=truncated,
                      note="最多预览3个工作表、每表100行和30列；显示原始值与公式，不在预览中计算。日期、数值格式与排版以原Excel文件为准。")
    elif ext in {".md", ".txt", ".json", ".py", ".html", ".svg", ".css", ".js"}:
        text = raw[:TEXT_LIMIT * 4 + 4].decode("utf-8-sig", errors="replace")
        result.update(kind="markdown" if ext == ".md" else "text", text=text[:TEXT_LIMIT],
                      truncated=len(text) > TEXT_LIMIT or len(raw) > TEXT_LIMIT * 4,
                      note="文本预览；下载为完整原文件，代码不会在预览中执行。")
    return result


def revision_context(root, sid, artifact_id):
    result = preview(root, sid, artifact_id)
    return {"selected_file": result["artifact"], "kind": result["kind"],
            "text": result.get("text", ""), "sheets": result.get("sheets", []),
            "truncated": result["truncated"], "note": result["note"],
            "instruction": "Revise the explicitly selected original input_path. A truncated preview is not the whole file. Use actual file tools for unshown content. Create a new file and retain the old version."}


def tag_revision(root, sid, artifacts, parent, turn_id):
    """Only newly created manifests acquire provenance; never rewrite an old version."""
    result = []
    for artifact in artifacts:
        tagged = {**artifact, "turn_id": turn_id, "created_at": datetime.now(timezone.utc).isoformat()}
        if parent:
            tagged.update(revision_of=parent["artifact_id"], root_artifact_id=parent.get("root_artifact_id", parent["artifact_id"]),
                          version=parent.get("version", 1) + 1)
        else:
            tagged.update(root_artifact_id=artifact["artifact_id"], version=1)
        folder = artifact_path(root, sid, artifact["artifact_id"]).parent
        old = json.loads((folder / "metadata.json").read_text())
        pending = folder / "metadata.revision.tmp"
        pending.write_text(json.dumps({**old, **tagged}, ensure_ascii=False))
        pending.replace(folder / "metadata.json")
        result.append(tagged)
    return result
