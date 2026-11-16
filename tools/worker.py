"""CPU tool worker. Inputs/output are container mounts; stdout is one JSON result."""
import contextlib
import csv
import io
import json
from pathlib import Path
import re
import sys
import traceback
import hashlib
from decimal import Decimal, InvalidOperation

INPUT, OUTPUT = Path("/input"), Path("/output")


def input_file(name):
    path = INPUT / Path(name).name
    if not path.is_file():
        raise ValueError("Input file is unavailable")
    return path


def output_file(name, suffix):
    name = Path(name.replace("\\", "/")).name
    stem = Path(name).stem[:100] or "result"
    return OUTPUT / (stem + suffix)


def load_table(path, sheet="", *, preserve_text=False):
    import pandas as pd
    if path.suffix.lower() == ".csv":
        for encoding in ["utf-8-sig", "gb18030"]:
            try:
                frame = pd.read_csv(path, encoding=encoding, nrows=50001,
                                    dtype=str if preserve_text else None, keep_default_na=not preserve_text)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("CSV encoding must be UTF-8 or GB18030")
    else:
        frame = pd.read_excel(path, sheet_name=sheet or 0, nrows=50001, engine="openpyxl",
                              dtype=object if preserve_text else None, keep_default_na=not preserve_text)
    truncated = len(frame) > 50000
    frame = frame.iloc[:50000].copy()
    frame.columns = [str(c) for c in frame.columns]
    if len(frame.columns) > 100:
        raise ValueError("A table may have at most 100 columns")
    if any(len(c) > 100 for c in frame.columns):
        raise ValueError("A column name may have at most 100 characters")
    return frame, truncated


def table_cell(value):
    """Stable value representation for independently checking the saved file."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float, Decimal)):
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("Non-finite table values cannot be exported")
        return format(number.normalize(), "f")
    return "" if value is None else str(value)


def load_preserved_table(path, sheet=""):
    """Read a value matrix without pandas changing duplicate/empty headers."""
    if path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                with path.open(encoding=encoding, newline="") as handle:
                    rows = []
                    for row in csv.reader(handle):
                        rows.append(row)
                        if len(rows) > 50001:
                            raise ValueError("表格超过 50000 行，不能确认保留全部原行；请拆分文件后导出。")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("CSV encoding must be UTF-8 or GB18030")
    else:
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=False)
        try:
            selected = workbook[sheet] if sheet else workbook.worksheets[0]
            if selected.max_row > 50001 or selected.max_column > 100:
                raise ValueError("完整导出最多支持 50000 行、100 列；请拆分表格。")
            rows = []
            for cells in selected.iter_rows():
                if any(cell.data_type == "f" for cell in cells):
                    raise ValueError("原表含公式，当前导出不能确认保留公式结果；请先另存为数值表格。")
                rows.append([cell.value for cell in cells])
        finally:
            workbook.close()
    if not rows or not rows[0] or not rows[1:]:
        raise ValueError("表格没有可导出的数据行。")
    width = len(rows[0])
    if width > 100 or any(len(str(value or "")) > 100 for value in rows[0]):
        raise ValueError("完整导出最多支持 100 列，列名最多 100 字。")
    if any(len(row) > width for row in rows[1:]):
        raise ValueError("CSV 数据行包含无列名的额外单元格，无法确认完整表结构；请先修正列名。")
    return [table_cell(value) for value in rows[0]], [row + [None] * (width - len(row)) for row in rows[1:]]


def export_table(request):
    """Keep all loaded values/columns; a footer augments rather than aggregates."""
    path = input_file(request["file"])
    columns, matrix = load_preserved_table(path, request.get("sheet", ""))
    cells = (len(matrix) + 2) * len(columns)
    if request["format"] == "docx" and cells > 5000 or request["format"] == "xlsx" and cells > 500000:
        raise ValueError("完整导出的单元格数量超过该格式的限制；请拆分表格或选择 CSV。")
    selected = request.get("total_columns", [])
    if any(c not in columns for c in selected):
        raise ValueError("Total column is absent from the source table")
    if any(columns.count(c) != 1 for c in selected):
        raise ValueError("所选合计列有重复名称，无法确认应汇总哪一列；请先使用唯一列名。")
    if any(columns.count(c) != 1 for c in selected):
        raise ValueError("求和列名重复，无法确定目标列；请先给求和列唯一名称。")
    original_rows = len(matrix)
    if selected:
        footer = ["" for _ in columns]
        for column in selected:
            values = [row[columns.index(column)] for row in matrix]
            try:
                total = sum((Decimal(str(v)) for v in values if v is not None and str(v).strip()), Decimal(0))
            except InvalidOperation as exc:
                raise ValueError("Total column is not wholly numeric: " + column) from exc
            if not total.is_finite():
                raise ValueError("Total column contains a non-finite value: " + column)
            footer[columns.index(column)] = table_cell(total)
        label = next((i for i, c in enumerate(columns) if c not in selected), None)
        if label is not None:
            footer[label] = "总计"
        matrix.append(footer)
    format_ = request["format"]
    rows = [[table_cell(v) for v in columns], *[[table_cell(v) for v in row] for row in matrix]]
    target = output_file(request.get("name", "table_export"), "." + format_)
    if format_ == "csv":
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerows(rows)
    elif format_ == "xlsx":
        from openpyxl import Workbook
        workbook = Workbook(); sheet = workbook.active
        for row in rows:
            sheet.append(row)
            # Original strings are values, including text that starts '='.
            # Do not turn an uploaded string into an executable Excel formula.
            for cell, value in zip(sheet[sheet.max_row], row):
                if isinstance(value, str):
                    cell.data_type = "s"
        workbook.save(target)
    elif format_ == "docx":
        from docx import Document
        document = Document()
        table = document.add_table(rows=1, cols=len(columns))
        for cell, value in zip(table.rows[0].cells, rows[0]):
            cell.text = value
        for row in rows[1:]:
            for cell, value in zip(table.add_row().cells, row):
                cell.text = value
        document.save(target)
    else:
        raise ValueError("A complete table export accepts CSV, XLSX or DOCX")
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"format": format_, "rows_loaded": original_rows, "result_rows": len(matrix), "columns": columns,
            "total_columns": selected, "truncated": False, "preserved_original_values": True,
            "preservation_boundary": "cell text values and row/column order of the selected sheet; not original formulas or styles",
            "expected_matrix_sha256": hashlib.sha256(encoded).hexdigest(), "preview": rows[:13],
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def table_info(request):
    import pandas as pd
    path = input_file(request["file"])
    frame, truncated = load_table(path, request.get("sheet", ""))
    sheets = pd.ExcelFile(path, engine="openpyxl").sheet_names if path.suffix.lower() == ".xlsx" else []
    preview = frame.head(12).to_csv(index=False)
    return {"columns": list(frame.columns), "rows_loaded": len(frame), "truncated": truncated,
            "numeric_columns": list(frame.select_dtypes(include="number").columns),
            "sheets": sheets, "preview": preview[:10000], "text": preview[:10000]}


def teaching_table_texts(request):
    """Ephemeral privacy text from every CSV row or workbook worksheet.

    This is independent of a teaching/search preview. A scan limit is a refusal,
    never permission to store the unchecked remainder of a teaching attachment.
    """
    path = input_file(request["file"])
    texts, characters, rows_seen = [], 0, 0

    def append(text):
        nonlocal characters
        characters += len(text) + 1
        if characters > 1_000_000:
            raise ValueError("教学表格可核查文字超过 100 万字符，未保存；请拆分或另存为脱敏文本后上传。")
        if text:
            texts.append(text)

    def scan_rows(rows, label, *, count_rows=True):
        nonlocal rows_seen
        append(label)
        headings = None
        identity_labels = {"患者姓名", "姓名", "patient name", "name", "住院号", "病历号", "门诊号", "就诊号",
                           "医疗记录号", "mrn", "家庭住址", "家庭地址", "详细地址", "住址", "address",
                           "床号", "床位号", "bed number", "bed no."}
        column_labels = identity_labels | {"性别", "年龄", "gender", "sex", "age"}
        for row in rows:
            if count_rows:
                rows_seen += 1
            if rows_seen > 50001 or len(row) > 100:
                raise ValueError("教学表格完整隐私核查最多支持合计 50001 行、每行 100 列，未保存；请拆分文件后上传。")
            values = [table_cell(value) for value in row]
            # A structural delimiter must not turn the headings 姓名,性别,年龄
            # into the invented clinical statement 姓名：性别：年龄.
            append(" | ".join(values))
            if headings is None:
                headings = [value.strip().rstrip(":：").strip() for value in values]
            else:
                # A name under a '姓名' column needs its header as context.
                append("\n".join(f"{heading}：{value}" for heading, value in zip(headings, values)
                                 if heading.strip() and value.strip()))
                # Label/value layouts under generic headings (项目,值) remain
                # inspectable without interpreting the first header row as data.
                append("\n".join(f"{value.strip().rstrip(':：').strip()}：{following}" for value, following in
                                 zip(values, values[1:])
                                 if value.strip().rstrip(":：").strip().casefold() in identity_labels
                                 and following.strip()
                                 and following.strip().rstrip(":：").strip().casefold() not in column_labels))

    if path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                # Decode before mutating counts, so an encoding retry never
                # treats a partly scanned first attempt as a complete scan.
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("CSV encoding must be UTF-8 or GB18030")
        if len(content) > 1_000_000:
            raise ValueError("教学表格可核查文字超过 100 万字符，未保存；请拆分或另存为脱敏文本后上传。")
        scan_rows(csv.reader(io.StringIO(content, newline="")), "CSV表格")
    else:
        from openpyxl import load_workbook
        # Formula text and any available saved result are both human-readable
        # workbook content. Scan all worksheets, including hidden worksheets.
        for data_only in (False, True):
            workbook = load_workbook(path, read_only=True, data_only=data_only)
            try:
                for sheet in workbook.worksheets:
                    if sheet.max_row > 50001 or sheet.max_column > 100:
                        raise ValueError("教学表格完整隐私核查超过行列限制，未保存；请拆分文件后上传。")
                    # Count physical rows once; both views still count toward
                    # the character bound, so neither view can be truncated.
                    scan_rows(sheet.iter_rows(values_only=True), "工作表：" + sheet.title,
                              count_rows=not data_only)
            finally:
                workbook.close()
    return {"texts": texts, "scan_complete": True, "characters": characters,
            "boundary": "all CSV cells or all worksheets' formula text and available saved cell values; no image OCR"}


def configure_plot():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    fonts = list(Path("/usr/share/fonts").rglob("*CJK*.ttc"))
    if fonts:
        font_manager.fontManager.addfont(str(fonts[0]))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False


def analyze_table(request):
    import pandas as pd
    path = input_file(request["file"])
    frame, truncated = load_table(path, request.get("sheet", ""))
    sheet = (request.get("sheet") or pd.ExcelFile(path, engine="openpyxl").sheet_names[0]) if path.suffix.lower() == ".xlsx" else ""
    columns = request.get("columns") or list(frame.select_dtypes(include="number").columns)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError("Columns not found: " + ", ".join(missing))
    operation, group = request["operation"], request.get("group_by", "")
    if not columns:
        raise ValueError("No numeric columns found; choose columns for a count/description or use Python")
    if operation in {"sum", "mean", "min", "max", "group_sum", "group_mean"}:
        nonnumeric = [c for c in columns if c != group and not pd.api.types.is_numeric_dtype(frame[c])]
        if nonnumeric:
            raise ValueError("These columns are not numeric: " + ", ".join(nonnumeric))
    if operation == "describe":
        result = frame[columns].describe().reset_index()
    elif operation == "count":
        result = frame[columns].count().rename("count").reset_index()
    elif operation in {"sum", "mean", "min", "max"}:
        result = getattr(frame[columns], operation)(numeric_only=True).rename(operation).reset_index()
    elif operation in {"group_sum", "group_mean"}:
        if group not in frame.columns:
            raise ValueError("Grouping column not found")
        numeric = [c for c in columns if c != group]
        result = getattr(frame.groupby(group, dropna=False)[numeric], operation.split("_")[1])(numeric_only=True).reset_index()
    elif operation == "group_count":
        if group not in frame.columns:
            raise ValueError("Grouping column not found")
        result = frame.groupby(group, dropna=False).size().rename("count").reset_index()
    else:
        raise ValueError("Unsupported table operation")
    if len(result) > 5000:
        raise ValueError("Result has more than 5000 rows; use a smaller grouping")
    csv_path = output_file(request.get("name", "analysis"), ".csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    xlsx_path = output_file(request.get("name", "analysis"), ".xlsx")
    result.to_excel(xlsx_path, index=False, engine="openpyxl")
    chart = request.get("chart", "")
    if chart:
        configure_plot()
        import matplotlib.pyplot as plt
        numeric = list(result.select_dtypes(include="number").columns)
        if not numeric:
            raise ValueError("The result has no numeric column to plot")
        labels = result.iloc[:, 0].astype(str)
        values = result[numeric[-1]]
        fig, ax = plt.subplots(figsize=(9, 5))
        if chart == "bar": ax.bar(labels, values)
        elif chart == "line": ax.plot(labels, values, marker="o")
        elif chart == "pie": ax.pie(values, labels=labels, autopct="%1.1f%%")
        else: raise ValueError("Unsupported chart type")
        ax.set_title(request.get("name") or "Analysis")
        if chart != "pie": ax.set_ylabel(numeric[-1]); ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout(); fig.savefig(output_file(request.get("name", "analysis"), ".png"), dpi=150); plt.close(fig)
    preview = json.loads(result.head(50).to_json(orient="records", force_ascii=False))
    while len(json.dumps(preview, ensure_ascii=False)) > 12000 and len(preview) > 1:
        preview = preview[:len(preview)//2]
    return {"rows_loaded": len(frame), "truncated": truncated, "columns": columns, "sheet": sheet,
            "operation": operation, "result": preview, "result_rows": len(result),
            "result_preview_partial": len(preview) < len(result)}


def create_document(request):
    content, format_ = request["content"], request["format"]
    expected = {}
    path = output_file(request.get("name", "document"), "." + format_)
    if format_ == "docx":
        from docx import Document
        document = Document()
        paragraphs = []
        for line in content.splitlines():
            heading = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading: document.add_heading(heading[2], level=len(heading[1]))
            else: document.add_paragraph(line)
            paragraphs.append(heading[2] if heading else line)
        document.save(path)
        expected["expected_paragraphs_sha256"] = hashlib.sha256(json.dumps(paragraphs, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    elif format_ == "xlsx":
        from openpyxl import Workbook
        workbook = Workbook();sheet = workbook.active
        rows = list(csv.reader(io.StringIO(content)))
        for row in rows:
            sheet.append(row)
            for cell in sheet[sheet.max_row]:
                cell.data_type = "s"
        workbook.save(path)
        width = max((len(row) for row in rows), default=0)
        matrix = [row + [""] * (width - len(row)) for row in rows]
        expected["expected_matrix_sha256"] = hashlib.sha256(json.dumps(matrix, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    elif format_ in {"txt", "md", "csv", "html"}:
        path.write_text(content, encoding="utf-8")
        expected["expected_content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    else:
        raise ValueError("Unsupported output format")
    return {"format": format_, "characters": len(content), "preview": content[:900], **expected}


def revise_document_title(request):
    from docx import Document
    document = Document(input_file(request["file"]))
    title = next((p for p in document.paragraphs if p.text.strip()), None)
    if title is None or not (title.style.name.startswith("Heading") or title.style.name == "Title"):
        raise ValueError("未找到明确的文档标题；请说明需要修改的正文位置。")
    previous = title.text
    title.text = request["new_title"]
    document.save(output_file(request["name"], ".docx"))
    return {"format": "docx", "old_title": previous, "new_title": title.text,
            "preserved_other_content": True}


def python_code(request):
    code = request["code"]
    captured = io.StringIO()
    scope = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile(code, "generated.py", "exec"), scope, scope)
    except Exception:
        return {"ok": False, "stdout": captured.getvalue()[-8000:], "error": traceback.format_exc()[-3000:]}
    return {"ok": True, "stdout": captured.getvalue()[-10000:]}


def ocr(request):
    import fitz
    from rapidocr_onnxruntime import RapidOCR
    from PIL import Image, ImageOps
    import numpy as np
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
    path = input_file(request["file"])
    pages, warnings = [], []
    def read_image(data, label):
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        if image.width * image.height > 20000000:
            raise ValueError("Image exceeds 20 megapixels")
        result, _ = engine(np.asarray(image))
        rows = [{"text": item[1], "confidence": round(float(item[2]), 4),
                 "box": np.asarray(item[0]).tolist()} for item in result or []]
        return {"location": label, "text": "\n".join(r["text"] for r in rows), "lines": rows, "method": "ocr"}
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as document:
            if document.page_count > 80:
                raise ValueError("OCR PDF accepts at most 80 pages; split larger files")
            for i, page in enumerate(document):
                text_ = page.get_text().strip()
                if text_:
                    pages.append({"location": f"第 {i+1} 页", "text": text_, "method": "native"})
                else:
                    pages.append(read_image(page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8)).tobytes("png"), f"第 {i+1} 页"))
    else:
        pages.append(read_image(path.read_bytes(), "图片 OCR"))
    if not any(p["text"].strip() for p in pages):
        raise ValueError("No text was recognized")
    for page in pages:
        if any(line["confidence"] < .8 for line in page.get("lines", [])):
            warnings.append(page["location"] + "含低置信度识别，请核对原图。")
    return {"pages": pages, "warnings": warnings, "text": "\n\n".join(p["text"] for p in pages)}


def main():
    request = json.load(sys.stdin)
    handlers = {"table_info": table_info, "teaching_table_texts": teaching_table_texts,
                "analyze_table": analyze_table, "export_table": export_table, "create_document": create_document,
                "revise_document_title": revise_document_title,
                "python": python_code, "ocr": ocr}
    try:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = handlers[request["task"]](request)
        json.dump({"ok": True, "data": result}, sys.stdout, ensure_ascii=False, default=str)
    except Exception as exc:
        json.dump({"ok": False, "error": str(exc), "details": traceback.format_exc()[-1500:]}, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
