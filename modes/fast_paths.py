"""Conservative routes for explicit, self-contained tasks; ambiguous work uses the model."""
import re

from .tool_runtime import from_calls
from .workspace import complete_explicit_file_plan


def explicit_plan(state, question):
    target = state.get("revision_target")
    if target and target["name"].lower().endswith(".docx"):
        title = re.fullmatch(r"请修改《(.+?)》[：:]\s*把标题改(?:为|成)(.{1,80}?)[，,]\s*保留其他内容[。.]?(?:[。.]?请保留原文件，生成可下载的新版本[。.]?)?", question)
        if title and title[1] == target["name"]:
            return {**from_calls({"calls": [{"name": "revise_document_title", "args": {"artifact_id": target["artifact_id"], "new_title": title[2]}}],
                    "reason": "Explicitly selected DOCX title change; preserve the remaining document."}),
                    "planner": "explicit_title_revision", "terminal_tools": True}
    # Follow-up tasks, selected artifacts, uploaded documents and preferences can
    # carry additional requirements. Only the table route may use uploaded files.
    users = [m for m in state["messages"] if m["role"] == "user"]
    if len(users) != 1 or state.get("revision_target") or state.get("memory_profile"):
        return None
    files = [f for f in state.get("attachments", []) if f.get("active", True)]
    if files:
        match = re.fullmatch(r"(?:请)?(?:计算|统计)(?:上传的?|这份)?表格中(.+?)的(合计与平均数|合计|平均数)[，,](?:并)?生成(柱状图|折线图)供下载[。.]?", question)
        tables = [f for f in files if f.get("kind") == "table"]
        if not match or len(files) != 1 or len(tables) != 1 or match[1] not in tables[0].get("numeric_columns", []) or len(tables[0].get("sheets", [])) > 1:
            return None
        plan = complete_explicit_file_plan({"tools": [], "calls": []}, state, question)
        if plan.get("calls") and all(c["name"] == "analyze_table" for c in plan["calls"]):
            return {**plan, "planner": "explicit_table", "terminal_tools": True}
        return None
    if re.fullmatch(r"\s*(?:你好|您好|嗨|哈喽|hello|hi|hey)[！!。.,，?？\s]*", question, re.I):
        return {**from_calls({"calls": [], "reason": "A standalone greeting needs no tools."}),
                "planner": "greeting"}
    match = re.fullmatch(r"(?:请)?(?:创建|生成|导出).{0,16}(?:Word|DOCX|docx)文档[，,]\s*标题为(.{1,80}?)[，,]\s*正文只(?:保留|使用)(?:这句话|这段文字)?[：:]([\s\S]+)", question)
    if match:
        return {**from_calls({"calls": [{"name": "create_document", "args": {"format": "docx", "name": match[1],
                         "content": "# " + match[1] + "\n\n" + match[2], "instruction": ""}}],
                         "reason": "Create a Word file using the user's explicit literal title and body."}),
                "planner": "literal_document", "terminal_tools": True}
    # Supplied text transformation and fictional/simple notices have no retrieval
    # requirement. Do not infer a fast path for knowledge questions or calculations.
    if re.search(r"https?://|联网|搜索|查找|核实|最新|最近|今天|天气|新闻|参考文献|指南|证据|计算|统计|下载|文件|Word|DOCX|Excel|MDRPI|患者|压力性损伤|根据(?:上|前|刚)|刚才|之前|上面|前面", question, re.I):
        return None
    transform = re.match(r"(?:请)?(?:翻译(?:成(?:中文|英文|英语))?|润色|改写|校对)[：:]\s*\S", question)
    notice = re.fullmatch(r"用(?:中文|英文|英语)写一封.{0,12}(?:通知|邀请函|感谢信)[：:][\s\S]+", question)
    if transform or notice:
        return {**from_calls({"calls": [], "reason": "Self-contained supplied text task."}), "planner": "explicit_text"}
    return None


def requested_file_output(question):
    return bool(re.search(r"Word|DOCX|Excel|XLSX|PDF|PPT|CSV|文件|下载|导出|保存|文档|报告|幻灯片", question, re.I))


def keep_requested_documents(plan, question, state):
    if requested_file_output(question) or state.get("revision_target") or state.get("attachments") or len([m for m in state["messages"] if m["role"] == "user"]) > 1:
        return plan
    calls = plan.get("calls")
    if calls is not None and any(c["name"] == "create_document" for c in calls):
        return {**plan, **from_calls({"calls": [c for c in calls if c["name"] != "create_document"],
                    "reason": plan.get("reason", "")}), "document_output_filtered": True}
    return plan
