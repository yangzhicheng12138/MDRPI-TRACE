"""Explicit, editable preferences shared only by sessions selecting one profile."""
import hashlib
import json
from pathlib import Path
import threading
import re
from datetime import datetime, timezone

LOCK = threading.RLock()
MUTATION_PATTERNS = {
    "remember_preference": r"记住|记下|保存.{0,12}偏好|(?:以后|今后|从现在起).{0,20}(?:回复|回答|称呼|叫我|使用|用中文|用英文)|"
                           r"偏好.*(?:修改|改成|改为)|\b(?:remember|save.{0,20}preference|from now on|always (?:reply|answer|call me|use))\b",
    "forget_preference": r"忘记|(?:清除|清空|删除).{0,20}(?:偏好|记忆|档案)|\b(?:forget|(?:remove|delete|clear).{0,30}(?:preference|memory|profile))\b",
}


def requested_mutations(question):
    """Only unquoted, unnegated instructions authorize persistent changes."""
    direct = re.sub(r'```.*?```|`[^`]*`|“[^”]*”|‘[^’]*’|"[^"\n]*"', "", question, flags=re.S)
    actions = []
    for name, pattern in MUTATION_PATTERNS.items():
        for clause in re.split(r"[，,。.!?！？;；\n]", direct):
            for match in re.finditer(pattern, clause, re.I):
                prefix = clause[:match.start()]
                if re.search(r"(?:不要|不用|不必|无需|别|禁止|不允许|不需要|请勿|勿).{0,12}$|"
                             r"\b(?:do not|don't|never|without)\b.{0,20}$|"
                             r"(?:解释|说明|介绍|翻译|讲解|如何|怎么|怎样|为什么|举例).{0,20}$|"
                             r"\b(?:explain|describe|translate|how to|why)\b.{0,30}$", prefix, re.I):
                    continue
                actions.append(name)
                break
            if name in actions:
                break
    return actions


def validate_profile(name):
    name = name.strip()
    if len(name) > 80 or any(ord(c) < 32 for c in name):
        raise ValueError("偏好档案名称最多 80 字，不能含控制字符。")
    return name


def path(root, profile):
    profile = validate_profile(profile)
    if not profile:
        raise ValueError("请先在设置中填写偏好档案名称，再保存或查看跨会话偏好。")
    return Path(root) / "preferences" / (hashlib.sha256(profile.encode()).hexdigest() + ".json")


def read(root, profile):
    if not profile:
        return {"profile": "", "items": []}
    with LOCK:
        file = path(root, profile)
        return json.loads(file.read_text()) if file.exists() else {"profile": profile, "items": []}


def update(root, profile, key, value=None):
    key = key.strip()
    if not key or len(key) > 80 or (value is not None and (not value.strip() or len(value) > 2000)):
        raise ValueError("偏好需有名称（最多 80 字）与内容（最多 2000 字）。")
    with LOCK:
        file = path(root, profile)
        data = read(root, profile)
        data["items"] = [item for item in data["items"] if item["key"] != key]
        if value is not None:
            data["items"].append({"key": key, "value": value.strip(), "updated_at": datetime.now(timezone.utc).isoformat()})
        if len(data["items"]) > 50:
            raise ValueError("每个偏好档案最多保存 50 项，请先清理已有内容。")
        if sum(len(i["key"]) + len(i["value"]) for i in data["items"]) > 12000:
            raise ValueError("偏好档案总文字最多 12000 字，请先精简已有内容。")
        file.parent.mkdir(parents=True, exist_ok=True)
        pending = file.with_suffix(".tmp")
        pending.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        pending.replace(file)
        return data


def clear(root, profile):
    with LOCK:
        file = path(root, profile)
        file.unlink(missing_ok=True)
        return read(root, profile)


def invoke(ctx, name, args):
    from .tool_runtime import Result
    profile = ctx.state.get("memory_profile", "")
    if name in {"remember_preference", "forget_preference"}:
        question = next(m["content"] for m in reversed(ctx.state["messages"]) if m["role"] == "user")
        quote = args["request_quote"].strip()
        direct = re.sub(r'```.*?```|`[^`]*`|“[^”]*”|‘[^’]*’|"[^"\n]*"', "", question, flags=re.S)
        pattern = (r"记住|记下|记忆|保存|以后|今后|从现在起|偏好.*(?:修改|改成|改为)|remember|save|from now on|always"
                   if name == "remember_preference" else r"忘记|清除|清空|删除|forget|remove|delete|clear")
        if (not quote or quote not in question or not re.search(pattern, quote, re.I)
                or not re.search(pattern, direct, re.I) or name not in requested_mutations(question)):
            raise ValueError("没有找到用户明确要求修改偏好记忆的原话，未修改。")
        if name == "forget_preference" and args["all"] and not re.search(r"全部|所有|清空|整个|\b(?:all|entire|everything)\b", direct, re.I):
            raise ValueError("未找到用户清空整个偏好档案的明确要求，未清空。")
    write = None
    with LOCK:
        if name == "remember_preference":
            update(ctx.root, profile, args["key"], args["value"])
            data = read(ctx.root, profile)
            confirmed = any(item["key"] == args["key"].strip() and item["value"] == args["value"].strip()
                            for item in data["items"])
            write = {"action": name, "profile": profile, "key": args["key"].strip(), "value": args["value"].strip(),
                     "write_confirmed": confirmed}
        elif name == "forget_preference":
            clear(ctx.root, profile) if args["all"] else update(ctx.root, profile, args["key"])
            data = read(ctx.root, profile)
            confirmed = not data["items"] if args["all"] else all(item["key"] != args["key"].strip() for item in data["items"])
            write = {"action": name, "profile": profile, "key": args["key"].strip(), "all": args["all"],
                     "write_confirmed": confirmed}
        else:
            data = read(ctx.root, profile)
    if write is not None and not write["write_confirmed"]:
        raise ValueError("偏好写入后的读取核对未通过，请稍后重试；本次不能确认已保存。")
    ctx.runtime["preferences"] = data
    result = {"preferences": data}
    if write is not None:
        result["preference_write"] = write
    return Result(name, args, data=result, events=[{"provider": "preferences", "status": "available", "action": name,
                                                 "write_confirmed": bool(write and write["write_confirmed"])}])


def extra_tools():
    from .tool_runtime import Tool, parameters, text
    return [Tool("remember_preference", "Save/change a preference ONLY when the user explicitly asks to remember it across conversations. Do not infer preferences or store uploaded documents.",
                 parameters(key=text(80), value=text(2000), request_quote=text(1200)), "memory", lambda ctx,args:invoke(ctx,"remember_preference",args)),
            Tool("read_preferences", "Show preferences of the explicitly selected shared profile.", parameters(), "memory", lambda ctx,args:invoke(ctx,"read_preferences",args)),
            Tool("forget_preference", "Delete the named preference, or all preferences when the user explicitly requests clearing the whole profile.",
                 parameters(key=text(80), all={"type": "boolean"}, request_quote=text(1200)), "memory", lambda ctx,args:invoke(ctx,"forget_preference",args))]
