"""Read-only placement of displayed teaching help for the current task.

No answer key, rubric, generation, scoring or hint-accounting mutation is performed here.
Original messages and learning records remain intact; uncertain legacy help
stays in the conversation instead of being assigned to another step or hint.
"""
from copy import deepcopy
import hashlib
import re


_EXPLANATIONS = {"explain", "clarify"}


def _exercise_id(state):
    return state.get("exercise_id") or (state.get("session_id", "") + ":first")


def _current_step(state):
    flow = state.get("clinical_flow") or {}
    if not flow:
        return None
    steps, index = flow.get("steps"), flow.get("current_step")
    if (flow.get("version") != "guided_v1" or not isinstance(steps, list)
            or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(steps)
            or not isinstance(steps[index], dict) or not steps[index].get("id")):
        return False
    return steps[index]


def _hint_events(state, step):
    if step:
        return deepcopy(step.get("hints") or [])
    from .learner_evidence import hint_events
    return hint_events(state)


def support_reply_metadata(state, saved, kind, intent=None):
    """Attach only message metadata; neither state nor accounting is changed."""
    if state.get("mode") != "teaching" or not saved.get("case"):
        return {}
    step = _current_step(saved)
    if step is False:
        return {}
    metadata = {"exercise_id": _exercise_id(saved), "step_id": step["id"] if step else ""}
    if kind == "clarification" and intent not in {"hint_detail", "explain", "clarify"}:
        return metadata  # Uncertain input-purpose clarification stays central.
    if kind == "help" and intent == "hint":
        hints = _hint_events(saved, step)
        if hints:
            metadata.update(support_intent="hint_notice", hint_at=hints[-1].get("at", ""))
        return metadata
    if kind not in {"hint", "reference", "teaching_dialogue", "social", "clarification"}:
        return metadata
    if kind == "hint":
        before_step = _current_step(state)
        before = _hint_events(state, before_step) if before_step is not False else []
        after = _hint_events(saved, step)
        if len(after) <= len(before):
            if not after:
                return metadata
            # Keep an exhaustion notice with the last shown hint without making
            # an extra hint card or changing the saved hint count.
            metadata.update(support_intent="hint_notice", hint_at=after[-1].get("at", ""))
        else:
            hint = after[-1]
            metadata.update(support_intent="hint", hint_level=hint.get("level") or len(after),
                            hint_at=hint.get("at", ""))
    elif kind == "reference":
        metadata["support_intent"] = "reference"
    else:
        if intent not in {"hint_detail", "explain", "clarify", "social"}:
            events = saved.get("teaching_dialogue_events") or []
            intent = events[-1].get("intent") if events else None
        if intent not in {"hint_detail", "explain", "clarify", "social"}:
            return metadata
        metadata["support_intent"] = intent
        if intent == "hint_detail":
            hints = _hint_events(saved, step)
            if hints:
                metadata.update(hint_level=hints[-1].get("level") or len(hints), hint_at=hints[-1].get("at", ""))
    turn_id = state.get("turn_id")
    if turn_id and any(row.get("role") == "user" and row.get("turn_id") == turn_id for row in state.get("messages") or []):
        metadata["help_request_id"] = turn_id
    return metadata


def _identity(row):
    return row.get("at"), row.get("role"), row.get("kind"), row.get("content")


def _current_message_pairs(state, projected):
    """Match this exercise's original messages to their display-copy indices."""
    lower = 0
    if state.get("practice_history"):
        lower = max((i + 1 for i, row in enumerate(projected) if row.get("kind") == "learning_section"), default=len(projected))
    cursor, pairs = len(projected) - 1, []
    for source_index in range(len(state.get("messages") or []) - 1, -1, -1):
        row = state["messages"][source_index]
        match = next((i for i in range(cursor, lower - 1, -1) if _identity(projected[i]) == _identity(row)), None)
        if match is not None:
            pairs.append((source_index, match, row, projected[match]))
            cursor = match - 1
    return list(reversed(pairs))


def _step_boundary(state, step):
    if not step:
        return 0
    flow, boundary = state["clinical_flow"], None
    for index, row in enumerate(state.get("messages") or []):
        if row.get("role") != "assistant" or row.get("kind") not in {"case", "step_question"}:
            continue
        if row.get("step_id") == step["id"]:
            boundary = index
            continue
        content = row.get("content", "")
        match = re.search(r"(?:第\s*([1-6])\s*/\s*6|Step\s+([1-6])\s*/\s*6)", content, re.I)
        if match and int(match[1] or match[2]) - 1 == flow["current_step"]:
            boundary = index
    return boundary


def _in_scope(row, index, exercise_id, step, boundary):
    if row.get("exercise_id") and row["exercise_id"] != exercise_id:
        return False
    if step:
        if row.get("step_id"):
            return row["step_id"] == step["id"]
        return boundary is not None and index > boundary
    return not row.get("step_id")


def _id(exercise_id, step_id, kind, at, text, index):
    raw = "\0".join(str(value) for value in (exercise_id, step_id, kind, at, text, index))
    return "support-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _content(row):
    value = row.get("display_content", row.get("content", ""))
    return value if isinstance(value, str) else ""


def _pending_support(state, step, objective):
    """Recover only an already accepted helper route, never default send input."""
    if state.get("cancelled") or state.get("error") or state.get("route_pending"):
        return None
    mixed = state.get("mixed_help_pending")
    if mixed and not state.get("mixed_grade_done"):
        return None
    if not state.get("pending") and not (mixed and state.get("mixed_grade_done")):
        return None
    route = state.get("input_route") or {}
    operation = state.get("operation") or state.get("request_action")
    if mixed:
        intent = route.get("help_intent")
    elif operation == "teaching_dialogue":
        if route.get("confidence") != "clear" or route.get("reason"):
            return None
        intent = route.get("intent")
    else:
        intent = operation
    if intent not in {"hint", "stuck", "hint_detail", "explain", "clarify", "reference"}:
        return None
    from .teaching_access import reference_access
    access = reference_access(state)
    if objective:
        if intent not in {"hint", "stuck"}:
            return None
        placement = "hint"
    elif intent in {"reference", "stuck"}:
        placement = "reference" if access["allowed"] else "hint"
    elif intent in {"explain", "clarify"}:
        placement = "explanation"
    else:
        placement = "hint"
        if intent == "hint" and not step and state.get("reviews"):
            latest = state["reviews"][-1]
            if latest.get("ready_for_summary") and not latest.get("material_errors") and access["allowed"]:
                return None
    request = next((row for row in reversed(state.get("messages") or []) if row.get("role") == "user"
                    and row.get("turn_id") == state.get("turn_id")), None)
    return {"placement": placement, "intent": intent,
            "hide_request": bool(request and request.get("kind") == "help" and not request.get("answer_text"))}


def teaching_support_view(state, projected_messages):
    """Return panel content and 0-based placements for this question/step only."""
    view = {"enabled": False, "objective": False, "exercise_id": _exercise_id(state), "step_id": "",
            "hints": [], "explanations": [], "references": [],
            "message_placements": {}, "request_placements": {}, "pending": None, "failure": None,
            "hint_total": None, "hint_seen": None}
    if state.get("mode") != "teaching" or not state.get("case") or state.get("stage") in {"setup", "case", "topic"}:
        return view
    from .rubric import objective_question_type
    objective = bool(objective_question_type(state.get("bank_item") or {}))
    step = _current_step(state)
    if step is False:
        return view
    view.update(enabled=True, objective=objective, step_id=step["id"] if step else "")
    if objective:
        from .question_bank import assessment_item, objective_hint_cues
        item = assessment_item(state)
        if item.get("hints") or item.get("hint_pathways"):
            latest = (state.get("reviews") or [{}])[-1]
            language = state.get("language", "zh")
            cues, _, _ = objective_hint_cues(item, language, latest.get("focus", "all"))
            events = _hint_events(state, step)
            view["hint_total"] = len(cues)
            view["hint_seen"] = sum(any(event.get("text") == cue and event.get("language", language) == language
                                        for event in events if isinstance(event, dict)) for cue in cues)
        else:
            view["hint_total"] = 0
            view["hint_seen"] = 0
    else:
        view["hint_total"] = 3
    view["pending"] = _pending_support(state, step, objective)
    if state.get("pending") and (state.get("error") or state.get("cancelled")):
        # Recover only the location of an accepted helper failure. The original
        # error/cancelled state is untouched, and this never restores loading.
        view["failure"] = _pending_support({**state, "cancelled": False, "error": ""}, step, objective)
    exercise_id, step_id = view["exercise_id"], view["step_id"]
    boundary = _step_boundary(state, step)
    pairs = _current_message_pairs(state, projected_messages)
    scoped = [pair for pair in pairs if _in_scope(pair[2], pair[0], exercise_id, step, boundary)]
    hints = [event for event in _hint_events(state, step) if isinstance(event, dict)
             and (not event.get("step_id") or event["step_id"] == step_id)]
    if not objective:
        hints = hints[:3]
    hint_sources = {}
    for index, event in enumerate(hints):
        text = event.get("text", "")
        if not isinstance(text, str) or not text:
            continue
        level = event.get("level") if type(event.get("level")) is int else index + 1
        card = {"id": _id(exercise_id, step_id, "hint", event.get("at", ""), text, index),
                "at": event.get("at", ""), "level": level, "text": text, "details": []}
        view["hints"].append(card)
        eligible = [pair for pair in scoped if pair[2].get("role") == "assistant"
                    and pair[2].get("kind") in {"hint", "help"}]
        matches = [pair for pair in eligible if event.get("at") and pair[2].get("hint_at") == event["at"]
                   and pair[2].get("support_intent") != "hint_notice"]
        if not matches:
            matches = [pair for pair in eligible if text in pair[2].get("content", "")
                       and pair[2].get("support_intent") != "hint_notice"]
        if len(matches) == 1:
            source_index, display_index, original, projected = matches[0]
            card["text"] = _content(projected)
            card["at"] = original.get("at") or card["at"]
            hint_sources[source_index] = card
            view["message_placements"][display_index] = "hint"
        card["_event_at"] = event.get("at", "")
        card["_event_level"] = level
    dialogue_events = list((step.get("teaching_dialogue_events") if step else state.get("teaching_dialogue_events")) or [])
    if step:
        dialogue_events += [event for event in state.get("teaching_dialogue_events") or []
                            if event.get("step_id") == step_id and event not in dialogue_events]
    previous_hint = None
    for source_index, display_index, original, projected in scoped:
        if original.get("role") != "assistant":
            continue
        card, placement = hint_sources.get(source_index), None
        if card:
            previous_hint, placement = card, "hint"
        elif original.get("kind") == "reference" and not objective:
            placement = "reference"
            view["references"].append({"id": _id(exercise_id, step_id, placement, original.get("at", ""), original.get("content", ""), source_index),
                                       "at": original.get("at", ""), "content": _content(projected)})
        elif (original.get("kind") in {"teaching_dialogue", "social", "help", "hint", "clarification"}
              and (not objective or original.get("support_intent") == "hint_notice")):
            intent = original.get("support_intent")
            if original.get("kind") == "clarification" and intent not in {"hint_detail", "explain", "clarify"}:
                continue
            if not intent:
                matches = [event for event in dialogue_events if event.get("text") == original.get("content")
                           and (not event.get("step_id") or event["step_id"] == step_id)]
                if len(matches) == 1:
                    intent = matches[0].get("intent")
            if intent in {"hint_detail", "hint_notice"}:
                matches = [hint for hint in view["hints"] if original.get("hint_at")
                           and hint["_event_at"] == original["hint_at"]]
                target = matches[0] if len(matches) == 1 else previous_hint if not original.get("hint_at") else None
                if target:
                    placement = "hint"
                    card = {"id": _id(exercise_id, step_id, "detail", original.get("at", ""), original.get("content", ""), source_index),
                            "at": original.get("at", ""), "request": "", "content": _content(projected)}
                    target["details"].append(card)
            elif intent in _EXPLANATIONS:
                placement = "explanation"
                card = {"id": _id(exercise_id, step_id, placement, original.get("at", ""), original.get("content", ""), source_index),
                        "at": original.get("at", ""), "request": "", "content": _content(projected)}
                view["explanations"].append(card)
        if not placement:
            continue
        view["message_placements"][display_index] = placement
        request_id = original.get("help_request_id")
        requests = [pair for pair in scoped if pair[0] < source_index and pair[2].get("role") == "user"
                    and request_id and pair[2].get("turn_id") == request_id]
        if not request_id:
            earlier = [pair for pair in scoped if pair[0] < source_index]
            if earlier and earlier[-1][2].get("role") == "user":
                requests = [earlier[-1]]
        if len(requests) == 1:
            request = requests[0]
            # Mixed answers are stored as student with their verbatim answer_text.
            # Their original input always stays in the central conversation.
            if request[2].get("kind") == "help" and not request[2].get("answer_text"):
                view["request_placements"][request[1]] = placement
                if card and "request" in card:
                    card["request"] = request[2].get("content", "")
    for card in view["hints"]:
        card.pop("_event_at", None)
        card.pop("_event_level", None)
    return view
