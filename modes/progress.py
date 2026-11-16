"""UI labels for real execution events; no model calls or estimated percentages."""
import time

from .language import localized


PHASES = {
    "正在理解问题…": ("理解问题与上下文", "Reading your question and context"),
    "正在检索参考资料…": ("检索参考资料", "Retrieving reference material"),
    "正在搜索网页并读取相关内容…": ("搜索网页与读取内容", "Searching and reading web pages"),
    "正在搜索网页并读取正文…": ("搜索网页与读取内容", "Searching and reading web pages"),
    "正在读取上传文件的相关片段…": ("查找文件中的相关内容", "Finding relevant file excerpts"),
    "正在读取系统时间…": ("读取系统时间", "Reading the system clock"),
    "正在查询天气…": ("查询当地天气与预报", "Checking local weather and forecasts"),
    "正在读取指定网页…": ("读取你提供的网页", "Reading the page you provided"),
    "正在计算…": ("计算数值", "Calculating"),
    "对话较长，正在整理早期对话摘要；原文会保留…": ("整理早期对话摘要", "Summarizing earlier conversation"),
    "正在回答…": ("准备回答", "Preparing your answer"),
    "正在准备练习案例…": ("准备练习案例", "Preparing your exercise"),
    "正在阅读你的回答并准备反馈…": ("结合此前作答，准备本轮反馈", "Reviewing your answer with earlier attempts"),
    "正在准备提示…": ("准备解释性提示", "Preparing an explanatory hint"),
    "正在整理已有反馈…": ("整理已有反馈", "Preparing existing feedback"),
    "正在生成六步总结…": ("整理六步总结", "Preparing the six-step summary"),
    "正在运行原性能测试流程，耗时可能较长…": ("运行六步生成与反思流程", "Running the six-step answer and reflection workflow"),
    "streaming": ("正在输出回答", "Writing your answer"),
}


class ResponseProgress:
    def __init__(self, language="zh"):
        self.language = language
        self.started = time.monotonic()
        self.ended = None
        self.status = "running"
        self.label = localized(language, "正在处理本轮请求", "Processing your request")
        self.history = []

    def phase(self, event):
        if self.status != "running":
            return False
        label = localized(self.language, *PHASES.get(event, ("正在处理本轮请求", "Processing your request")))
        if self.label == label:
            return False
        self.label = label
        self.history.append(label)
        return True

    def finish(self, error=False, cancelled=False):
        if self.status == "running":
            self.ended = time.monotonic()
            self.status = "cancelled" if cancelled else "error" if error else "done"

    def props(self):
        return {"language": self.language, "status": self.status, "label": self.label,
                "history": list(self.history),
                "elapsed_ms": round(((self.ended if self.ended is not None else time.monotonic()) - self.started) * 1000)}
