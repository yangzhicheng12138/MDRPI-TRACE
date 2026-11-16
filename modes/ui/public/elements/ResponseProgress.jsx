// Uses Chainlit CustomElement and native streaming; no extra model requests.
import { useEffect, useState } from "react";
import { CheckCircle2, CircleAlert, Clock3, LoaderCircle, CircleStop } from "lucide-react";

export default function ResponseProgress() {
  const running = props.status === "running";
  const failed = props.status === "error";
  const stopped = props.status === "cancelled";
  const zh = props.language === "zh";
  const [elapsed, setElapsed] = useState(props.elapsed_ms);
  useEffect(() => {
    const baseline = performance.now();
    setElapsed(props.elapsed_ms);
    if (!running) return;
    const timer = setInterval(() => setElapsed(props.elapsed_ms + performance.now() - baseline), 1000);
    return () => clearInterval(timer);
  }, [props.elapsed_ms, running]);
  const seconds = Math.floor(elapsed / 1000);
  const duration = seconds < 60 ? `${seconds}${zh ? " 秒" : "s"}` :
    `${Math.floor(seconds / 60)}${zh ? " 分 " : "m "}${seconds % 60}${zh ? " 秒" : "s"}`;
  const Icon = running ? LoaderCircle : stopped ? CircleStop : failed ? CircleAlert : CheckCircle2;
  const title = running ? props.label : stopped ? (zh ? "本轮已停止" : "Generation stopped") : failed ? (zh ? "本轮未完成" : "Turn incomplete") :
    (zh ? "本轮已完成" : "Turn complete");
  return (
    <div className="mdrpi-response-progress rounded-lg border bg-muted/30 p-3 text-sm"
         data-status={props.status} data-elapsed={seconds}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2" role="status" aria-live="polite">
          <Icon className={running ? "h-4 w-4 animate-spin motion-reduce:animate-none" : "h-4 w-4"} aria-hidden="true" />
          <span className="font-medium">{title}</span>
        </div>
        <span className="flex shrink-0 items-center gap-1 text-muted-foreground" aria-live="off">
          <Clock3 className="h-3 w-3" aria-hidden="true" />
          <span>{running ? (zh ? "已等待 " : "Elapsed ") : (zh ? "用时 " : "Took ")}{duration}</span>
        </span>
      </div>
      {running && <p className="mt-2 text-xs text-muted-foreground">
        {props.label === (zh ? "正在输出回答" : "Writing your answer") ?
          (zh ? "回答会逐步显示在下方。" : "The answer is appearing below as it is generated.") :
          (zh ? "处理进行中，完成后会显示结果。" : "Processing is in progress; the result will appear when ready.")}
      </p>}
      {props.history.length > 0 && <details className="mt-2 text-xs text-muted-foreground">
        <summary className="cursor-pointer">{zh ? "本轮处理阶段" : "Stages in this turn"}</summary>
        <div className="mt-2 flex flex-wrap gap-2">
          {props.history.map((label, index) => <span key={index} className="rounded border px-2 py-1">{label}</span>)}
        </div>
      </details>}
    </div>
  );
}
