from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MaterialGenerationPreview:
    session_id: str
    transcript_count: int
    character_count: int
    connection_name: str
    model: str
    base_url: str
    reasoning_level: str


def legacy_summary_to_markdown(result: Mapping[str, object]) -> str:
    """Keep old adapters readable while new adapters return Markdown directly."""
    lines = ["# 会话总结", "", str(result.get("summary") or "暂无会话摘要。")]
    knowledge_points = result.get("knowledge_points")
    lines.extend(["", "## 知识点"])
    if isinstance(knowledge_points, list) and knowledge_points:
        for index, item in enumerate(knowledge_points, 1):
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or f"知识点 {index}"
            lines.extend(["", f"### {index}. {name}", "", str(item.get("explanation") or "")])
            if item.get("evidence_time_s") is not None:
                lines.extend(["", f"> 字幕证据：{item['evidence_time_s']} 秒附近"])
    else:
        lines.extend(["", "暂未提取到明确的知识点。"])

    mistakes = result.get("mistakes")
    lines.extend(["", "## 疑问与错题"])
    if isinstance(mistakes, list) and mistakes:
        for index, item in enumerate(mistakes, 1):
            if not isinstance(item, Mapping):
                continue
            issue = item.get("issue") or f"问题 {index}"
            lines.extend(
                [
                    "",
                    f"### {index}. {issue}",
                    "",
                    f"**订正：** {item.get('correction') or '暂无订正内容。'}",
                ]
            )
            metadata = []
            if item.get("evidence_time_s") is not None:
                metadata.append(f"字幕证据：{item['evidence_time_s']} 秒附近")
            if item.get("confidence"):
                metadata.append(f"置信度：{item['confidence']}")
            if metadata:
                lines.extend(["", "> " + "；".join(metadata)])
    else:
        lines.extend(["", "本次字幕中未确认到可提取的错题或错误。"])
    return "\n".join(lines)
