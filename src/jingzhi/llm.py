from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jingzhi.context import QuestionContext
from jingzhi.transcript_correction import CorrectionRequest


class ProviderRequestError(RuntimeError):
    """A concise, user-facing provider request failure."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


@dataclass(frozen=True, slots=True)
class AnswerModelResult:
    text: str
    request_id: str | None = None
    model: str | None = None


class OpenAIContextModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "",
        api_mode: str = "responses",
    ) -> None:
        if api_mode not in {"responses", "chat_completions"}:
            raise ValueError(f"Unsupported API mode: {api_mode}")
        self.model = model
        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.api_mode = api_mode

    def _client(self):
        if not self.api_key:
            raise RuntimeError("API Key is empty; recording and local transcription still work")
        from openai import OpenAI

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _provider_error(self, exc: Exception) -> ProviderRequestError:
        raw = str(exc).strip()
        request_id = getattr(exc, "request_id", None)

        def error(message: str) -> ProviderRequestError:
            return ProviderRequestError(message, request_id=request_id)

        lowered = raw.lower()
        status = getattr(exc, "status_code", None)
        target = self.base_url or "OpenAI 官方地址"
        if "<!doctype html" in lowered or "<html" in lowered:
            return error(
                "Provider 返回了 HTML 页面，而不是 OpenAI 兼容的 JSON。"
                f"请检查 Base URL（当前：{target}）是否应以 /v1 结尾，"
                "并确认所选 API 类型与服务端一致。"
            )
        if status in {401, 403}:
            return error("Provider 拒绝鉴权，请检查 API Key 和访问权限。")
        if status == 404:
            endpoint = "/responses" if self.api_mode == "responses" else "/chat/completions"
            return error(f"Provider 未找到接口 {endpoint}。请检查 Base URL 或切换 API 类型。")
        if status == 429:
            return error("Provider 返回限流错误，请稍后重试或检查账户额度。")
        if not raw:
            raw = type(exc).__name__
        # Never let an upstream HTML document or oversized SDK dump stretch the UI.
        compact = re.sub(r"\s+", " ", raw)[:360]
        return error(f"Provider 请求失败：{compact}")

    @staticmethod
    def _image_part(path: Path) -> dict[str, str]:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        media_type = "image/webp" if path.suffix.lower() == ".webp" else "image/png"
        return {
            "type": "input_image",
            "image_url": f"data:{media_type};base64,{encoded}",
        }

    @staticmethod
    def _chat_image_part(path: Path) -> dict[str, Any]:
        image = OpenAIContextModel._image_part(path)
        return {"type": "image_url", "image_url": {"url": image["image_url"]}}

    @staticmethod
    def _chat_text(result: Any) -> str:
        content = result.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                for item in content
            )
        return str(content or "")

    def answer(self, question: str, context: QuestionContext) -> AnswerModelResult:
        content: list[dict[str, str]] = [
            {
                "type": "input_text",
                "text": (
                    "你是桌面上下文助手。回答中的每段内容必须使用以下适用的 Markdown "
                    "标题标明依据边界，不要求输出没有内容的标题：\n"
                    "## 会话证据确认：仅陈述能由所给字幕或关键帧确认的内容，并引用方括号中的"
                    "稳定证据标识与会话相对时间。\n"
                    "## 补充解释：会话证据之外的通用知识或推断，不得伪装成会话事实。\n"
                    "## 无法确认：明确说明证据不足以及缺少什么。\n"
                    f"证据状态：{'存在可用的会话证据' if context.transcripts or context.frames else '当前没有可用的会话证据'}。\n\n"
                    f"问题：{question}\n\n字幕：\n{context.transcript or '（当前没有字幕）'}"
                ),
            }
        ]
        for frame in context.frames:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"[{frame.stable_id}][{frame.ts_ms / 1000:.1f}s][{frame.source}] 关键帧"
                    ),
                }
            )
            content.append({"type": "input_image", "image_url": frame.image_url})
        try:
            client = self._client()
            if self.api_mode == "responses":
                response = client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": content}],
                )
                return AnswerModelResult(
                    response.output_text,
                    getattr(response, "id", None),
                    getattr(response, "model", self.model),
                )

            chat_content: list[dict[str, Any]] = []
            for item in content:
                if item["type"] == "input_text":
                    chat_content.append({"type": "text", "text": item["text"]})
                else:
                    chat_content.append(
                        {"type": "image_url", "image_url": {"url": item["image_url"]}}
                    )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": chat_content}],
            )
            return AnswerModelResult(
                self._chat_text(response),
                getattr(response, "id", None),
                getattr(response, "model", self.model),
            )
        except ProviderRequestError:
            raise
        except Exception as exc:
            raise self._provider_error(exc) from exc

    def test_connection(self) -> str:
        try:
            client = self._client()
            if self.api_mode == "responses":
                response = client.responses.create(model=self.model, input="只回复 OK")
                return response.output_text.strip()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "只回复 OK"}],
            )
            return self._chat_text(response).strip()
        except ProviderRequestError:
            raise
        except Exception as exc:
            raise self._provider_error(exc) from exc

    def correct(self, request: CorrectionRequest) -> dict[int, str]:
        segment_lines = "\n".join(
            f"[{item.id}][{item.start_ms}–{item.end_ms}ms][{item.source}] {item.text}"
            for item in request.context_segments
        )
        target_ids = [item.id for item in request.target_segments]
        prompt = (
            "校订指定字幕片段，只修正识别错误，不扩写或总结。相邻字幕仅供上下文参考。"
            "图片前的来源与会话时间标签是视觉证据位置。返回严格 JSON 对象，键为字幕片段 "
            f"ID，值为校订文本；必须只包含这些 ID：{target_ids}。\n\n字幕：\n{segment_lines}"
        )
        response_content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
        chat_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in request.frames:
            if not frame.path.is_file():
                continue
            label = f"关键帧来源：{frame.source_id}；会话时间：{frame.ts_ms}ms"
            response_content.append({"type": "input_text", "text": label})
            response_content.append(self._image_part(frame.path))
            chat_content.append({"type": "text", "text": label})
            chat_content.append(self._chat_image_part(frame.path))
        try:
            client = self._client()
            if self.api_mode == "responses":
                response = client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": response_content}],
                )
                raw = response.output_text.strip()
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": chat_content}],
                )
                raw = self._chat_text(response).strip()
        except ProviderRequestError:
            raise
        except Exception as exc:
            raise self._provider_error(exc) from exc
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            parsed = json.loads(raw)
            corrected = {int(segment_id): str(text).strip() for segment_id, text in parsed.items()}
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            preview = re.sub(r"\s+", " ", raw)[:240]
            raise ProviderRequestError(f"字幕校订模型未返回要求的 JSON：{preview}") from exc
        if any(not text for text in corrected.values()):
            raise ProviderRequestError("字幕校订模型返回了空文本")
        return corrected

    def summarize(self, transcript: str) -> dict[str, Any]:
        prompt = (
            "根据会话字幕生成严格 JSON，不要使用 Markdown 代码块。结构必须是："
            '{"summary":"...","knowledge_points":[{"name":"...","explanation":"...",'
            '"evidence_time_s":0}],"mistakes":[{"issue":"...","correction":"...",'
            '"evidence_time_s":0,"confidence":"high|medium|low"}]。'
            "mistakes 只提取字幕中能确认的错误、困惑或纠正；不能确认时返回空数组。\n\n"
            f"字幕：\n{transcript}"
        )
        try:
            client = self._client()
            if self.api_mode == "responses":
                response = client.responses.create(model=self.model, input=prompt)
                raw = response.output_text.strip()
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = self._chat_text(response).strip()
        except ProviderRequestError:
            raise
        except Exception as exc:
            raise self._provider_error(exc) from exc
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            preview = re.sub(r"\s+", " ", raw)[:240]
            raise ProviderRequestError(f"模型未返回要求的 JSON：{preview}") from exc
        if not isinstance(parsed, dict):
            raise TypeError("Summary response is not a JSON object")
        return parsed
