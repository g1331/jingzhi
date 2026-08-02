
from jingzhi.rich_text import ASSET_ROOT, RENDERER_PATH
from jingzhi.ui import MainWindow


def test_renderer_uses_local_sanitized_markdown_and_math_assets() -> None:
    html = RENDERER_PATH.read_text(encoding="utf-8")
    renderer = (ASSET_ROOT / "renderer.js").read_text(encoding="utf-8")

    assert "https://" not in html
    assert "Content-Security-Policy" in html
    assert "marked.umd.js" in html
    assert "purify.min.js" in html
    assert "katex.min.js" in html
    assert "DOMPurify.sanitize" in renderer
    assert "renderMathInElement" in renderer
    assert "\\\\\\[" in renderer
    assert (ASSET_ROOT / "vendor" / "katex" / "fonts" / "KaTeX_Main-Regular.woff2").is_file()


def test_summary_is_formatted_as_readable_markdown() -> None:
    result = {
        "summary": "本节讨论函数连续性。",
        "knowledge_points": [
            {"name": "连续性", "explanation": r"使用 $\\epsilon$ 定义。", "evidence_time_s": 12}
        ],
        "mistakes": [
            {
                "issue": "混淆左右极限",
                "correction": "分别检查左右极限。",
                "evidence_time_s": 25,
                "confidence": "high",
            }
        ],
    }

    markdown = MainWindow._format_summary(result)

    assert markdown.startswith("# 会话总结")
    assert "## 知识点" in markdown
    assert r"$\\epsilon$" in markdown
    assert "## 疑问与错题" in markdown
    assert "分别检查左右极限" in markdown
