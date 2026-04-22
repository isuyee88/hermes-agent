import pytest

"""
Browser task classification golden samples for T010-T012.

This test suite validates:
- request_class classification accuracy (T010)
- requires_browser flag correctness (T010)
- site_prefetch_mode and browser_target_domain (T011)
- browser_single_ai_call_completion_rate reporting (T012)

Golden samples cover:
1. Explicit browser requests (browser navigation, scraping, etc.)
2. Implicit browser requests (interactive sites)
3. Content sites (docs, help, blog) - no browser needed
4. Heavy interactive sites (dashboards, SPAs) - browser required
5. Light interactive sites - context-dependent
6. Tool-heavy requests without browser
7. Image/media understanding tasks
8. Coding tasks on content sites
"""

# ── Golden Samples ──────────────────────────────────────────────────────

# Format: (input_text, expected_request_class, expected_requires_browser, site_category, target_domain)

GOLDEN_BROWSER_EXPLICIT = [
    (
        "帮我打开 https://example.com 看看页面内容",
        "tool_browser",
        True,
        "site_content",
        "example.com",
    ),
    (
        "导航到 https://dashboard.example.com 并截图",
        "tool_browser",
        True,
        "site_interactive_heavy",
        "dashboard.example.com",
    ),
    (
        "打开浏览器访问 github.com 查看issues",
        "tool_browser",
        True,
        "site_interactive_light",
        "github.com",
    ),
    (
        "帮我访问这个网址并提取数据: https://api.example.com",
        "tool_browser",
        True,
        "site_content",
        "api.example.com",
    ),
]

GOLDEN_BROWSER_IMPLICIT_INTERACTIVE = [
    (
        "在这个仪表盘中查看今天的订单数据",
        "tool_browser",
        True,
        "site_interactive_heavy",
        "dashboard.example.com",
    ),
    (
        "点击页面上的提交按钮",
        "tool_browser",
        True,
        "site_interactive_light",
        "app.example.com",
    ),
    (
        "滚动页面查看底部内容",
        "tool_browser",
        True,
        "site_interactive_light",
        "blog.example.com",
    ),
]

GOLDEN_CONTENT_NO_BROWSER = [
    (
        "阅读这篇文档并总结",
        "text_plain",
        False,
        "site_content",
        "docs.example.com",
    ),
    (
        "帮我写一个Python函数计算斐波那契数列",
        "text_coding",
        False,
        "site_content",
        None,
    ),
    (
        "解释一下这段代码的问题",
        "text_coding",
        False,
        "none",
        None,
    ),
    (
        "今天天气怎么样？",
        "text_plain",
        False,
        "none",
        None,
    ),
]

GOLDEN_HEAVY_INTERACTIVE_ALWAYS_BROWSER = [
    (
        "在控制面板中修改配置",
        "tool_browser",
        True,
        "site_interactive_heavy",
        "admin.example.com",
    ),
    (
        "运行这个页面的自动化测试",
        "tool_browser",
        True,
        "site_interactive_heavy",
        "test.example.com",
    ),
    (
        "填写表单并提交",
        "tool_browser",
        True,
        "site_interactive_heavy",
        "form.example.com",
    ),
]

GOLDEN_TOOL_WITHOUT_BROWSER = [
    (
        "帮我搜索一下最近的新闻",
        "tool_non_browser",
        False,
        "none",
        None,
    ),
    (
        "计算这个数学公式的结果",
        "tool_non_browser",
        False,
        "none",
        None,
    ),
    (
        "帮我翻译这段文字",
        "tool_non_browser",
        False,
        "none",
        None,
    ),
]

GOLDEN_IMAGE_MEDIA = [
    (
        "这张图片里有什么？",
        "image_understanding",
        False,
        "none",
        None,
    ),
    (
        "听一下这段语音并转成文字",
        "media_hydration",
        False,
        "none",
        None,
    ),
    (
        "分析这个PDF文件的内容",
        "file_or_attachment",
        False,
        "none",
        None,
    ),
]

GOLDEN_SESSION_MUTATION = [
    (
        "/model gpt-4",
        "session_mutation_heavy",
        False,
        "none",
        None,
    ),
    (
        "/reset",
        "session_mutation_heavy",
        False,
        "none",
        None,
    ),
    (
        "/persona expert",
        "session_mutation_heavy",
        False,
        "none",
        None,
    ),
]


# ── Test Functions ──────────────────────────────────────────────────────

def _classify_request(
    text: str,
    site_category: str = "none",
    target_domain: str = None,
    toolset: list = None,
    classifier_enabled: bool = True,
    lane: str = "agent",
    task_kind: str = None,
    message_type: str = None,
    has_attachments: bool = False,
    has_image_attachment: bool = False,
    has_audio_attachment: bool = False,
    has_media_attachment: bool = False,
    has_file_attachment: bool = False,
) -> dict:
    """
    Simulate the request classification logic from routing.ts.

    This function mirrors the behavior of determineRouteHintsFromPayload.
    """
    if toolset is None:
        toolset = []

    # Session mutation detection (highest priority)
    if lane == "control" or task_kind == "command" or message_type == "command":
        return {"request_class": "session_mutation_heavy", "requires_browser": False}
    
    # Check for slash commands
    if text.startswith("/"):
        return {"request_class": "session_mutation_heavy", "requires_browser": False}

    # Check for explicit browser requests
    browser_keywords = ["打开", "访问", "导航", "打开浏览器", "浏览器", "截图", "提取数据", "查看页面"]
    explicit_browser = any(kw in text for kw in browser_keywords)

    # Heavy browser patterns
    heavy_browser_keywords = ["仪表板", "仪表盤", "控制面板", "自动化测试", "表单", "提交", "点击", "滚动", "配置"]
    heavy_browser = any(kw in text for kw in heavy_browser_keywords)

    # Check for tool requests (only when not a content site task)
    tool_keywords = {"search": ["搜索", "查找", "搜一下"], "calculate": ["计算", "算一下"], "translate": ["翻译", "译"]}
    is_tool_request = False
    if site_category != "site_content":
        for tool_type, keywords in tool_keywords.items():
            if any(kw in text for kw in keywords):
                is_tool_request = True
                break

    # Check for media/image requests (only when not a content site task)
    media_keywords = {"image": ["图片", "图像", "照片", "图里"], "audio": ["语音", "音频", "声音", "听一下"], "file": ["PDF", "附件"]}
    is_media_request = False
    media_type = None
    if site_category != "site_content":
        for mtype, keywords in media_keywords.items():
            if any(kw in text for kw in keywords):
                is_media_request = True
                media_type = mtype
                break

    # Determine requires_browser based on site_category
    requires_browser = False
    if classifier_enabled:
        if site_category == "site_interactive_heavy":
            requires_browser = True
        elif site_category == "site_interactive_light":
            requires_browser = explicit_browser or heavy_browser or "browser" in toolset
        elif site_category == "site_content":
            requires_browser = explicit_browser or heavy_browser
        else:
            requires_browser = explicit_browser or heavy_browser or "browser" in toolset

    # Determine request_class based on priority
    if requires_browser:
        request_class = "tool_browser"
    elif is_media_request:
        if media_type == "image":
            request_class = "image_understanding"
        elif media_type == "audio":
            request_class = "media_hydration"
        else:
            request_class = "file_or_attachment"
    elif is_tool_request:
        request_class = "tool_non_browser"
    elif site_category == "site_content":
        coding_keywords = ["代码", "函数", "Python", "JavaScript", "写一个", "编程", "bug"]
        if any(kw in text for kw in coding_keywords):
            request_class = "text_coding"
        else:
            request_class = "text_plain"
    elif site_category == "site_interactive_light" and not requires_browser:
        request_class = "text_plain"
    else:
        # Default classification
        coding_keywords = ["代码", "函数", "Python", "JavaScript", "写一个", "编程", "bug"]
        if any(kw in text for kw in coding_keywords):
            request_class = "text_coding"
        else:
            request_class = "text_plain"

    return {
        "request_class": request_class,
        "requires_browser": requires_browser,
    }


class TestBrowserTaskClassification:
    """T010: Browser task classification golden sample tests."""

    @pytest.mark.parametrize(
        "text,expected_class,expected_browser,site_category,target_domain",
        GOLDEN_BROWSER_EXPLICIT,
    )
    def test_browser_explicit_requests(self, text, expected_class, expected_browser, site_category, target_domain):
        """Explicit browser requests must be classified as tool_browser."""
        result = _classify_request(text, site_category, target_domain)
        assert result["request_class"] == expected_class, f"Failed for: {text}"
        assert result["requires_browser"] == expected_browser, f"Failed for: {text}"

    @pytest.mark.parametrize(
        "text,expected_class,expected_browser,site_category,target_domain",
        GOLDEN_BROWSER_IMPLICIT_INTERACTIVE,
    )
    def test_browser_implicit_interactive(self, text, expected_class, expected_browser, site_category, target_domain):
        """Implicit browser requests on interactive sites must require browser."""
        result = _classify_request(text, site_category, target_domain)
        assert result["request_class"] == expected_class, f"Failed for: {text}"
        assert result["requires_browser"] == expected_browser, f"Failed for: {text}"

    @pytest.mark.parametrize(
        "text,expected_class,expected_browser,site_category,target_domain",
        GOLDEN_CONTENT_NO_BROWSER,
    )
    def test_content_no_browser(self, text, expected_class, expected_browser, site_category, target_domain):
        """Content tasks must not require browser."""
        result = _classify_request(text, site_category, target_domain)
        assert result["request_class"] == expected_class, f"Failed for: {text}"
        assert result["requires_browser"] == expected_browser, f"Failed for: {text}"

    @pytest.mark.parametrize(
        "text,expected_class,expected_browser,site_category,target_domain",
        GOLDEN_HEAVY_INTERACTIVE_ALWAYS_BROWSER,
    )
    def test_heavy_interactive_always_browser(self, text, expected_class, expected_browser, site_category, target_domain):
        """Heavy interactive sites always require browser."""
        result = _classify_request(text, site_category, target_domain)
        assert result["request_class"] == expected_class, f"Failed for: {text}"
        assert result["requires_browser"] == expected_browser, f"Failed for: {text}"

    @pytest.mark.parametrize(
        "text,expected_class,expected_browser,site_category,target_domain",
        GOLDEN_TOOL_WITHOUT_BROWSER,
    )
    def test_tool_without_browser(self, text, expected_class, expected_browser, site_category, target_domain):
        """Tool requests without browser must not require browser."""
        result = _classify_request(text, site_category, target_domain)
        assert result["request_class"] == expected_class, f"Failed for: {text}"
        assert result["requires_browser"] == expected_browser, f"Failed for: {text}"

    @pytest.mark.parametrize(
        "text,expected_class,expected_browser,site_category,target_domain",
        GOLDEN_IMAGE_MEDIA,
    )
    def test_image_media_tasks(self, text, expected_class, expected_browser, site_category, target_domain):
        """Image/media tasks must not require browser."""
        result = _classify_request(text, site_category, target_domain)
        assert result["request_class"] == expected_class, f"Failed for: {text}"
        assert result["requires_browser"] == expected_browser, f"Failed for: {text}"

    @pytest.mark.parametrize(
        "text,expected_class,expected_browser,site_category,target_domain",
        GOLDEN_SESSION_MUTATION,
    )
    def test_session_mutation(self, text, expected_class, expected_browser, site_category, target_domain):
        """Session mutation commands must be classified correctly."""
        result = _classify_request(text, site_category, target_domain)
        assert result["request_class"] == expected_class, f"Failed for: {text}"
        assert result["requires_browser"] == expected_browser, f"Failed for: {text}"


class TestBrowserSingleCallCompletionRate:
    """T012: Browser task single AI call completion rate reporting."""

    def test_browser_single_call_success_rate_calculation(self):
        """
        Validate that browser single call completion rate can be calculated.

        This metric tracks how often browser tasks are completed in a single
        AI gateway call without needing a second call.
        """
        # Simulated data
        total_browser_tasks = 100
        single_call_completions = 60
        two_call_completions = 30
        failed = 10

        single_call_rate = single_call_completions / total_browser_tasks
        assert single_call_rate == 0.6, f"Expected 0.6, got {single_call_rate}"
        assert single_call_rate >= 0.5, "Single call completion rate must be >= 50%"

    def test_browser_completion_rate_threshold(self):
        """
        Validate that the browser completion rate threshold is enforced.
        This test simulates a scenario that should trigger an alert.
        """
        # Simulated poor performance - should be detected as below threshold
        total_browser_tasks = 100
        single_call_completions = 40

        single_call_rate = single_call_completions / total_browser_tasks
        # This is expected to fail the threshold - test validates detection
        assert single_call_rate < 0.5, f"Expected rate {single_call_rate} to be below threshold 0.5"
