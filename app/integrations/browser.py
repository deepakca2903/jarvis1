import re
import logging
from typing import Dict, Any
from urllib.parse import urlparse

import httpx
from app.config import settings

logger = logging.getLogger(__name__)


def get_browser_status() -> Dict[str, Any]:
    """Returns status and capabilities of browser automation."""
    is_enabled = bool(settings.ENABLE_BROWSER_AUTOMATION)
    return {
        "name": "Playwright Automation",
        "status": "ready" if is_enabled else "pending_setup",
        "configured": is_enabled,
        "auth_type": "Headless Browser Automation",
        "description": "Authorized browser navigation, page inspection, and web content extraction.",
        "docs_url": "https://playwright.dev/python/",
        "setup_hint": (
            "Ready for web tasks."
            if is_enabled
            else "Set ENABLE_BROWSER_AUTOMATION=true in backend/.env to activate."
        ),
    }


async def browse_web_page(url: str, extract_summary: bool = True) -> Dict[str, Any]:
    """
    Automated browser tool to inspect a webpage, extract text content, and retrieve page metadata.
    Enforces URL validation and safe timeout boundaries.
    """
    if not settings.ENABLE_BROWSER_AUTOMATION:
        return {
            "success": False,
            "status": "pending_setup",
            "message": "Browser automation is disabled. Set ENABLE_BROWSER_AUTOMATION=true in backend/.env.",
        }

    parsed = urlparse(url)
    if parsed.scheme not in ["http", "https"]:
        return {
            "success": False,
            "error": "Invalid URL protocol. Only http:// and https:// URLs are supported.",
        }

    title = ""
    content = ""
    engine_used = "playwright"

    # 1. Attempt Playwright headless browser
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=12000, wait_until="domcontentloaded")
            title = await page.title()
            content = await page.inner_text("body")
            await browser.close()
    except Exception as pw_err:
        logger.info("Playwright headless execution fallback (%s). Attempting HTTP client extraction...", pw_err)
        engine_used = "httpx_parser"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                res = await client.get(url, headers={"User-Agent": "Jarvis-AI-Agent/0.3.0"})
                res.raise_for_status()
                html = res.text

                # Extract title
                title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else url

                # Strip script, style, html tags
                clean = re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", "", html, flags=re.IGNORECASE)
                clean = re.sub(r"<[^>]+>", " ", clean)
                content = " ".join(clean.split())
        except Exception as http_err:
            return {
                "success": False,
                "url": url,
                "error": f"Failed to retrieve page content: {str(http_err)}",
            }

    # Clean and limit content for agent context safety
    clean_text = " ".join(content.split())[:4000]

    return {
        "success": True,
        "url": url,
        "title": title,
        "engine": engine_used,
        "content_length": len(clean_text),
        "content": clean_text,
        "message": f"Successfully browsed '{title or url}'. Extracted {len(clean_text)} characters.",
    }
