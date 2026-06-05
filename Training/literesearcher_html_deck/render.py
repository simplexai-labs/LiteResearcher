"""
Render html_deck/deck.html to:
  1. per-slide PNG previews   → preview/slide-NN.png
  2. a single PDF             → LiteResearcher.pdf

Each <section class="slide"> is rendered at exactly 1280x720 px.
"""
import asyncio, sys, pathlib, re
from playwright.async_api import async_playwright

HERE = pathlib.Path(__file__).parent.resolve()
HTML = HERE / "deck.html"
PREV = HERE / "preview"
PDF  = HERE / "LiteResearcher.pdf"

PREV.mkdir(exist_ok=True)
for f in PREV.glob("*.png"):
    f.unlink()


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            device_scale_factor=2,           # 2x for crisp screenshots
        )
        page = await ctx.new_page()
        await page.goto(HTML.as_uri(), wait_until="networkidle")

        # discover slides
        slide_ids = await page.eval_on_selector_all(
            "section.slide", "els => els.map(e => e.id)")
        print(f"found {len(slide_ids)} slides: {slide_ids[:5]}{'...' if len(slide_ids)>5 else ''}")

        # screenshot each slide as a clipped PNG (1280x720)
        for i, sid in enumerate(slide_ids, 1):
            el = await page.query_selector(f"#{sid}")
            await el.screenshot(path=str(PREV / f"slide-{i:02d}.png"))
        print(f"wrote {len(slide_ids)} PNGs to {PREV}")

        # Render a single multi-page PDF.
        # Make every slide print on its own 1280x720 page.
        await page.emulate_media(media="print")
        await page.pdf(
            path=str(PDF),
            width="1280px",
            height="720px",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            prefer_css_page_size=False,
        )
        print(f"wrote PDF → {PDF}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
