"""Drive the web UI end to end in headless Chromium and save screenshots.

    python tests/e2e_ui.py OUT_DIR [difficulty]   (server must be running on :8000, or set RC_URL)
    RC_MODE=360 python tests/e2e_ui.py OUT_DIR veteran   (the 360-degree theater)
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
url = os.environ.get("RC_URL", "http://localhost:8000/")
errors = []

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/chromium-browser",
                          args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--no-sandbox"])
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(url)
    pg.wait_for_selector("#dlg-intro[open]")
    pg.click("#dlg-intro button")
    pg.wait_for_timeout(800)
    if os.environ.get("RC_MODE") == "360":
        pg.click('#seg-mode button[data-mode="360"]')
        pg.wait_for_function("window.__rc.S.mode === '360' && window.__rc.S.map")
        pg.wait_for_timeout(1500)

    def click_cell(c):
        xy = pg.evaluate("""(c) => { const {scene} = window.__rc; const v = scene.cellWorld(c).project(scene.camera);
            const r = scene.renderer.domElement.getBoundingClientRect();
            return [r.left + (v.x*0.5+0.5)*r.width, r.top + (-v.y*0.5+0.5)*r.height]; }""", c)
        pg.mouse.move(*xy)
        pg.wait_for_timeout(150)
        pg.mouse.click(*xy)
        pg.wait_for_timeout(250)

    base = pg.evaluate("window.__rc.S.map.base_cell")
    G = 32
    bx, by = base % G, base // G
    cell = lambda dx, dy: (by + dy) * G + bx + dx
    pg.keyboard.press("1"); click_cell(cell(0, -5)); click_cell(cell(-7, -4))
    pg.keyboard.press("3"); click_cell(cell(2, -2)); click_cell(cell(-2, -3))
    pg.keyboard.press("2"); click_cell(cell(5, -6))
    pg.keyboard.press("1")
    pg.mouse.move(700, 300)
    pg.wait_for_timeout(600)
    print("layout", pg.evaluate("JSON.stringify(window.__rc.S.layout)"))
    pg.screenshot(path=out / "1_defend.png")

    if len(sys.argv) > 2:
        pg.click(f"#seg-diff button[data-key={sys.argv[2]}]")
    pg.click("#btn-launch")
    pg.wait_for_selector("#playback:not([hidden])", timeout=60000)
    pg.wait_for_function("window.__rc.S.replay && window.__rc.S.replay.t > 12", timeout=60000)
    pg.screenshot(path=out / "2_attack.png")
    if pg.evaluate("!!window.__rc.S.replay.res.rep.thinking"):   # learned enemy: open the neural network view
        pg.keyboard.press("n")
        pg.wait_for_function("window.__rc.S.net.data !== null && window.__rc.S.net.drawn !== ''", timeout=90000)
        pg.wait_for_timeout(1500)
        print("network view:", pg.text_content("#net-why"))
        pg.screenshot(path=out / "2b_network.png")
        pg.keyboard.press("n")
        pg.wait_for_timeout(500)
    pg.click("#btn-skip")
    pg.wait_for_selector("#panel-right:not([hidden])")
    pg.wait_for_timeout(600)
    pg.screenshot(path=out / "3_debrief.png")
    pg.check("#toggles input[data-ov=astar]")
    pg.uncheck("#toggles input[data-ov=breach]")
    pg.wait_for_function("window.__rc.scene.pathGroup.children.length > 2", timeout=60000)
    pg.wait_for_timeout(500)
    pg.screenshot(path=out / "4_astar.png")
    if pg.evaluate("(window.__rc.S.cfg.timeline || []).length > 1"):
        pg.click("#btn-learn")
        pg.wait_for_selector("body[data-learning]", timeout=120000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=out / "6_timelapse.png")
        pg.wait_for_selector("#panel-right:not([hidden])", timeout=240000)
        pg.wait_for_timeout(1200)
        print("timelapse stages:", pg.evaluate("window.__rc.S.learn.stages.map(r => r.breach_rate)"))
        pg.screenshot(path=out / "7_stages.png")
    pg.click("#btn-hood")
    pg.wait_for_timeout(1200)
    pg.screenshot(path=out / "5_hood.png")
    b.close()

print("console errors:", errors or "none")
