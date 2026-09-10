#!/usr/bin/env python3
"""无头 Chromium + Perfetto UI 截图.

用 Perfetto 官方的 postMessage 接口而不是 ?url= 深链: 后者要 Perfetto 自己去
fetch, https 页面取 http://127.0.0.1 会被混合内容策略挡掉.

postMessage 方式下 trace 由本地页面读成 ArrayBuffer 再传过去, **完全不出这台机器**,
只有 Perfetto 的前端代码是从公网加载的.
"""
import functools, http.server, socketserver, sys, threading, time
from pathlib import Path

TRACE_DIR = Path(sys.argv[1]).resolve()
NAME, OUT = sys.argv[2], Path(sys.argv[3]).resolve()
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 8733

(TRACE_DIR / "_open.html").write_text("""<!doctype html><meta charset=utf-8><body>
<script>
const f = new URLSearchParams(location.search).get('f');
const ORIGIN = 'https://ui.perfetto.dev';
fetch('./' + f).then(r => r.arrayBuffer()).then(buf => {
  const win = window.open(ORIGIN);
  const timer = setInterval(() => win.postMessage('PING', ORIGIN), 50);
  window.addEventListener('message', e => {
    if (e.data !== 'PONG') return;
    clearInterval(timer);
    win.postMessage({perfetto: {buffer: buf, title: f}}, ORIGIN);
  });
});
</script></body>""", encoding="utf-8")

class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
socketserver.TCPServer.allow_reuse_address = True
srv = socketserver.TCPServer(("127.0.0.1", PORT),
                             functools.partial(H, directory=str(TRACE_DIR)))
threading.Thread(target=srv.serve_forever, daemon=True).start()

from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    ctx = b.new_context(viewport={"width": 1800, "height": 1000}, device_scale_factor=2)
    pg = ctx.new_page()
    with pg.expect_popup(timeout=90000) as info:
        pg.goto(f"http://127.0.0.1:{PORT}/_open.html?f={NAME}", timeout=60000)
    ui = info.value
    ui.wait_for_load_state("networkidle", timeout=90000)
    for _ in range(60):
        time.sleep(1)
        t = ui.inner_text("body")
        if "Open trace file" not in t and len(t) > 30:
            break
    time.sleep(4)

    def click_text(txt, exact=True):
        try:
            ui.get_by_text(txt, exact=exact).first.click(timeout=2500)
            time.sleep(0.4); return True
        except Exception:
            return False

    # 1) 关掉 cookie 横幅（挡住左下角）
    click_text("OK")
    # 2) 关掉底部的 Trace Doctor 面板（占了半屏）：点它标签右边的 x
    try:
        tab = ui.get_by_text("Trace Doctor", exact=True).first
        box = tab.bounding_box()
        if box:
            ui.mouse.click(box["x"] + box["width"] + 12, box["y"] + box["height"] / 2)
            time.sleep(0.5)
    except Exception:
        pass
    # 3) 收起左侧边栏，把画布让出来（左上角的汉堡按钮）
    try:
        ui.mouse.click(201, 21); time.sleep(0.6)
    except Exception:
        pass
    # 4) 展开所有轨道分组：点开每个 group 的标题
    try:
        groups = ui.locator('.pf-track-shell__title, .pf-track__title')
        n = min(groups.count(), 12)
        for i in range(n):
            try: groups.nth(i).click(timeout=1000); time.sleep(0.2)
            except Exception: pass
    except Exception:
        pass
    # 5) 全览：Perfetto 的 ctrl 组合不稳，用键盘 'f' 之前先选中画布
    try:
        ui.mouse.click(1200, 400); time.sleep(0.3)
        for _ in range(3):
            ui.keyboard.press("w"); time.sleep(0.2)   # 放大到能看清 kernel
    except Exception:
        pass
    time.sleep(2)

    # 裁到内容区: 全屏截图下面有一大片空白画布, 放进 blog 很难看.
    # .pf-track 是每条轨道, 取它们最下沿 + 一点边距就是时间轴的真实高度.
    # 上边从 44 起, 去掉那条"Search or type '>'"的搜索栏(对读者没信息量),
    # 但保留时间标尺 -- 那是让读者一眼认出"这是真的 profiler 截图"的东西.
    box = ui.evaluate('''() => {
      const els = document.querySelectorAll('.pf-track');
      if (!els.length) return null;
      let b = 0;
      for (const e of els) b = Math.max(b, e.getBoundingClientRect().bottom);
      return Math.round(b);
    }''')
    vp = ui.viewport_size
    clip = None
    if box:
        top = 44
        clip = {"x": 0, "y": top, "width": vp["width"],
                "height": min(vp["height"] - top, box - top + 16)}
    ui.screenshot(path=str(OUT), clip=clip)
    print(f"截图 -> {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print("页面文本:", ui.inner_text("body")[:180].replace("\n", " | "))
    b.close()
srv.shutdown()
(TRACE_DIR / "_open.html").unlink(missing_ok=True)
