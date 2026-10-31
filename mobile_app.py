"""手機版的 APP 式操作：底部分頁列、「更多」選單、目前標的迷你列、全螢幕自選股。
只在寬度 768px 以下生效，桌面版的畫面與導覽完全不變。"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

PHONE_QUERY = "(max-width:768px)"

CSS = """<style>
#m-tabbar,#m-stock,#m-sheet,#m-sheet-bg,.m-subnav{display:none}
@media(max-width:768px){
  html,body{overscroll-behavior:none}
  .stApp{-webkit-tap-highlight-color:transparent;-webkit-text-size-adjust:100%}
  .stApp button,.stApp a,.stApp [role="button"],.stApp [role="tab"]{touch-action:manipulation}
  /* iOS 在字級小於 16px 的輸入框會自動放大整頁 */
  .stApp input,.stApp textarea{font-size:16px!important}

  /* 頁首縮成一列，第一屏留給內容 */
  .stApp .brand-block{display:flex!important;align-items:center;justify-content:flex-start;gap:8px;padding:8px 0 2px!important;margin:0!important;text-align:left}
  .stApp .brand-block .brand-mark{width:28px!important;height:28px!important;margin:0!important;flex:none}
  .stApp .brand-block .brand-mark svg{width:28px!important;height:28px!important}
  .stApp .brand-block .brand-name{font-size:17px!important;letter-spacing:.04em!important;margin:0!important;white-space:nowrap}
  .stApp .st-key-auth_header_actions{position:fixed!important;left:-10000px!important;top:0!important;width:1px!important;height:1px!important;overflow:hidden!important;opacity:0!important}
  .stApp .page-image-banner,.stApp .site-rule,#ux-navtab,#sb-toggle{display:none!important}
  .stApp .content-heading{margin:-22px 0 8px!important;padding:0!important}
  .stApp .content-heading .eyebrow,.stApp .content-heading p{display:none!important}
  .stApp .content-heading h1{font-size:22px!important;margin:0!important}
  .stApp .hero-carousel{height:170px!important}
  .stApp .st-key-auth_drawer [data-testid="stHorizontalBlock"]:first-of-type{flex-wrap:nowrap!important}
  .stApp .st-key-auth_drawer [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stColumn"]:last-child{flex:0 0 48px!important;min-width:48px!important;width:48px!important}
  .stApp .tp-table{display:block!important;max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch}
  .stApp [data-testid="stMain"] .block-container{padding-top:0!important}
  .stApp .st-key-stock_search_box [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;gap:0!important}
  .stApp .st-key-stock_search_box [data-testid="stColumn"]:first-child{flex:1 1 auto!important;min-width:0!important;width:auto!important}
  .stApp .st-key-stock_search_box [data-testid="stColumn"]:last-child{flex:0 0 56px!important;min-width:56px!important;width:56px!important}
  .stApp .fin-strip{grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:6px!important}
  .stApp .fin-tile{padding:8px 9px!important;min-width:0}
  .stApp .fin-tile b{font-size:15px!important;white-space:nowrap}
  .stApp .fin-tile small{font-size:10.5px!important}

  /* 上方導覽換成本頁功能捷徑，主題切換交給底部分頁列 */
  .stApp .st-key-site_nav_sticky{padding:2px 0 6px!important}
  .stApp .st-key-site_nav_sticky .site-nav{display:none!important}
  .stApp .m-subnav{display:flex;gap:7px;overflow-x:auto;padding:5px 2px;scrollbar-width:none;-webkit-overflow-scrolling:touch}
  .stApp .m-subnav::-webkit-scrollbar{display:none}
  .stApp .m-subnav a{flex:none;display:flex;align-items:center;min-height:34px;padding:0 13px;border:1px solid #e6d3d3;border-radius:999px;background:#fff;
    color:#6b4d4d!important;font-size:13px;font-weight:600;white-space:nowrap;text-decoration:none!important}
  .stApp .m-subnav a:active{background:#8e2b2f;border-color:#8e2b2f;color:#fff!important}

  .stApp [data-testid="stMain"]{overscroll-behavior:contain}
  .stApp [data-testid="stMain"] .block-container{padding-bottom:calc(84px + env(safe-area-inset-bottom))!important}
  body.m-has-stock .stApp [data-testid="stMain"] .block-container{padding-bottom:calc(140px + env(safe-area-inset-bottom))!important}
  .stApp .site-footer{margin-bottom:0!important}
  #ux-top{bottom:calc(72px + env(safe-area-inset-bottom))!important;right:12px!important}
  body.m-has-stock #ux-top{bottom:calc(128px + env(safe-area-inset-bottom))!important}
  #ux-toast{bottom:calc(84px + env(safe-area-inset-bottom))!important}

  /* 手指放在圖表上也能捲動頁面；工具列在手機上點不準 */
  .stApp .js-plotly-plot .modebar-container{display:none!important}
  .stApp .js-plotly-plot,.stApp .js-plotly-plot *{touch-action:manipulation!important}

  /* 自選股側欄改成全螢幕頁面 */
  .stApp [data-testid="stSidebar"]:is([aria-expanded="true"],[data-collapsed="false"]){width:100vw!important;min-width:100vw!important;max-width:100vw!important}
  .stApp [data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-bottom:calc(84px + env(safe-area-inset-bottom))!important;overscroll-behavior:contain}
  .stApp [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button{min-width:44px;min-height:44px}

  #m-tabbar{display:flex;position:fixed;left:0;right:0;bottom:0;z-index:999995;padding-bottom:env(safe-area-inset-bottom);background:#fff;
    border-top:1px solid #eadcdc;box-shadow:0 -4px 14px -8px rgba(90,40,40,.3)}
  #m-tabbar .m-tab{flex:1;min-width:0;height:58px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;padding:0;border:0;background:none;
    color:#8a7a7a!important;font:600 11px/1.1 inherit;font-family:inherit;text-decoration:none!important;cursor:pointer;-webkit-tap-highlight-color:transparent}
  #m-tabbar .m-tab svg{width:23px;height:23px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
  #m-tabbar .m-tab.on{color:#8e2b2f!important;font-weight:800}
  #m-tabbar .m-tab:active{background:#fbf1f1}

  body.m-has-stock #m-stock{display:flex}
  #m-stock{position:fixed;left:8px;right:8px;bottom:calc(64px + env(safe-area-inset-bottom));z-index:98990;align-items:center;gap:10px;min-height:48px;padding:6px 12px;
    border:1px solid #e6d3d3;border-radius:12px;background:rgba(255,255,255,.97);box-shadow:0 6px 16px -8px rgba(90,40,40,.45);font-family:inherit;color:#34404a;text-align:left;cursor:pointer}
  #m-stock .n{flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-size:14px;font-weight:800}
  #m-stock .n small{margin-left:4px;font-size:12px;font-weight:600;color:#8a7a7a}
  #m-stock .v{flex:none;font-variant-numeric:tabular-nums;font-size:13px;font-weight:700}
  #m-stock .v b{margin-right:7px;font-size:16px;font-weight:900}
  #m-stock:active{background:#fbf1f1}

  #m-sheet-bg{position:fixed;inset:0;z-index:999996;background:rgba(30,15,15,.42);opacity:0;pointer-events:none;transition:opacity .2s ease;display:block}
  #m-sheet{position:fixed;left:0;right:0;bottom:0;z-index:999997;display:block;padding:8px 16px calc(18px + env(safe-area-inset-bottom));background:#fff;border-radius:18px 18px 0 0;
    transform:translateY(105%);transition:transform .24s cubic-bezier(.2,.7,.2,1);visibility:hidden}
  body.m-sheet-open #m-sheet-bg{opacity:1;pointer-events:auto}
  body.m-sheet-open #m-sheet{transform:none;visibility:visible}
  #m-sheet .grip{width:38px;height:4px;margin:0 auto 14px;border-radius:9px;background:#e0d2d2}
  #m-sheet h4{margin:0 0 8px;font-size:13px;font-weight:700;color:#8a7a7a}
  #m-sheet .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-bottom:16px}
  #m-sheet .item{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;min-height:74px;border-radius:12px;color:#34404a!important;font-size:12.5px;font-weight:700;text-decoration:none!important}
  #m-sheet .item svg{width:25px;height:25px;fill:none;stroke:#8e2b2f;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
  #m-sheet .item.on{background:#fbf1f1;color:#8e2b2f!important}
  #m-sheet .item:active{background:#fbf1f1}
  #m-sheet .acct{display:flex;gap:8px}
  #m-sheet .acct button{flex:1;min-height:46px;border:1px solid #d9c2c2;border-radius:11px;background:#fff;color:#8e2b2f;font:700 15px inherit;font-family:inherit;cursor:pointer}
  #m-sheet .acct button.pri{background:#8e2b2f;border-color:#8e2b2f;color:#fff}
}
</style>"""

_RUNTIME = """<script>
(function() {
  let w, d;
  try { w = window.parent; d = w.document; } catch (e) { return; }
  if (w.__mobileAppV1) return;
  w.__mobileAppV1 = true;
  const MORE = __MORE__, ICONS = __ICONS__;
  const phone = w.matchMedia("__PHONE__");
  const svg = (paths) => '<svg viewBox="0 0 24 24">' + paths + "</svg>";

  // 加到主畫面時用的名稱與顏色；Streamlit Cloud 的網站包在外層頁面裡，兩層都要放
  const addMeta = (doc) => {
    [["theme-color", "#ffffff"], ["apple-mobile-web-app-capable", "yes"], ["mobile-web-app-capable", "yes"],
     ["apple-mobile-web-app-title", "台股分析"], ["apple-mobile-web-app-status-bar-style", "default"]].forEach(function(kv) {
      if (doc.querySelector('meta[name="' + kv[0] + '"]')) return;
      const m = doc.createElement("meta"); m.name = kv[0]; m.content = kv[1]; doc.head.appendChild(m);
    });
  };
  addMeta(d);
  try { if (w.parent !== w) addMeta(w.parent.document); } catch (e) {}

  const tabbar = d.createElement("nav");
  tabbar.id = "m-tabbar"; tabbar.setAttribute("translate", "no");
  const link = (key, label, icon) => '<a class="m-tab" data-s="' + key + '" href="?section=' + key + '" target="_self">' + svg(icon) + label + "</a>";
  tabbar.innerHTML = link("home", "首頁", ICONS.home) + link("finance", "財經", ICONS.finance)
    + '<button class="m-tab" id="m-tab-watch" type="button">' + svg('<path d="M12 3.5l2.6 5.4 5.9.8-4.3 4.1 1 5.9-5.2-2.8-5.2 2.8 1-5.9-4.3-4.1 5.9-.8z"/>') + "自選</button>"
    + link("strategy", "個股", ICONS.strategy)
    + '<button class="m-tab" id="m-tab-more" type="button">' + svg('<path d="M4 7h16M4 12h16M4 17h16"/>') + "更多</button>";
  d.body.appendChild(tabbar);

  const stockBar = d.createElement("button");
  stockBar.id = "m-stock"; stockBar.type = "button"; stockBar.setAttribute("translate", "no");
  stockBar.innerHTML = '<span class="n"></span><span class="v"></span>';
  d.body.appendChild(stockBar);

  const sheetBg = d.createElement("div"); sheetBg.id = "m-sheet-bg";
  const sheet = d.createElement("div"); sheet.id = "m-sheet"; sheet.setAttribute("translate", "no");
  sheet.innerHTML = '<div class="grip"></div><h4>更多功能</h4><div class="grid">'
    + MORE.map(function(m) { return '<a class="item" data-s="' + m[0] + '" href="?section=' + m[0] + '" target="_self">' + svg(ICONS[m[0]]) + m[1] + "</a>"; }).join("")
    + '</div><h4>帳號</h4><div class="acct"></div>';
  d.body.appendChild(sheetBg); d.body.appendChild(sheet);

  const closeSheet = () => d.body.classList.remove("m-sheet-open");
  const pressHidden = (key) => { const b = d.querySelector(".st-key-" + key + " button"); if (b) b.click(); };
  const fillAccount = () => {
    const box = sheet.querySelector(".acct");
    const items = d.querySelector(".st-key-auth_top_logout")
      ? [["auth_top_logout", "登出", ""]]
      : [["auth_top_login", "登入", "pri"], ["auth_top_request", "申請帳號", ""]];
    box.innerHTML = items.map(function(x) { return '<button type="button" class="' + x[2] + '" data-press="' + x[0] + '">' + x[1] + "</button>"; }).join("");
  };
  sheet.addEventListener("click", function(event) {
    const press = event.target.closest("[data-press]");
    if (press) { closeSheet(); pressHidden(press.getAttribute("data-press")); return; }
    if (event.target.closest("a")) closeSheet();
  });
  sheetBg.addEventListener("click", closeSheet);

  const sidebar = () => d.querySelector('[data-testid="stSidebar"]');
  const sidebarOpen = () => { const s = sidebar(); return !!s && (s.getAttribute("aria-expanded") === "true" || s.getAttribute("data-collapsed") === "false"); };
  const setSidebar = (open) => { if (w.__sbApply) w.__sbApply(open); };
  const currentSection = () => {
    const a = d.querySelector(".site-nav-link.active");
    const m = a && (a.getAttribute("href") || "").match(/section=(\\w+)/);
    return m ? m[1] : "";
  };
  const scrollMainTop = () => { const m = d.querySelector('[data-testid="stMain"]'); if (m) m.scrollTo({top: 0, behavior: "smooth"}); };

  let wantWatch = 0;   // 從別頁按「自選」：等操作策略頁載入後才打開
  let sidebarSeenAt = 0, userOpened = false;
  const openWatch = () => {
    closeSheet();
    userOpened = true;
    if (d.querySelector('.st-key-page_strategy') && sidebar()) { setSidebar(!sidebarOpen()); return; }
    wantWatch = Date.now();
    const go = tabbar.querySelector('[data-s="strategy"]');
    if (go) go.click();
  };
  d.getElementById("m-tab-watch").addEventListener("click", openWatch);
  stockBar.addEventListener("click", openWatch);
  d.getElementById("m-tab-more").addEventListener("click", function() {
    if (d.body.classList.contains("m-sheet-open")) return closeSheet();
    fillAccount(); d.body.classList.add("m-sheet-open");
  });

  // 比網站導覽的攔截更早執行：按目前所在的分頁只回到頂端，不重新載入；換分頁前先收起全螢幕自選股
  w.addEventListener("click", function(event) {
    const t = event.target;
    if (!phone.matches || !t || !t.closest) return;
    const tab = t.closest("#m-tabbar a, #m-sheet a");
    if (tab) {
      const same = tab.getAttribute("data-s") === currentSection();
      if (sidebarOpen() && !wantWatch) setSidebar(false);
      if (same) {
        event.preventDefault(); event.stopPropagation(); closeSheet(); scrollMainTop();
      }
      return;
    }
    // 在自選股選了一檔：收起名單、回到該股分析的最上方
    if (t.closest('[data-testid="stSidebar"] [class*="st-key-watch_select_"] button')) {
      setTimeout(function() { setSidebar(false); scrollMainTop(); }, 60);
    }
  }, true);

  // 圖表會攔下手指、讓頁面捲不動：不把 touchstart 交給 Plotly，由瀏覽器捲動；點一下仍會以滑鼠事件顯示數值
  d.addEventListener("touchstart", function(event) {
    const t = event.target;
    if (phone.matches && t && t.closest && t.closest(".js-plotly-plot")) event.stopPropagation();
  }, {capture: true, passive: true});

  let lastStock = "";
  w.setInterval(function() {
    if (!phone.matches) { d.body.classList.remove("m-has-stock"); return; }
    const section = currentSection(), open = sidebarOpen();
    // Streamlit 會在瀏覽器記住側欄上次的開關，側欄一出現就可能是打開的：除非是使用者自己開的，一律先收起
    if (!sidebar()) {
      sidebarSeenAt = 0;
      if (!wantWatch) userOpened = false;
    } else {
      sidebarSeenAt = sidebarSeenAt || Date.now();
      if (open && !wantWatch && !userOpened && Date.now() - sidebarSeenAt < 3000 && !d.querySelector(".st-key-auth_drawer")) setSidebar(false);
    }
    tabbar.querySelectorAll("a.m-tab").forEach(function(a) { a.classList.toggle("on", !open && a.getAttribute("data-s") === section); });
    d.getElementById("m-tab-watch").classList.toggle("on", open);
    d.getElementById("m-tab-more").classList.toggle("on", !open && MORE.some(function(m) { return m[0] === section; }));
    sheet.querySelectorAll("a.item").forEach(function(a) { a.classList.toggle("on", a.getAttribute("data-s") === section); });

    if (wantWatch) {
      if (Date.now() - wantWatch > 30000) wantWatch = 0;
      else if (d.querySelector(".st-key-page_strategy") && sidebar() && d.querySelector(".pin-card")) { wantWatch = 0; setSidebar(true); }
    }

    // 迷你列的內容直接取自側欄卡片，側欄定時更新時跟著變
    const pin = section === "strategy" && !open ? d.querySelector(".pin-card") : null;
    d.body.classList.toggle("m-has-stock", !!pin);
    if (pin) {
      const name = pin.querySelector(".pin-name"), vals = pin.querySelector(".pin-vals");
      if (name && vals) {
        const key = name.textContent + "|" + vals.textContent + "|" + vals.style.color;
        if (key !== lastStock) {
          lastStock = key;
          const code = name.querySelector("span");
          const nameEl = stockBar.querySelector(".n"), valEl = stockBar.querySelector(".v");
          nameEl.textContent = name.firstChild ? name.firstChild.textContent : "";
          if (code) { const s = d.createElement("small"); s.textContent = code.textContent.replace(/[（）]/g, ""); nameEl.appendChild(s); }
          const spans = vals.querySelectorAll("span"), price = vals.querySelector("b");
          valEl.style.color = vals.style.color;
          valEl.innerHTML = "";
          if (price) { const b = d.createElement("b"); b.textContent = price.textContent; valEl.appendChild(b); }
          if (spans[1]) valEl.appendChild(d.createTextNode(spans[1].textContent));
        }
      }
    }
  }, 300);
})();
</script>"""


def inject(icons: dict[str, str], more: list[tuple[str, str]]) -> None:
    """more：放進「更多」選單的 (section, 名稱)。"""
    st.markdown(CSS, unsafe_allow_html=True)
    script = (_RUNTIME.replace("__MORE__", json.dumps(more, ensure_ascii=False))
              .replace("__ICONS__", json.dumps(icons, ensure_ascii=False))
              .replace("__PHONE__", PHONE_QUERY))
    components.html(script, height=0)
