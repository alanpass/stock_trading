# -*- coding: utf-8 -*-
"""Fugle 法說會備忘錄 AI Agent v61

Primary source: https://blog.fugle.tw/topic/earnings-call-memo

每篇文章都必須由 Ollama Agent 呼叫 read_fugle_memo(url) tool，tool 才會開啟
Fugle 詳細文章頁並回傳正文。MOPS/TWSE/TPEx 不會混入 memo 摘要；它們僅能在
其他模組中作為獨立官方事件資料。
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, quote_plus

import requests

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

TOPIC_URL = "https://blog.fugle.tw/topic/earnings-call-memo"
HOME_URL = "https://blog.fugle.tw/"
POST_PREFIX = "https://blog.fugle.tw/post/earnings-call-"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

POSITIVE_WORDS = [
    "營收成長", "營收增加", "獲利成長", "獲利改善", "毛利率提升", "毛利率改善",
    "需求強勁", "需求回溫", "訂單增加", "在手訂單", "訂單滿載", "產能滿載",
    "稼動率提升", "漲價", "價格上漲", "報價上調", "ASP上升", "新客戶", "新產品",
    "量產", "放量", "擴產", "創高", "創歷史新高", "展望樂觀", "正向",
]
NEGATIVE_WORDS = [
    "營收下滑", "營收減少", "獲利下滑", "獲利衰退", "毛利率下降", "毛利率下滑",
    "需求疲弱", "需求下降", "需求放緩", "訂單減少", "砍單", "取消訂單", "庫存調整",
    "庫存過高", "稼動率下降", "降價", "價格下跌", "報價下調", "ASP下降", "成本上升",
    "原料上漲", "匯兌損失", "下修", "資本支出下修", "放緩", "虧損", "風險",
]

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_fugle_memo",
        "description": "開啟指定的富果法說會備忘錄詳細文章，讀取正文；不能用網址或搜尋摘要代替。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "https://blog.fugle.tw/post/earnings-call-* 文章網址",
                }
            },
            "required": ["url"],
        },
    },
}


def clean(v: Any) -> str:
    s = html.unescape(str(v or "")).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_date(v: Any) -> str:
    s = str(v or "")
    for p in (r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})", r"(20\d{2})(\d{2})(\d{2})"):
        m = re.search(p, s)
        if m:
            try:
                return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            except Exception:
                pass
    return ""


def symbol_from_url(url: str) -> str:
    m = re.search(r"earnings-call-([0-9A-Za-z]+)(?:-20\d{2}-\d{2}-\d{2})?$", url.rstrip("/"))
    return m.group(1).upper() if m else ""


class FugleMemoCrawler:
    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir).resolve()
        self.root = self.base / "data" / "research" / "fugle_memo_agent"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "earnings_memo.db"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": TOPIC_URL,
            "Upgrade-Insecure-Requests": "1",
        })
        self._driver = None
        self._init_db()

    def _looks_like_block_or_error(self, raw: str) -> bool:
        t = clean(raw)[:12000].lower()
        bad = (
            "cache miss", "internal server error", "bad gateway",
            "service unavailable", "too many requests", "403 forbidden",
            "404 not found", "cloudflare", "access denied",
            "文章不存在", "找不到文章", "網站發生錯誤", "請重新整理",
        )
        return len(t) < 700 or any(x in t for x in bad)

    def _fetch_requests(self, url: str, timeout: int = 35):
        r = self.session.get(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        if self._looks_like_block_or_error(r.text):
            raise RuntimeError("requests 取得的是錯誤/阻擋頁")
        return r.text, r.url, "requests"

    def _fetch_curl_cffi(self, url: str, timeout: int = 35):
        try:
            from curl_cffi import requests as c_requests
        except Exception as exc:
            raise RuntimeError(f"curl_cffi 未安裝：{exc}")
        r = c_requests.get(
            url,
            impersonate="chrome",
            timeout=timeout,
            allow_redirects=True,
            headers={"Referer": TOPIC_URL, "Accept-Language": "zh-TW,zh;q=0.9"},
        )
        r.raise_for_status()
        if self._looks_like_block_or_error(r.text):
            raise RuntimeError("curl_cffi 取得的是錯誤/阻擋頁")
        return r.text, str(r.url), "curl_cffi"

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1440,2400")
        options.add_argument("--lang=zh-TW")
        options.page_load_strategy = "eager"
        self._driver = webdriver.Chrome(options=options)
        self._driver.set_page_load_timeout(45)
        return self._driver

    def _fetch_browser(self, url: str, timeout: int = 45, load_more: bool = False):
        """使用 Chrome DOM；抓取文章頁時不把網站導覽當成正文否決條件。"""
        driver=self._get_driver()
        driver.get(url)
        time.sleep(2.0)
        if load_more:
            for _ in range(12):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.8)
                clicked=driver.execute_script("""
                    const els=[...document.querySelectorAll('button,a')];
                    const el=els.find(e=>/載入更多|load more|顯示更多/i.test((e.innerText||'').trim()));
                    if(el){el.click(); return true;} return false;
                """)
                if not clicked: break
                time.sleep(1.0)
        else:
            for frac in (0.35,0.75,1.0):
                driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight*{frac});")
                time.sleep(0.8)
        raw=driver.page_source
        body=clean(driver.execute_script("return document.body.innerText || ''") or '')
        if len(body)<700:
            raise RuntimeError(f"Chrome body 太短：{len(body)} chars")
        self._last_rendered_text=body
        return raw,driver.current_url,"selenium"

    def close(self):
        if self._driver is not None:
            try: self._driver.quit()
            except Exception: pass
            self._driver = None

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        con.execute("""CREATE TABLE IF NOT EXISTS memos(
            url TEXT PRIMARY KEY, symbol TEXT, title TEXT, published_date TEXT, modified_date TEXT,
            content_hash TEXT, last_seen TEXT, last_analyzed TEXT, impact TEXT, confidence REAL,
            analysis_json TEXT, memo_text TEXT
        )""")
        con.commit(); con.close()

    def fetch(self, url: str, timeout: int = 35, allow_browser: bool = True, load_more: bool = False) -> tuple[str, str, str]:
        errors=[]
        for fn in (
            lambda: self._fetch_requests(url, timeout),
            lambda: self._fetch_curl_cffi(url, timeout),
        ):
            try:
                return fn()
            except Exception as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
        if allow_browser and os.getenv("FUGLE_MEMO_USE_BROWSER", "true").lower() in {"1","true","yes","on"}:
            try:
                return self._fetch_browser(url, min(timeout+10,60), load_more=load_more)
            except Exception as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
        raise RuntimeError("；".join(errors[-3:]))

    def discover(self) -> list[dict[str, str]]:
        """Primary discovery from the requested topic page, plus homepage and a URL-only search fallback."""
        found: dict[str, dict[str, str]] = {}
        for page in (TOPIC_URL, HOME_URL):
            try:
                raw, final, reader = self.fetch(page, allow_browser=True, load_more=(page == TOPIC_URL))
            except Exception:
                continue
            urls = set(re.findall(r"https?://blog\.fugle\.tw/post/earnings-call-[^\"'<>\s&]+", raw))
            if BeautifulSoup is not None:
                soup = BeautifulSoup(raw, "html.parser")
                for a in soup.find_all("a", href=True):
                    u = urljoin(final, a.get("href"))
                    if POST_PREFIX in u:
                        urls.add(u.split("?")[0])
                    txt = clean(a.get_text(" ", strip=True))
                    if POST_PREFIX in u:
                        found[u.split("?")[0]] = {"url":u.split("?")[0], "title":txt, "symbol":symbol_from_url(u), "published_date":parse_date(u)}
            for u in urls:
                u = html.unescape(u).rstrip("\")'.,)")
                if u.startswith(POST_PREFIX):
                    found.setdefault(u, {"url":u,"title":"","symbol":symbol_from_url(u),"published_date":parse_date(u)})
        rows = list(found.values())
        # No external search engine is used for the content itself; only if the topic page lacks a company URL.
        if not rows and os.getenv("FUGLE_MEMO_ALLOW_SEARCH_FALLBACK", "true").lower() in {"1","true","yes","on"}:
            q = quote_plus("site:blog.fugle.tw/post/earnings-call-")
            try:
                raw, _, _ = self.fetch(f"https://www.google.com/search?q={q}", timeout=15, allow_browser=False)
                for u in re.findall(r"https?://blog\.fugle\.tw/post/earnings-call-[^\"'<>\s&]+", raw):
                    u = html.unescape(u).rstrip("\")'.,)")
                    if u.startswith(POST_PREFIX): found.setdefault(u, {"url":u,"title":"","symbol":symbol_from_url(u),"published_date":parse_date(u)})
            except Exception:
                pass
            rows = list(found.values())
        rows.sort(key=lambda x:(x.get("published_date", ""), x.get("url", "")), reverse=True)
        return rows

    def _extract_article_text(self, raw: str, rendered: str = ""):
        """隔離 Fugle 真正文章；優先 JSON-LD/正文 container，移除 nav/footer。"""
        title=published=modified=""
        candidates=[]
        if rendered: candidates.append(("selenium-body",clean(rendered)))
        if BeautifulSoup is None:
            txt=clean(re.sub(r"<script.*?</script>|<style.*?</style>|<nav.*?</nav>|<header.*?</header>|<footer.*?</footer>"," ",raw,flags=re.I|re.S))
            return clean(re.sub(r"<[^>]+>"," ",txt)),title,published,modified,"raw"
        soup=BeautifulSoup(raw,"html.parser")
        h1=soup.find("h1")
        title=clean(h1.get_text(" ",strip=True) if h1 else "")
        for k in ("article:published_time","datePublished"):
            tag=soup.find("meta",attrs={"property":k}) or soup.find("meta",attrs={"name":k})
            if tag and tag.get("content"): published=parse_date(tag.get("content")) or published
        for k in ("article:modified_time","dateModified"):
            tag=soup.find("meta",attrs={"property":k}) or soup.find("meta",attrs={"name":k})
            if tag and tag.get("content"): modified=parse_date(tag.get("content")) or modified
        # JSON-LD articleBody
        for script in soup.find_all("script",type="application/ld+json"):
            try:
                data=json.loads(script.string or script.get_text())
                objs=data if isinstance(data,list) else [data]
                for obj in objs:
                    if isinstance(obj,dict):
                        title=clean(obj.get("headline")) or title
                        published=parse_date(obj.get("datePublished")) or published
                        modified=parse_date(obj.get("dateModified")) or modified
                        ab=clean(obj.get("articleBody"))
                        if len(ab)>=700: candidates.append(("jsonld",ab))
            except Exception: pass
        anchors=[]
        if h1: anchors.append(h1)
        for el in soup.find_all(string=re.compile(r"法說會備忘錄|營運摘要|財務表現")):
            if el.parent: anchors.append(el.parent)
        seen=set()
        for a in anchors:
            node=a
            for _ in range(8):
                if not node or getattr(node,"name",None) in ("html","body"): break
                if id(node) in seen: node=node.parent; continue
                seen.add(id(node))
                txt=clean(node.get_text(" ",strip=True))
                if 900<=len(txt)<=70000:
                    markers=sum(txt.count(k) for k in ("營運摘要","財務表現","展望","Q&A"))
                    headings=len(re.findall(r"\b\d+\.\s*",txt))
                    ident=(" ".join(node.get("class",[]))+" "+str(node.get("id", ""))).lower()
                    class_bonus=sum(5 for k in ("article","post","content","prose","entry") if k in ident)
                    nav_noise=sum(txt.count(k) for k in ("修改反詐騙連結","新增 t179sb01","重大消息上市","反詐騙聯防行動專區","公告券商對媒體轉載"))
                    score=markers*200+headings*5+class_bonus+len(txt)/4000-nav_noise*25
                    candidates.append((f"ancestor-{score:.1f}",txt,score))
                node=node.parent
        if not candidates: 
            for bad in soup.find_all(["script","style","noscript","svg","nav","header","footer","form","aside"]): bad.decompose()
            candidates.append(("document",clean(soup.get_text(" ",strip=True)),0))
        scored=[]
        for c in candidates:
            if len(c)==2:
                method,txt=c; score=sum(txt.count(k) for k in ("營運摘要","財務表現","展望","Q&A"))*200+len(txt)/4000
            else: method,txt,score=c
            # 不把站內 nav 雜訊全部判死刑，只扣分。
            scored.append((score,len(txt),method,txt))
        scored.sort(reverse=True)
        _,_,reader,text=scored[0]
        # 正文邊界
        starts=[]
        for pat in (r"〖[^〗]{1,100}法說會重點內容備忘錄[^〗]{0,100}〗",r"法說會重點內容備忘錄",r"1\.\s*[^ ]{1,60}營運摘要"):
            m=re.search(pat,text)
            if m: starts.append(m.start())
        if starts: text=text[min(starts):]
        for ep in ("免責聲明","註冊富果會員"):
            pos=text.find(ep)
            if pos>1000: text=text[:pos]
        return clean(text),title,published,modified,reader

    def extract_article(self, url: str) -> dict[str, Any]:
        if not url.startswith(POST_PREFIX): raise ValueError("只允許讀取 Fugle earnings-call 詳細文章")
        errors=[]; raw=""; final_url=url; fetch_method=""; self._last_rendered_text=""
        for fn in (
            lambda:self._fetch_browser(url,timeout=45,load_more=False),
            lambda:self._fetch_curl_cffi(url,timeout=35),
            lambda:self._fetch_requests(url,timeout=35),
        ):
            try: raw,final_url,fetch_method=fn(); break
            except Exception as exc: errors.append(f"{type(exc).__name__}: {exc}")
        if not raw: raise RuntimeError("；".join(errors[-3:]))
        text,title,published,modified,reader=self._extract_article_text(raw,self._last_rendered_text)
        markers=sum(1 for k in ("營運摘要","財務表現","展望","Q&A") if k in text)
        if len(text)<1200 or markers<2:
            raise ValueError(f"Fugle 法說會文章正文不足：{len(text)} chars；markers={markers}；fetch={fetch_method}；reader={reader}；errors={' | '.join(errors)}")
        published=parse_date(final_url) or published or parse_date(text[:3000])
        if not modified:
            m=re.search(r"(?:本文已於|最後更新)[^0-9]{0,20}(20\d{2})[./-](\d{1,2})[./-](\d{1,2})",text)
            if m: modified=f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        sections={}; names=["營運摘要","主要業務與產品組合","財務表現","市場與產品發展動態","營運策略與未來發展","展望與指引","Q&A 重點","Q&A"]
        for name in names:
            m=re.search(re.escape(name),text,re.I)
            if not m: continue
            ends=[]
            for other in names:
                if other==name: continue
                mm=re.search(re.escape(other),text[m.end():],re.I)
                if mm: ends.append(m.end()+mm.start())
            block=clean(text[m.start():min(ends) if ends else len(text)])
            if len(block)>=30: sections[name]=block[:12000]
        return {"url":final_url,"symbol":symbol_from_url(final_url),"title":title,"published_date":published,"modified_date":modified,"memo_text":text[:40000],"sections":sections,"content_hash":hashlib.sha256(text.encode("utf-8","ignore")).hexdigest(),"read_at":datetime.now().isoformat(),"reader_method":fetch_method,"article_reader":reader,"detail_read_verified":True,"memo_chars":len(text),"read_errors":errors}

    def db_get(self,url:str):
        con=sqlite3.connect(self.db_path); row=con.execute("SELECT content_hash,analysis_json FROM memos WHERE url=?",(url,)).fetchone(); con.close(); return row or ("","")

    def db_save(self,article:dict[str,Any],analysis:dict[str,Any]|None):
        con=sqlite3.connect(self.db_path)
        con.execute("""INSERT INTO memos(url,symbol,title,published_date,modified_date,content_hash,last_seen,last_analyzed,impact,confidence,analysis_json,memo_text)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET symbol=excluded.symbol,title=excluded.title,published_date=excluded.published_date,modified_date=excluded.modified_date,content_hash=excluded.content_hash,last_seen=excluded.last_seen,last_analyzed=COALESCE(excluded.last_analyzed,memos.last_analyzed),impact=COALESCE(excluded.impact,memos.impact),confidence=COALESCE(excluded.confidence,memos.confidence),analysis_json=COALESCE(excluded.analysis_json,memos.analysis_json),memo_text=excluded.memo_text""",
        (article["url"],article.get("symbol",""),article.get("title",""),article.get("published_date",""),article.get("modified_date",""),article.get("content_hash",""),datetime.now().isoformat(),datetime.now().isoformat() if analysis else None,analysis.get("impact") if analysis else None,analysis.get("confidence") if analysis else None,json.dumps(analysis,ensure_ascii=False) if analysis else None,article.get("memo_text","")))
        con.commit(); con.close()

    def read_memo_tool(self,url:str):
        article=self.extract_article(url); self.db_save(article,None)
        return {"ok":True,**{k:article[k] for k in ("url","symbol","title","published_date","modified_date","content_hash","sections","memo_text","reader_method","detail_read_verified")}}


class FugleEarningsCallAgent:
    def __init__(self,base_dir=".",ollama_host=None,ollama_model=None):
        self.base=Path(base_dir).resolve(); self.crawler=FugleMemoCrawler(self.base)
        self.ollama_host=ollama_host or os.getenv("OLLAMA_HOST",DEFAULT_OLLAMA_HOST)
        self.ollama_model=ollama_model or os.getenv("OLLAMA_MODEL",DEFAULT_OLLAMA_MODEL)
        self.max_rounds=int(os.getenv("EARNINGS_AGENT_MAX_TOOL_ROUNDS","5"))

    def _client(self):
        import ollama
        return ollama.Client(host=self.ollama_host)

    @staticmethod
    def _call_name(call):
        try:return str(call.function.name)
        except Exception:return ""

    @staticmethod
    def _call_args(call):
        try:
            a=call.function.arguments
            return a if isinstance(a,dict) else json.loads(str(a))
        except Exception:return {}

    @staticmethod
    def _assistant_message(resp):
        m=getattr(resp,"message",None)
        out={"role":"assistant","content":getattr(m,"content","") or ""}
        calls=[]
        for c in (getattr(m,"tool_calls",None) or []):
            try:calls.append({"function":{"name":str(c.function.name),"arguments":c.function.arguments}})
            except Exception:pass
        if calls:out["tool_calls"]=calls
        return out

    @staticmethod
    def _parse_json(text):
        s=re.sub(r"^```(?:json)?\s*|\s*```$","",str(text or "").strip(),flags=re.I)
        m=re.search(r"\{.*\}",s,flags=re.S)
        if not m:raise ValueError("Agent JSON not found")
        return json.loads(m.group(0))

    def analyze_one(self,url:str):
        """Agent：Qwen3 先呼叫 read_fugle_memo，再依實際正文輸出結構化摘要。"""
        client=self._client()
        messages=[
          {"role":"system","content":"你是 Fugle 法說會備忘錄研究 Agent。第一個必要動作必須呼叫 read_fugle_memo(url)。不能只看網址/標題，也不能使用記憶補資料。工具成功取得正文後，才能分析。"},
          {"role":"user","content":f"先呼叫 read_fugle_memo 讀取並理解這篇詳細法說會：{url}"},
        ]
        tool_used=False; article=None
        for _ in range(max(2,self.max_rounds)):
            resp=client.chat(model=self.ollama_model,messages=messages,tools=[TOOL_SCHEMA],options={"temperature":0},think=True)
            calls=getattr(getattr(resp,"message",None),"tool_calls",None) or []
            if not calls:
                messages.append({"role":"user","content":"不要先回答；請立即呼叫 read_fugle_memo 工具。"}); continue
            messages.append(self._assistant_message(resp))
            for call in calls:
                if self._call_name(call)!="read_fugle_memo": continue
                tool_used=True; target=self._call_args(call).get("url") or url
                try:
                    article=self.crawler.read_memo_tool(target)
                    if not article.get("detail_read_verified") or len(article.get("memo_text", ""))<1200: raise RuntimeError("正文驗證失敗")
                    messages.append({"role":"tool","content":json.dumps(article,ensure_ascii=False)[:65000]})
                except Exception as exc:
                    messages.append({"role":"tool","content":json.dumps({"ok":False,"error":str(exc),"detail_read_verified":False},ensure_ascii=False)})
            break
        if not tool_used or not article or not article.get("detail_read_verified"): raise RuntimeError("Ollama Agent 沒有成功讀取 Fugle 詳細文章")
        prompt=("你是嚴謹的台股法說會研究 Agent。只根據 read_fugle_memo 讀取的文章正文輸出合法 JSON，禁止補造文章外資訊。"
                 "欄位：impact(利多/利空/中性/混合), confidence(0-100), one_line_summary(繁體中文 1-2 句，指出最重要的營運結論), "
                 "key_points[](3-5 條可直接顯示在網站的重點，每條約 30-100 字，開頭標示「財務：」「營運：」「展望：」「利多：」「風險：」等類別，保留正文中的公司、期間、數字、成長率、資本支出與時程；不要重複同一句), "
                 "highlights[](5-12 個需在 one_line_summary 或 key_points 中逐字出現的關鍵字／數字，例如營收、EPS、毛利率、成長率、產能、訂單、展望、風險；不得加入未出現在文字中的詞), "
                 "financial_highlights[](最多 3 條), operating_highlights[](最多 3 條), guidance[](最多 3 條), positive_factors[](最多 4 條), negative_factors[](最多 4 條), key_risks[](最多 4 條), qa_highlights[](最多 3 條), reasoning(簡述判斷依據)。"
                 "每一條都要精簡、具體、可驗證；數字單位與期間必須正確；若正文沒有資訊，對應陣列留空。")
        context=article.get("memo_text","")
        resp2=client.chat(model=self.ollama_model,messages=[{"role":"system","content":prompt},{"role":"user","content":context[:48000]}],options={"temperature":0},format="json",think=False)
        txt=getattr(getattr(resp2,"message",None),"content","") or ""
        try: a=self._parse_json(txt)
        except Exception:
            resp3=client.chat(model=self.ollama_model,messages=[{"role":"system","content":prompt},{"role":"user","content":context[:24000]}],options={"temperature":0},format="json",think=False)
            a=self._parse_json(getattr(getattr(resp3,"message",None),"content","") or "")
        a["agent_tool_used"]=True; a["memo_read_success"]=True
        out=self._validate(a)
        if not out.get("one_line_summary") and not any(out.get(k) for k in ("financial_highlights","operating_highlights","guidance","positive_factors","negative_factors","qa_highlights")): raise RuntimeError("Qwen3 沒有產生有效摘要")
        return out

    @staticmethod
    def _fallback(article):
        t = str(article.get("memo_text") or "").strip()
        sections = article.get("sections") or {}
        base_text = str(sections.get("營運摘要") or sections.get("財務表現") or t).strip()
        # 去掉站內註冊提示與多餘空白，只保留可直接閱讀的一段。
        base_text = re.sub(r"\s+", " ", base_text)
        base_text = re.sub(r"註冊富果會員.*?(?=\d+\. |$)", "", base_text)
        summary = base_text[:620].strip(" ：:；;")
        if not summary:
            summary = f"{article.get('title','')}：已讀取 Fugle 法說會備忘錄正文。"
        p = [x for x in POSITIVE_WORDS if x in t]
        n = [x for x in NEGATIVE_WORDS if x in t]
        impact = "利多" if len(p) > len(n) + 2 else ("利空" if len(n) > len(p) + 2 else "混合")
        def first_sentence(value: Any, limit: int = 100) -> str:
            text = re.sub(r"\\s+", " ", clean(value))
            fragments = [x.strip(" ：:；;，,") for x in re.split(r"(?<=[。！？；])\\s*|(?<=\\n)", text) if x.strip()]
            chosen = next((x for x in fragments if len(x) >= 12), text)
            return chosen[:limit].rstrip("，、；;：: ") + ("…" if len(chosen) > limit else "")

        key_points = []
        if summary:
            key_points.append("摘要：" + first_sentence(summary, 95))
        for label, value in (
            ("財務", sections.get("財務表現")),
            ("營運", sections.get("營運摘要")),
            ("展望", sections.get("展望與指引")),
            ("風險", "；".join(n)),
        ):
            if not value:
                continue
            point = f"{label}：" + first_sentence(value, 92)
            if point not in key_points:
                key_points.append(point)
            if len(key_points) >= 5:
                break
        highlights = []
        point_blob = " ".join(key_points)
        highlights.extend(m.group(0) for m in re.finditer(r"(?<![A-Za-z])\\d+(?:,\\d{3})*(?:\\.\\d+)?%?(?![A-Za-z])", point_blob))
        highlights.extend(word for word in POSITIVE_WORDS + NEGATIVE_WORDS if word in point_blob)
        highlights = list(dict.fromkeys(highlights))[:12]
        return {
            "impact": impact,
            "confidence": 45 + min(40, abs(len(p) - len(n)) * 5),
            "one_line_summary": first_sentence(summary, 180),
            "key_points": key_points[:5],
            "highlights": highlights,
            "financial_highlights": [first_sentence(sections.get("財務表現"), 300)] if sections.get("財務表現") else [],
            "operating_highlights": [first_sentence(sections.get("營運摘要"), 300)] if sections.get("營運摘要") else [],
            "guidance": [first_sentence(sections.get("展望與指引"), 300)] if sections.get("展望與指引") else [],
            "positive_factors": p[:4],
            "negative_factors": n[:4],
            "key_risks": n[:4],
            "qa_highlights": [first_sentence(sections.get("Q&A 重點") or sections.get("Q&A"), 320)] if (sections.get("Q&A 重點") or sections.get("Q&A")) else [],
            "reasoning": "Ollama 無法完成結構化輸出，改用已成功讀取的 Fugle 正文與可辨識段落產生保守重點。",
            "agent_tool_used": False,
            "memo_read_success": True,
        }

    @staticmethod
    def _validate(a):
        impact=str(a.get("impact","混合")); impact=impact if impact in {"利多","利空","中性","混合"} else "混合"
        try:conf=max(0,min(100,float(a.get("confidence",0))))
        except Exception:conf=0
        def arr(k,n):return [clean(x)[:n] for x in (a.get(k) or []) if clean(x)]
        summary = clean(a.get("one_line_summary", ""))[:600]
        key_points = arr("key_points", 140)[:5]
        highlights = arr("highlights", 24)[:12]
        return {
            "impact": impact,
            "confidence": round(conf, 1),
            "one_line_summary": summary,
            "summary": summary,
            "key_points": key_points,
            "highlights": highlights,
            "financial_highlights": arr("financial_highlights", 320)[:3],
            "operating_highlights": arr("operating_highlights", 320)[:3],
            "guidance": arr("guidance", 320)[:3],
            "positive_factors": arr("positive_factors", 260)[:4],
            "negative_factors": arr("negative_factors", 260)[:4],
            "key_risks": arr("key_risks", 260)[:4],
            "qa_highlights": arr("qa_highlights", 360)[:3],
            "reasoning": clean(a.get("reasoning", ""))[:1200],
            "agent_tool_used": bool(a.get("agent_tool_used")),
            "memo_read_success": bool(a.get("memo_read_success", True)),
        }

    def _fugle_memo_events(self, symbol: str = "", max_events: int = 20):
        rows=self.crawler.discover(); sym=str(symbol or "").strip().upper()
        if sym: rows=[x for x in rows if str(x.get("symbol","")).upper()==sym]
        return rows[:max_events]

    def daily_run(self,days=14,limit=80,force=False,watchlist=None,skip_urls=None):
        discovered=self.crawler.discover(); cutoff=(datetime.now()-timedelta(days=max(0, int(days)-1))).date(); chosen=[]
        for x in discovered:
            d=x.get("published_date",""); inside=True
            if d:
                try:inside=datetime.fromisoformat(d).date()>=cutoff
                except Exception:pass
            if skip_urls and x["url"] in skip_urls:
                continue
            old_hash,_=self.crawler.db_get(x["url"])
            if inside or not old_hash:chosen.append(x)
        chosen=chosen[:limit]; items=[]; errors=[]
        for x in chosen:
            try:
                try:
                    # 主要路徑：由 Qwen3 Agent 自己呼叫 read_fugle_memo。
                    a=self.analyze_one(x["url"])
                    article=self.crawler.extract_article(x["url"])
                except Exception as agent_exc:
                    # 保底路徑：即使 Ollama tool-calling / JSON 輸出失敗，也不要丟掉已成功取得的法說正文。
                    article=self.crawler.extract_article(x["url"])
                    a=self._fallback(article)
                    errors.append({"url":x.get("url"),"title":x.get("title"),"warning":f"Ollama 分析失敗，使用正文 fallback：{agent_exc}"})
                self.crawler.db_save(article,a)
                items.append({**article,**a,"memo_opened":True,"source_type":"Fugle 法說會備忘錄","agent_source":"Ollama Tool Calling -> read_fugle_memo(url) -> detailed article正文 / fallback","detail_read_verified":True})
            except Exception as exc:errors.append({"url":x.get("url"),"title":x.get("title"),"error":str(exc)})
        watch={str(s).strip().upper() for s in (watchlist or [])}
        for x in items:x["in_watchlist"]=str(x.get("symbol","")) in watch
        items.sort(key=lambda x:(x.get("published_date",""),x.get("symbol","")),reverse=True)
        digest=self._digest(items,watch)
        out={"as_of":datetime.now().isoformat(),"topic_url":TOPIC_URL,"discovered_count":len(discovered),"processed_count":len(items),"errors":errors,"items":items,"daily_digest":digest,"primary_source":"Fugle 法說會備忘錄","agent_flow":"Ollama Agent -> read_fugle_memo(url) -> article正文 -> structured analysis"}
        od=self.base/"output"/"research_reports";od.mkdir(parents=True,exist_ok=True); (od/f"fugle_earnings_memo_{datetime.now():%Y-%m-%d}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        self.crawler.close()
        return out

    @staticmethod
    def _digest(items,watch):
        counts={k:0 for k in ("利多","利空","中性","混合")}
        for x in items:counts[x.get("impact","混合")]=counts.get(x.get("impact","混合"),0)+1
        inds={}
        for x in items:inds.setdefault(x.get("industry_name") or "未分類",[]).append(x)
        return {"headline":f"本次共取得 {len(items)} 篇 Fugle 法說會備忘錄；利多 {counts['利多']}、利空 {counts['利空']}、中性 {counts['中性']}、混合 {counts['混合']}。","watchlist_count":sum(1 for x in items if str(x.get("symbol","")) in watch),"industry_signals":[{"industry":k,"memo_count":len(v),"positive_signals":sum(x.get("impact") in ("利多","混合") for x in v),"negative_signals":sum(x.get("impact") in ("利空","混合") for x in v)} for k,v in sorted(inds.items(),key=lambda kv:len(kv[1]),reverse=True)[:30]]}


# Compatibility class used by ResearchAgent.
class EarningsCallAgent(FugleEarningsCallAgent):
    pass
