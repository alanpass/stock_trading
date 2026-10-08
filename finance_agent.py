# -*- coding: utf-8 -*-
"""
財經資訊 Agent：把爬回來的新聞／法說會變成「可讀的重點」。

每次財經資訊更新（08:10、11:00、13:30、16:00、18:00、23:00）都會呼叫：

    agent = FinanceNewsAgent(BASE)
    news, news_digest = agent.analyze_news(articles)
    earnings, earnings_digest = agent.analyze_earnings(items)

做的事（全部「依據文章內容」，不編造數字）：
1. 摘要      每篇新聞整理成 1~3 個重點
2. 畫重點    挑出要用紅字／螢光標示的關鍵詞（數字、公司、事件）
3. 分析      判斷 利多／利空／中性／混合、影響的產業、相關個股、重要度
4. 去重      同一事件的重複報導合併
5. 總覽      產出「今日重點」：最重要的 5~8 件事、利多／利空主題、產業熱度、待追蹤

兩層設計，確保任何情況都有結果：
- 規則層（永遠可用）：關鍵句抽取 + 詞庫判斷，不需要網路或 GPU。
- LLM 層（Ollama Qwen3，可用時才啟動）：重要度前 N 篇由 Qwen3 重寫摘要與判斷；
  已經摘要過的文章會存在 finance_summary_cache.json，下次更新不重做。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

TAIPEI = ZoneInfo("Asia/Taipei")
DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# ----------------------------------------------------------------------
# 詞庫
# ----------------------------------------------------------------------
SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AI／伺服器": ("AI", "伺服器", "輝達", "NVIDIA", "GB200", "GB300", "資料中心", "算力", "緯穎", "廣達", "緯創", "雲端", "ASIC", "GPU"),
    "半導體": ("半導體", "晶圓", "台積電", "聯電", "世界先進", "晶圓代工", "製程", "奈米", "IC設計", "聯發科", "晶片", "設備商"),
    "先進封裝／矽光子": ("CoWoS", "先進封裝", "封測", "矽光子", "CPO", "HBM", "Chiplet"),
    "記憶體": ("記憶體", "DRAM", "NAND", "南亞科", "華邦電", "旺宏", "群聯", "快閃"),
    "PCB／載板": ("PCB", "載板", "ABF", "CCL", "銅箔基板", "臻鼎", "欣興", "台光電", "高速材料"),
    "散熱／電源": ("散熱", "液冷", "水冷", "電源供應", "BBU", "台達電", "奇鋐", "雙鴻"),
    "網通／光通訊": ("網通", "光通訊", "交換器", "光模組", "800G", "1.6T", "低軌衛星"),
    "電動車／車用": ("電動車", "車用", "特斯拉", "充電樁", "電池", "自駕"),
    "面板／光電": ("面板", "友達", "群創", "LED", "光電"),
    "航運／航空": ("航運", "長榮", "陽明", "萬海", "運價", "貨櫃", "散裝", "航空", "華航", "星宇"),
    "金融": ("金融", "銀行", "壽險", "金控", "富邦金", "國泰金", "中信金", "殖利率", "保險"),
    "生技醫療": ("生技", "新藥", "醫療", "藥證", "FDA", "臨床", "疫苗"),
    "原物料／能源": ("鋼鐵", "塑化", "原油", "油價", "銅價", "金價", "黃金", "鋁價", "煤", "天然氣"),
    "營建／資產": ("營建", "房市", "建商", "房價", "不動產", "REITs", "都更"),
    "綠能／重電": ("綠能", "太陽能", "風電", "儲能", "重電", "台電", "電網", "核能"),
    "總體經濟": ("Fed", "聯準會", "降息", "升息", "通膨", "CPI", "PCE", "非農", "失業率", "GDP", "央行", "利率", "PMI", "景氣"),
    "國際股市": ("美股", "道瓊", "那斯達克", "S&P", "標普", "費半", "日經", "歐股", "陸股", "港股", "恆生", "韓股"),
    "匯率": ("匯率", "新台幣", "日圓", "人民幣", "美元指數", "升值", "貶值"),
    "地緣／政策": ("關稅", "制裁", "戰爭", "中東", "俄烏", "川普", "出口管制", "地緣", "貿易戰", "晶片法"),
}

POSITIVE_WORDS = (
    "大漲", "上漲", "漲停", "創高", "創新高", "新高", "看好", "成長", "利多", "受惠", "上修", "調升", "升評", "優於預期", "超預期",
    "擴產", "大單", "訂單", "轉盈", "獲利", "增資", "買超", "回升", "反彈", "強勁", "樂觀", "突破", "翻倍", "營收年增", "雙增", "加碼", "旺季",
)
NEGATIVE_WORDS = (
    "大跌", "下跌", "重挫", "跌停", "下修", "調降", "降評", "衰退", "虧損", "裁員", "不如預期", "低於預期", "利空", "制裁", "關稅",
    "升息", "違約", "賣超", "示警", "崩", "風險", "擔憂", "疲弱", "下滑", "減產", "砍單", "停工", "跳水", "恐慌", "營收年減", "警告",
)
IMPORTANT_WORDS = (
    "法說", "財報", "營收", "EPS", "重大訊息", "庫藏股", "增資", "減資", "併購", "股利", "配息", "Fed", "聯準會", "降息", "升息", "關稅",
    "輝達", "台積電", "鴻海", "聯發科", "停牌", "漲停", "跌停", "創高", "調升", "調降", "上修", "下修", "財測", "指引",
)
BOILERPLATE = re.compile(
    r"(延伸閱讀|更多.{0,12}(請見|請看|報導)|點我|加入.{0,8}(LINE|Line|臉書|Telegram)|免責聲明|版權所有|※.*$|▲.*$|圖[／/:：].*$|"
    r"鉅亨網(記者|編譯|新聞中心)[^，。]{0,12}(綜合報導|報導|台北|編譯)?|（?(圖|照片)[／/:：][^）。]{0,30}）?)"
)
NUMBER_PATTERN = re.compile(
    r"[+\-－]?\d[\d,]*(?:\.\d+)?\s?(?:%|％|倍|億元|億美元|億|萬元|萬張|萬輛|萬|兆|元|美元|點|bp|BP|檔|張|季|年增|年減)"
)
STOCK_CODE_PATTERN = re.compile(r"[（(]\s*([1-9]\d{3}[A-Z]?)\s*(?:[-.]?TW[O]?)?\s*[）)]|\b([1-9]\d{3})[-.]TW[O]?\b")


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------

_RATE_NOISE = re.compile(r"(毛利率|淨利率|營業利益率|營益率|稅率|利潤率|殖利率)")


def _prep(text: str) -> str:
    return _RATE_NOISE.sub(" ", str(text or ""))


def _has_word(text: str, word: str) -> bool:
    """中文詞直接比對；純英數詞需要前後不是英文字母（避免 AI 命中 Taiwan 之類）。"""
    if re.fullmatch(r"[A-Za-z0-9&.\-]+", word):
        return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, re.I) is not None
    return word in text


def _clean(text: Any) -> str:
    t = str(text or "")
    t = re.sub(r"<[^>]+>", " ", t)
    t = BOILERPLATE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])|\n+", text)
    out = []
    for p in parts:
        p = p.strip(" 　\t")
        if len(p) >= 12:
            out.append(p)
    return out


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TAIPEI)
        return dt.astimezone(TAIPEI)
    except Exception:
        pass
    m = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?", str(value))
    if m:
        y, mo, d = (int(m.group(i)) for i in (1, 2, 3))
        hh = int(m.group(4) or 0)
        mm = int(m.group(5) or 0)
        try:
            return datetime(y, mo, d, hh, mm, tzinfo=TAIPEI)
        except Exception:
            return None
    return None


def _norm_title(title: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(title or "").lower())


def _article_key(a: dict[str, Any]) -> str:
    return str(a.get("article_id") or a.get("url") or a.get("title") or "")


def _fingerprint(a: dict[str, Any]) -> str:
    raw = f"{a.get('title','')}|{len(str(a.get('content') or ''))}|{len(str(a.get('summary') or ''))}"
    return hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest()[:12]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------
class FinanceNewsAgent:
    def __init__(
        self,
        base_dir: str | Path = ".",
        model: str | None = None,
        host: str | None = None,
        use_llm: bool | None = None,
        llm_max_articles: int | None = None,
        llm_time_budget_sec: int | None = None,
    ):
        self.base = Path(base_dir).resolve()
        self.out_dir = self.base / "output" / "research_reports"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.out_dir / "finance_summary_cache.json"
        self.model = model or DEFAULT_MODEL
        self.host = (host or DEFAULT_HOST).rstrip("/")
        env_llm = os.getenv("FINANCE_USE_LLM", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.use_llm = env_llm if use_llm is None else bool(use_llm)
        self.llm_max_articles = int(llm_max_articles if llm_max_articles is not None else os.getenv("FINANCE_LLM_MAX_ARTICLES", "40"))
        self.llm_time_budget = int(llm_time_budget_sec if llm_time_budget_sec is not None else os.getenv("FINANCE_LLM_TIME_BUDGET_SEC", "600"))
        self.per_call_timeout = int(os.getenv("FINANCE_LLM_CALL_TIMEOUT", "120"))
        self.llm_ready: bool | None = None
        self.llm_note = ""
        self.stats: dict[str, Any] = {}

    # ---------------- LLM 基礎 ----------------
    def _check_llm(self) -> bool:
        if self.llm_ready is not None:
            return self.llm_ready
        if not self.use_llm:
            self.llm_ready, self.llm_note = False, "FINANCE_USE_LLM=false，只使用規則層"
            return False
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=4)
            r.raise_for_status()
            names = [str(m.get("name") or m.get("model") or "") for m in r.json().get("models", [])]
            base = self.model.split(":")[0]
            if not any(n == self.model or n.startswith(base) for n in names):
                self.llm_ready, self.llm_note = False, f"Ollama 已連線但找不到模型 {self.model}（ollama pull {self.model}）"
            else:
                self.llm_ready, self.llm_note = True, "ok"
        except Exception as exc:
            self.llm_ready, self.llm_note = False, f"無法連線 Ollama：{type(exc).__name__}"
        return bool(self.llm_ready)

    def _chat_json(self, system: str, user: str) -> dict[str, Any] | None:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": 6144},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user + "\n/no_think"},
            ],
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.per_call_timeout)
        r.raise_for_status()
        content = str((r.json().get("message") or {}).get("content") or "")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # ---------------- 快取 ----------------
    def _load_cache(self) -> dict[str, Any]:
        try:
            obj = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.cache_path)

    # ---------------- 規則層 ----------------
    @staticmethod
    def detect_sectors(text: str) -> list[str]:
        scored = []
        low = _prep(text)
        for sector, words in SECTOR_KEYWORDS.items():
            hits = sum(1 for w in words if _has_word(low, w))
            if hits:
                scored.append((hits, sector))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [s for _, s in scored[:3]]

    @staticmethod
    def detect_sentiment(text: str) -> tuple[str, int]:
        pos = sum(text.count(w) for w in POSITIVE_WORDS)
        neg = sum(text.count(w) for w in NEGATIVE_WORDS)
        score = pos - neg
        if pos >= 2 and neg >= 2 and abs(score) <= 1:
            return "混合", score
        if score >= 2:
            return "利多", score
        if score <= -2:
            return "利空", score
        if pos and neg:
            return "混合", score
        return "中性", score

    @staticmethod
    def detect_stocks(article: dict[str, Any], text: str) -> list[str]:
        codes: list[str] = []
        for ref in article.get("stock_refs") or []:
            m = re.search(r"[1-9]\d{3}[A-Z]?", str(ref))
            if m:
                codes.append(m.group(0))
        for m in STOCK_CODE_PATTERN.finditer(text):
            codes.append(m.group(1) or m.group(2))
        return _dedupe_keep_order(codes)[:8]

    def _score_sentence(self, sentence: str, idx: int, title: str) -> float:
        score = 0.0
        score += min(3, len(NUMBER_PATTERN.findall(sentence))) * 1.2
        score += sum(1.0 for w in IMPORTANT_WORDS if w in sentence)
        score += sum(0.6 for w in POSITIVE_WORDS + NEGATIVE_WORDS if w in sentence)
        if idx == 0:
            score += 1.5
        elif idx == 1:
            score += 0.8
        if title:
            overlap = sum(1 for ch in set(title) if ch in sentence and "\u4e00" <= ch <= "\u9fff")
            score += min(2.0, overlap / 8)
        if len(sentence) > 140:
            score -= 1.0
        return score

    def extract_points(self, article: dict[str, Any], max_points: int = 3, max_chars: int = 90) -> list[str]:
        title = str(article.get("title") or "")
        body = _clean(article.get("content") or "")
        summ = _clean(article.get("summary") or "")
        text = body if len(body) >= len(summ) else summ
        if summ and summ not in text:
            text = f"{summ} {text}"
        sentences = _split_sentences(text)
        if not sentences:
            return [title] if title else []
        ranked = sorted(((self._score_sentence(s, i, title), i, s) for i, s in enumerate(sentences)), reverse=True)
        chosen = sorted(ranked[:max_points], key=lambda x: x[1])
        points = []
        for _, _, s in chosen:
            s = s.strip("，、 ")
            if len(s) > max_chars:
                cut = s[:max_chars]
                k = max(cut.rfind("，"), cut.rfind("、"), cut.rfind("；"))
                s = (cut[:k] if k >= 30 else cut).rstrip("，、；") + "…"
            points.append(s)
        return _dedupe_keep_order(points)[:max_points]

    def extract_highlights(self, points: list[str], title: str, stocks: list[str], sectors: list[str]) -> list[str]:
        blob = " ".join(points)
        prepped = _prep(blob)
        found: list[str] = []
        found += [m.group(0).strip() for m in NUMBER_PATTERN.finditer(blob)]
        for w in IMPORTANT_WORDS + POSITIVE_WORDS + NEGATIVE_WORDS:
            if _has_word(prepped, w):
                found.append(w)
        for sec in sectors:
            for w in SECTOR_KEYWORDS.get(sec, ()):
                if len(w) >= 2 and _has_word(prepped, w):
                    found.append(w)
        for code in stocks:
            if code in blob:
                found.append(code)
        found = [h for h in _dedupe_keep_order(found) if 1 < len(h) <= 14]
        found.sort(key=lambda x: -len(x))  # 先標長詞，避免被短詞切碎
        return found[:10]

    def importance(self, article: dict[str, Any], text: str, now: datetime) -> float:
        score = 1.0
        title = str(article.get("title") or "")
        if article.get("stock_refs"):
            score += 1.5
        score += min(3.0, sum(1.0 for w in IMPORTANT_WORDS if w in title))
        score += min(2.0, sum(0.4 for w in IMPORTANT_WORDS if w in text))
        score += min(1.5, len(NUMBER_PATTERN.findall(text)) * 0.3)
        if article.get("list_category") == "headline":
            score += 1.0
        dt = _parse_dt(article.get("published_ts") or article.get("published"))
        if dt:
            age_h = max(0.0, (now - dt).total_seconds() / 3600)
            score += 2.0 if age_h <= 6 else 1.0 if age_h <= 24 else 0.0
        if len(str(article.get("content") or "")) >= 600:
            score += 0.5
        return round(min(10.0, score), 2)

    def _rule_analysis(self, article: dict[str, Any], now: datetime) -> dict[str, Any]:
        title = str(article.get("title") or "")
        text = f"{title} {_clean(article.get('summary'))} {_clean(article.get('content'))}"
        points = self.extract_points(article)
        sectors = self.detect_sectors(text)
        sentiment, score = self.detect_sentiment(text)
        stocks = self.detect_stocks(article, text)
        return {
            "points": points,
            "sentiment": sentiment,
            "sentiment_score": score,
            "sectors": sectors,
            "stocks": stocks,
            "why": "",
            "highlights": self.extract_highlights(points, title, stocks, sectors),
            "importance": self.importance(article, text, now),
            "ai_source": "rule",
        }

    # ---------------- LLM 層 ----------------
    @classmethod
    def _canon_sectors(cls, raw_sectors: list[Any], text: str) -> list[str]:
        """把 LLM 回傳的產業名稱對到固定分類；對不到的丟掉，全都對不到就用規則判斷。"""
        allowed = list(SECTOR_KEYWORDS)
        out: list[str] = []
        for raw in raw_sectors:
            name = str(raw).strip()
            if not name:
                continue
            if name in allowed:
                out.append(name)
                continue
            hit = [a for a in allowed if name in a or a in name]
            if not hit:
                hit = cls.detect_sectors(name)[:1]
            out.extend(hit)
        out = _dedupe_keep_order(out)[:3]
        return out or cls.detect_sectors(text)

    ARTICLE_SYSTEM = (
        "你是台股財經研究助理。只能根據使用者提供的新聞內容整理，不可編造任何數字、公司或事件。"
        "輸出 JSON，欄位：points（1~3 條繁體中文重點，每條 60 字內，保留關鍵數字與公司名）、"
        "sentiment（利多／利空／中性／混合，指對台股或相關產業的影響）、sectors（最多 3 個，只能從這份清單挑：" + "、".join(SECTOR_KEYWORDS) + "）、"
        "stocks（新聞明確提到的台股代號字串陣列，沒有就空陣列）、why（一句話說明為什麼值得注意，40 字內）、"
        "highlights（3~8 個關鍵詞，必須逐字出現在 points 裡，例如數字、公司、事件）。"
    )

    def _llm_article(self, article: dict[str, Any]) -> dict[str, Any] | None:
        body = _clean(article.get("content") or article.get("summary") or "")[:2200]
        user = f"標題：{article.get('title','')}\n發佈：{article.get('published','')}\n分類：{article.get('category','')}\n內容：{body}"
        obj = self._chat_json(self.ARTICLE_SYSTEM, user)
        if not obj:
            return None
        points = [str(x).strip() for x in (obj.get("points") or []) if str(x).strip()][:3]
        if not points:
            return None
        sentiment = str(obj.get("sentiment") or "").strip()
        if sentiment not in {"利多", "利空", "中性", "混合"}:
            sentiment = "中性"
        blob = " ".join(points)
        highlights = [str(h).strip() for h in (obj.get("highlights") or []) if str(h).strip() and str(h).strip() in blob]
        sectors = self._canon_sectors(obj.get("sectors") or [], f"{article.get('title','')} {body}")
        stocks = [re.search(r"[1-9]\d{3}[A-Z]?", str(x)).group(0) for x in (obj.get("stocks") or []) if re.search(r"[1-9]\d{3}[A-Z]?", str(x))][:8]
        return {
            "points": points,
            "sentiment": sentiment,
            "sectors": sectors,
            "stocks": stocks,
            "why": str(obj.get("why") or "").strip()[:80],
            "highlights": highlights,
        }

    # ---------------- 新聞主流程 ----------------
    def _dedupe_articles(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """合併同一事件的重複報導。標題幾乎相同、且標題裡的數字也相同才算重複（避免把不同日期／不同數字的系列新聞誤併）。"""
        kept: list[dict[str, Any]] = []
        sigs: list[tuple[str, list[str]]] = []
        for a in articles:
            title = str(a.get("title") or "")
            n = _norm_title(title)
            if not n:
                continue
            digits = re.findall(r"\d+(?:\.\d+)?", title)
            dup_idx = None
            for i, (k, kd) in enumerate(sigs):
                if digits != kd:
                    continue
                if n == k or (abs(len(n) - len(k)) <= 4 and SequenceMatcher(None, n, k).ratio() >= 0.92):
                    dup_idx = i
                    break
            if dup_idx is None:
                kept.append(a)
                sigs.append((n, digits))
            else:
                kept[dup_idx].setdefault("duplicates", []).append(a.get("url") or a.get("title"))
        return kept

    def analyze_news(self, articles: list[dict[str, Any]], now: datetime | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        now = now or datetime.now(TAIPEI)
        t0 = time.time()
        rows = [dict(a) for a in articles if isinstance(a, dict) and a.get("title")]
        before = len(rows)
        rows = self._dedupe_articles(rows)

        # 1) 規則層：全部文章
        for a in rows:
            ai = self._rule_analysis(a, now)
            a["_rule"] = ai

        # 2) LLM 層：依重要度挑前 N 篇，且略過已經摘要過的
        cache = self._load_cache()
        llm_done = llm_cached = llm_fail = 0
        llm_on = self._check_llm()
        ranked = sorted(rows, key=lambda x: -x["_rule"]["importance"])
        budget_hit = False
        for idx, a in enumerate(ranked):
            key = _article_key(a)
            fp = _fingerprint(a)
            hit = cache.get(key)
            if hit and hit.get("fp") == fp and hit.get("ai"):
                a["_llm"] = hit["ai"]
                llm_cached += 1
                continue
            if not llm_on or idx >= self.llm_max_articles:
                continue
            if time.time() - t0 > self.llm_time_budget:
                budget_hit = True
                continue
            try:
                ai = self._llm_article(a)
            except Exception as exc:
                ai = None
                if llm_fail == 0:
                    print(f"[agent] Qwen3 呼叫失敗（之後同類錯誤不再重複顯示）：{type(exc).__name__}: {exc}", flush=True)
            if (llm_done + llm_fail) % 5 == 0:
                print(f"[agent] Qwen3 摘要進度 {llm_done + llm_fail + 1}/{min(self.llm_max_articles, len(ranked))}（已用 {time.time() - t0:.0f} 秒）", flush=True)
            if ai:
                a["_llm"] = ai
                cache[key] = {"fp": fp, "ai": ai, "ts": now.isoformat()}
                llm_done += 1
            else:
                llm_fail += 1

        # 3) 合併：LLM 優先、規則補洞
        for a in rows:
            r = a.pop("_rule")
            l = a.pop("_llm", None)
            use = dict(r)
            if l:
                use.update({k: v for k, v in l.items() if v not in (None, "", [])})
                use["ai_source"] = "qwen3"
                if not l.get("highlights"):
                    use["highlights"] = self.extract_highlights(use["points"], a.get("title", ""), use.get("stocks", []), use.get("sectors", []))
                use["stocks"] = _dedupe_keep_order(list(use.get("stocks", [])) + list(r.get("stocks", [])))[:8]
                # 舊快取裡的 LLM 產業名稱可能不是固定分類，這裡一律再對應一次
                use["sectors"] = self._canon_sectors(use.get("sectors", []), f"{a.get('title','')} {_clean(a.get('content'))[:1500]}") or r.get("sectors", [])
            a["ai_points"] = use["points"]
            a["ai_summary"] = "；".join(use["points"])
            a["summary_raw"] = a.get("summary", "")
            a["summary"] = a["ai_summary"] or a.get("summary", "")  # 讓晨報／Email／舊頁面也直接拿到 AI 摘要
            a["sentiment"] = use["sentiment"]
            a["sectors"] = use["sectors"]
            a["stocks"] = use["stocks"]
            a["why"] = use.get("why", "")
            a["highlights"] = use["highlights"]
            a["importance"] = r["importance"]
            a["ai_source"] = use["ai_source"]
            dt = _parse_dt(a.get("published_ts") or a.get("published"))
            a["published_at"] = dt.isoformat(timespec="seconds") if dt else str(a.get("published") or "")

        # 4) 快取只保留目前視窗內的文章（取代舊資料）
        live = {_article_key(a) for a in rows}
        cache = {k: v for k, v in cache.items() if k in live}
        try:
            self._save_cache(cache)
        except Exception:
            pass

        rows.sort(key=lambda x: (-x["importance"], str(x.get("published_ts") or x.get("published") or "")), reverse=False)
        rows.sort(key=lambda x: x["importance"], reverse=True)

        digest = self._build_digest(rows, now)
        digest["llm"] = {
            "enabled": llm_on,
            "note": self.llm_note,
            "model": self.model if llm_on else "",
            "summarized_now": llm_done,
            "from_cache": llm_cached,
            "failed": llm_fail,
            "time_budget_hit": budget_hit,
        }
        self.stats = {
            "input": before,
            "after_dedupe": len(rows),
            "llm_summarized": llm_done,
            "llm_cached": llm_cached,
            "llm_failed": llm_fail,
            "seconds": round(time.time() - t0, 1),
            "llm_note": self.llm_note,
        }
        return rows, digest

    # ---------------- 總覽 ----------------
    def _build_digest(self, rows: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
        total = len(rows)
        counts = {"利多": 0, "利空": 0, "中性": 0, "混合": 0}
        sector_stat: dict[str, dict[str, int]] = {}
        for a in rows:
            counts[a.get("sentiment", "中性")] = counts.get(a.get("sentiment", "中性"), 0) + 1
            for sec in a.get("sectors", []) or []:
                st = sector_stat.setdefault(sec, {"count": 0, "利多": 0, "利空": 0})
                st["count"] += 1
                if a.get("sentiment") in ("利多", "利空"):
                    st[a["sentiment"]] += 1
        sector_heat = sorted(
            ({"sector": k, **v, "tone": "偏多" if v["利多"] > v["利空"] else "偏空" if v["利空"] > v["利多"] else "中性"} for k, v in sector_stat.items()),
            key=lambda x: -x["count"],
        )[:10]

        top = rows[:8]
        key_points = []
        for a in top:
            p = (a.get("ai_points") or [""])[0]
            if p:
                key_points.append({"title": a.get("title", ""), "point": p, "sentiment": a.get("sentiment", "中性"), "url": a.get("url", ""), "highlights": a.get("highlights", [])[:6]})

        def theme(sent: str) -> list[dict[str, Any]]:
            out = []
            for a in rows:
                if a.get("sentiment") == sent and len(out) < 4:
                    out.append({"theme": a.get("title", ""), "reason": a.get("why") or (a.get("ai_points") or [""])[0], "sectors": a.get("sectors", []), "url": a.get("url", "")})
            return out

        top_sec = "、".join(x["sector"] for x in sector_heat[:3]) or "—"
        headline = (
            f"最近 2 天共整理 {total} 則財經新聞：利多 {counts['利多']}、利空 {counts['利空']}、中性 {counts['中性']}、混合 {counts['混合']}；"
            f"討論度最高的產業：{top_sec}。"
        )
        digest = {
            "generated_at": now.isoformat(timespec="seconds"),
            "headline": headline,
            "stats": {"total": total, **counts},
            "key_points": key_points,
            "bullish": theme("利多"),
            "bearish": theme("利空"),
            "sector_heat": sector_heat,
            "watch_items": [],
            "generated_by": "rule",
        }
        # LLM 總結（可用時）：把前 25 則重點交給 Qwen3 寫成「今日重點」
        if self.llm_ready:
            try:
                lines = []
                for a in rows[:25]:
                    lines.append(f"- [{a.get('sentiment','中性')}]{a.get('title','')}｜{'；'.join((a.get('ai_points') or [])[:2])}")
                system = (
                    "你是台股財經早報主編。只能根據提供的新聞條列整理，不可編造。輸出 JSON："
                    "headline（60 字內總結今天市場焦點）、key_points（5~8 條最重要的事件重點，每條 50 字內）、"
                    "watch_items（3~5 條接下來要追蹤的事項）、risks（最多 3 條風險）。"
                )
                obj = self._chat_json(system, "\n".join(lines))
                if obj and obj.get("key_points"):
                    digest["agent_headline"] = str(obj.get("headline") or "").strip()
                    digest["agent_key_points"] = [str(x).strip() for x in obj.get("key_points", []) if str(x).strip()][:8]
                    digest["watch_items"] = [str(x).strip() for x in obj.get("watch_items", []) if str(x).strip()][:5]
                    digest["risks"] = [str(x).strip() for x in obj.get("risks", []) if str(x).strip()][:3]
                    digest["generated_by"] = "qwen3"
            except Exception:
                pass
        return digest

    # ---------------- 法說會 ----------------
    def analyze_earnings(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        out = []
        counts = {"利多": 0, "利空": 0, "中性": 0, "混合": 0}
        for it in items:
            if not isinstance(it, dict):
                continue
            row = dict(it)
            raw = row.get("one_line_summary") or row.get("summary") or row.get("memo_text") or row.get("content") or ""
            if isinstance(raw, list):
                raw = "；".join(str(x) for x in raw)
            fake = {"title": row.get("title", ""), "content": str(raw), "summary": str(raw)}
            points = self.extract_points(fake, max_points=3, max_chars=100)
            text = f"{row.get('title','')} {raw}"
            sentiment = str(row.get("impact") or row.get("judgement") or "").strip()
            if sentiment not in counts:
                sentiment, _ = self.detect_sentiment(text)
            counts[sentiment] += 1
            sectors = self.detect_sectors(text)
            row["ai_points"] = points
            row["sentiment"] = sentiment
            row["sectors"] = sectors
            row["highlights"] = self.extract_highlights(points, row.get("title", ""), [str(row.get("symbol") or "")], sectors)
            out.append(row)
        digest = {
            "total": len(out),
            **counts,
            "headline": f"最近 2 天共 {len(out)} 場法說會備忘錄：利多 {counts['利多']}、利空 {counts['利空']}、中性 {counts['中性']}、混合 {counts['混合']}。" if out else "最近 2 天沒有可用的法說會備忘錄。",
        }
        return out, digest
