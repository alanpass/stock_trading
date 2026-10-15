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
- AI 層：Claude 排程任務（finance-news-digest skill，finance_digest.py apply）為重要新聞撰寫
  逐篇重點並存進 finance_summary_cache.json；每次規則更新都會沿用，不必重做。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


TAIPEI = ZoneInfo("Asia/Taipei")

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
    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir).resolve()
        self.out_dir = self.base / "output" / "research_reports"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.out_dir / "finance_summary_cache.json"
        self.stats: dict[str, Any] = {}

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

    def normalize_article_ai(self, obj: dict[str, Any], article: dict[str, Any]) -> dict[str, Any] | None:
        """驗證 Claude 寫出的單篇新聞 AI 重點（points／sentiment／sectors／stocks／why／highlights）。"""
        body = _clean(article.get("content") or article.get("summary") or "")[:2200]
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

        # 2) AI 層：沿用 Claude 已寫好的逐篇重點（finance_summary_cache.json）
        cache = self._load_cache()
        ai_cached = 0
        for a in rows:
            key = _article_key(a)
            hit = cache.get(key)
            # 以 article_id＋標題比對：爬蟲重抓時內文長度可能不同，但仍是同一篇新聞。
            same = bool(hit and (hit.get("fp") == _fingerprint(a) or hit.get("title") == a.get("title")))
            if same and hit.get("ai"):
                a["_llm"] = hit["ai"]
                ai_cached += 1

        # 3) 合併：AI 重點優先、規則補洞
        for a in rows:
            r = a.pop("_rule")
            l = a.pop("_llm", None)
            use = dict(r)
            if l:
                use.update({k: v for k, v in l.items() if v not in (None, "", [])})
                use["ai_source"] = l.get("ai_source") or "qwen3"  # 舊快取為 Qwen3 寫的
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
        digest["llm"] = {"source": "Claude 排程任務", "from_cache": ai_cached}
        self.stats = {
            "input": before,
            "after_dedupe": len(rows),
            "ai_cached": ai_cached,
            "seconds": round(time.time() - t0, 1),
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
        # 「AI 今日重點」由 Claude 排程任務寫入（finance_digest.py apply），並在每次規則更新時沿用。
        return digest

    # ---------------- 法說會 ----------------
    def analyze_earnings(self, items: list[dict[str, Any]], days: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """保留法說會 Agent 的結構化分析，再補足精簡要點與可驗證的螢光標記；不影響新聞分析流程。"""
        out = []
        counts = {"利多": 0, "利空": 0, "中性": 0, "混合": 0}

        def as_values(value: Any) -> list[str]:
            if isinstance(value, (list, tuple)):
                source = value
            elif value in (None, ""):
                source = []
            else:
                source = [value]
            return [clean for clean in (_clean(x) for x in source) if clean]

        def compact_point(label: str, value: Any, title: str) -> str:
            """將單一 Agent 段落縮成一條可讀重點，保留重要數字與事實。"""
            text = _clean(value)
            if not text:
                return ""
            points = self.extract_points(
                {"title": title, "content": text, "summary": ""},
                max_points=1,
                max_chars=105,
            )
            snippet = points[0] if points else text[:105].rstrip("，、；;：: ") + ("…" if len(text) > 105 else "")
            if not snippet:
                return ""
            if label and not snippet.startswith((label + "：", label + ":")):
                snippet = f"{label}：{snippet}"
            return snippet[:130]

        for it in items:
            if not isinstance(it, dict):
                continue
            row = dict(it)
            title = str(row.get("title") or row.get("headline") or "法說會備忘錄").strip()
            summary = _clean(
                row.get("one_line_summary")
                or row.get("agent_headline")
                or row.get("summary")
                or ""
            )
            # 正文與結構化段落都要參與判斷，避免只用一句摘要遺失風險／展望脈絡。
            sections = row.get("sections") if isinstance(row.get("sections"), dict) else {}
            section_text = " ".join(
                _clean(value)
                for key in (
                    "financial_highlights", "operating_highlights", "guidance",
                    "positive_factors", "negative_factors", "key_risks", "qa_highlights",
                )
                for value in as_values(row.get(key))
            )
            raw = _clean(
                row.get("memo_text")
                or row.get("content")
                or row.get("summary")
                or summary
                or ""
            )
            if not raw and sections:
                raw = _clean(" ".join(str(v) for v in sections.values()))
            analysis_text = f"{title} {summary} {raw[:9000]} {section_text[:4000]}"
            sentiment = str(
                row.get("impact") or row.get("judgement") or row.get("sentiment") or ""
            ).strip()
            if sentiment not in counts:
                sentiment, _ = self.detect_sentiment(analysis_text)
            counts[sentiment] += 1

            sectors = row.get("sectors")
            if not isinstance(sectors, list) or not sectors:
                sectors = self.detect_sectors(analysis_text)
            else:
                sectors = self._canon_sectors(sectors, analysis_text)

            symbol = str(
                row.get("symbol") or row.get("stock_code") or row.get("code") or ""
            ).strip().upper()
            stocks = [symbol] if symbol else []
            for stock in row.get("stocks") or []:
                text_stock = str(stock).strip().upper()
                if re.fullmatch(r"[0-9]{4}[A-Z]?", text_stock):
                    stocks.append(text_stock)
            stocks = _dedupe_keep_order(stocks)[:8]

            # 優先採用法說會 Agent 的 key_points；舊快取沒有該欄位時，
            # 再從 Agent 已整理的財務、營運、展望、風險段落補出同格式重點。
            points = as_values(row.get("key_points") or row.get("ai_points"))
            points = [p[:140] for p in points if len(p.strip()) >= 6][:5]
            if not points and summary:
                point = compact_point("摘要", summary, title)
                if point:
                    points.append(point)

            # 舊快取可能有一行摘要但沒有新格式 key_points；仍從 Agent 的財務／營運／
            # 展望／利多與風險欄位補足 3～5 點，避免一整場法說會只剩一個摘要句。
            if len(points) < 5:
                for label, values in (
                    ("財務", row.get("financial_highlights") or sections.get("財務表現")),
                    ("營運", row.get("operating_highlights") or sections.get("營運摘要")),
                    ("展望", row.get("guidance") or sections.get("展望與指引")),
                    ("利多", row.get("positive_factors")),
                    ("風險", row.get("negative_factors") or row.get("key_risks")),
                    ("Q&A", row.get("qa_highlights") or sections.get("Q&A 重點") or sections.get("Q&A")),
                ):
                    for value in as_values(values):
                        point = compact_point(label, value, title)
                        point_core = re.sub(r"^[^：:]{1,8}[：:]", "", point)
                        point_signature = re.sub(r"[\W_]+", "", point_core).lower()
                        duplicates = False
                        for existing in points:
                            existing_core = re.sub(r"^[^：:]{1,8}[：:]", "", existing)
                            existing_signature = re.sub(r"[\W_]+", "", existing_core).lower()
                            if point_signature and existing_signature and (
                                point_signature in existing_signature
                                or existing_signature in point_signature
                            ):
                                duplicates = True
                                break
                        if point and not duplicates:
                            points.append(point)
                            break
                    if len(points) >= 5:
                        break
            if not points:
                points = self.extract_points(
                    {"title": title, "content": raw[:6000], "summary": summary},
                    max_points=3,
                    max_chars=100,
                )

            # LLM 產生的 highlights 必須逐字出現在摘要／重點中，並以規則再補公司代號、
            # 數字和產業詞；畫面只會標示真正在文字裡的字詞，不創造額外關鍵字。
            highlight_blob = " ".join([summary] + points)
            agent_highlights = [
                word for word in as_values(row.get("highlights"))
                if word in highlight_blob
            ]
            rule_highlights = self.extract_highlights(points, title, stocks, sectors)
            highlights = _dedupe_keep_order(agent_highlights + rule_highlights)
            highlights = [
                word for word in highlights
                if len(word) > 1 and word in highlight_blob
            ][:12]

            row["one_line_summary"] = summary or (points[0] if points else title)
            row["ai_points"] = points
            row["ai_summary"] = "；".join(points)
            row["summary"] = row["one_line_summary"]
            row["sentiment"] = sentiment
            row["impact"] = str(row.get("impact") or sentiment)
            row["sectors"] = sectors
            row["stocks"] = stocks
            row["highlights"] = highlights
            row["analysis_source"] = row.get("analysis_source") or (
                "AI 法說會 Agent"
                if row.get("agent_tool_used")
                else "正文規則備援"
            )
            out.append(row)

        digest = {
            "total": len(out),
            **counts,
            "headline": (
                f"最近 {int(days)} 天共 {len(out)} 場法說會備忘錄："
                f"利多 {counts['利多']}、利空 {counts['利空']}、"
                f"中性 {counts['中性']}、混合 {counts['混合']}。"
                if out else f"最近 {int(days)} 天沒有可用的法說會備忘錄。"
            ),
        }
        return out, digest
