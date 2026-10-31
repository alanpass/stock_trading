# -*- coding: utf-8 -*-
"""TPEx「產業價值鏈資訊平台」全市場產業鏈分類。

ic.tpex.org.tw 依產品把上市櫃公司分到 47 個產業、數百個子產業鏈（上游／中游／下游），同一家公司可屬於多個子鏈。
每個產業一頁（introduce.php?ic=<代碼>），頁面內含所有子鏈名稱、上中下游位置與公司清單，47 次請求就能取得全市場分類。
只保留上市、上櫃（含外國上市 KY 股）公司，不含興櫃與創櫃。

輸出：data/value_chain_master.json，並發布到 GitHub（output/research_reports/value_chain_public.json），網站靠這份顯示公司主要業務
  {"generated_at", "source", "industries": [{"code", "name", "chains": [{"id", "name", "stage", "members": [...]}]}]}

  python value_chain.py            # 重新爬一次（約 1 分鐘）
  python value_chain.py --stats    # 只看現有檔案的統計
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "value_chain_master.json"
PUBLIC_REPO_PATH = "output/research_reports/value_chain_public.json"
PUBLIC_OUT = BASE / PUBLIC_REPO_PATH
INTRO_URL = "https://ic.tpex.org.tw/introduce.php?ic={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TaiwanStockDashboard/1.0"}
LISTED_HEADINGS = ("上市", "上櫃")        # 「本國上市」「本國上櫃」「外國上市」…；興櫃、創櫃不納入
MAX_AGE_DAYS = 30                         # 分類很少變，一個月重爬一次即可


def _get(url: str) -> str:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 + attempt * 3)
    return ""


def industry_list(page: str) -> list[tuple[str, str]]:
    """頁首下拉選單的產業清單：[(代碼, 名稱)]。"""
    return [(c, html.unescape(n).strip()) for c, n in re.findall(r"<option value='([A-Z0-9]+)'\s*[^>]*>([^<]+)</option>", page)]


def parse_industry(page: str) -> list[dict]:
    """解析一個產業頁：子鏈 id、名稱、上中下游與上市櫃公司代號。"""
    chains: dict[str, dict] = {}
    stage = ""
    # 依出現順序走過「上游／中游／下游」標題與子鏈方塊，子鏈的位置就是它前面最近的標題
    for m in re.finditer(r'chain-title-panel">([^<]+)<|id="ic_link_([A-Z0-9]+)"[^>]*>([\s\S]*?)</div>', page):
        if m.group(1):
            stage = html.unescape(m.group(1)).strip()
        else:
            # 名稱裡常有 <br/> 換行（例如「生產製程<br/>及<br/>檢測設備」），去掉標籤再接起來
            cid, name = m.group(2), re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", m.group(3))))
            chains.setdefault(cid, {"id": cid, "name": name, "stage": stage, "members": []})
    # 每個子鏈的公司清單從 companyList_<id> 開始、到下一個 companyList_ 為止
    # 頁面裡同一份清單會出現兩次（第二份沒有第三層名稱），只用第一份
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'<div id="companyList_([A-Z0-9]+)"', page)]
    done: set[str] = set()
    for n, (pos, cid) in enumerate(starts):
        if cid not in chains or cid in done:
            continue
        done.add(cid)
        block = page[pos:starts[n + 1][0] if n + 1 < len(starts) else len(page)]
        chain = chains[cid]
        # 第三層（例如 IC設計 > 電源管理IC）：名稱在 sc_link_<id>，公司在 sc_company_<id> 表格
        sub_names = {sid: re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", raw))).strip("►")
                     for sid, raw in re.findall(r'id="sc_link_([A-Z0-9]+)"[^>]*>([\s\S]*?)</div>', block)}
        tables = re.findall(r'<table id="sc_company_([A-Z0-9]+)"[^>]*>([\s\S]*?)</table>', block)
        if tables:
            chain["subs"] = []
            for sid, table in tables:
                name = re.sub(r"\(\d+家\)$", "", sub_names.get(sid, sid))
                members = _listed_codes(table)
                if members:
                    chain["subs"].append({"id": sid, "name": name, "members": members})
                for code in members:
                    if code not in chain["members"]:
                        chain["members"].append(code)
        else:
            chain["members"] = _listed_codes(block.split("</table>", 1)[0])
    return [c for c in chains.values() if c["members"]]


def _listed_codes(table: str) -> list[str]:
    """表格裡上市、上櫃公司的代號。「本國上市公司(n家)」標題用 rowspan 跨好幾列，標題之後到下一個標題之前的公司都屬於它。"""
    out, listed = [], False
    for m in re.finditer(r"<b>([^<(]+)\(\d+家\)</b>|stk_code=([0-9A-Z]+)", table):
        if m.group(1):
            listed = any(k in m.group(1) for k in LISTED_HEADINGS)
        elif listed and m.group(2) not in out:
            out.append(m.group(2))
    return out


def leaf_groups(data: dict) -> list[dict]:
    """最細一層的分類（有第三層就用第三層）：[{"id", "name", "path", "industry", "stage", "members"}]。"""
    out = []
    for ind in data.get("industries") or []:
        for c in ind["chains"]:
            for leaf in c.get("subs") or [c]:
                path = f"{ind['name']} > {c['name']}" + (f" > {leaf['name']}" if leaf is not c else "")
                out.append({"id": leaf["id"], "name": leaf["name"], "path": path, "industry": ind["name"],
                            "chain": c["name"], "stage": c.get("stage", ""), "members": list(leaf["members"])})
    return out


def crawl(sleep_seconds: float = 0.8, log=print) -> dict:
    first = _get(INTRO_URL.format("L000"))
    industries = industry_list(first)
    if len(industries) < 30:
        raise RuntimeError(f"產業清單只有 {len(industries)} 個，頁面格式可能改了")
    out = []
    for i, (code, name) in enumerate(industries, 1):
        page = first if code == "L000" else _get(INTRO_URL.format(code))
        chains = parse_industry(page)
        out.append({"code": code, "name": name, "chains": chains})
        log(f"[{i}/{len(industries)}] {name}：{len(chains)} 個子鏈、{len({s for c in chains for s in c['members']})} 家")
        time.sleep(sleep_seconds)
    data = {"generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
            "source": "https://ic.tpex.org.tw/", "industries": out}
    if sum(len(i["chains"]) for i in out) < 100:
        raise RuntimeError("子鏈太少，可能沒有解析到公司清單；不覆蓋舊檔")
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    publish(data)
    return data


def publish(data: dict) -> None:
    """分類檔寫一份到公開路徑並發布；失敗時網站沿用上一版，不影響爬取結果。"""
    try:
        PUBLIC_OUT.parent.mkdir(parents=True, exist_ok=True)
        PUBLIC_OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        from trend_pattern_engine import _publish_to_github
        _publish_to_github(data, PUBLIC_REPO_PATH, "更新產業價值鏈分類")
    except Exception as exc:
        print(f"產業價值鏈分類發布失敗：{exc}", file=sys.stderr)


def load(max_age_days: float | None = None) -> dict:
    """讀取分類檔；超過 max_age_days 或不存在就重爬（重爬失敗時沿用舊檔）。"""
    data = {}
    for path in (OUT, PUBLIC_OUT):   # 雲端沒有 data/ 下的檔案，讀發布到 GitHub 的那份
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            pass
    if max_age_days is not None:
        try:
            age = (datetime.now(ZoneInfo("Asia/Taipei")) - datetime.fromisoformat(data["generated_at"])).days
        except (KeyError, ValueError, TypeError):
            age = 10 ** 6
        if age > max_age_days:
            try:
                data = crawl(log=lambda _m: None)
            except Exception:
                pass
    return data


def stats(data: dict) -> dict:
    codes = {s for i in data.get("industries") or [] for c in i["chains"] for s in c["members"]}
    return {"industries": len(data.get("industries") or []),
            "chains": sum(len(i["chains"]) for i in data.get("industries") or []), "stocks": len(codes)}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    d = load() if "--stats" in sys.argv else crawl()
    print(json.dumps(stats(d), ensure_ascii=False))
