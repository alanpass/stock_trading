"""名人投資人持股：機構大師抓 SEC 13F 最新申報，政治人物（川普、裴洛西）整理公開財產申報重點。

- 13F：data.sec.gov 免金鑰，需帶 User-Agent；每季結束後 45 天內申報，所以資料約有一季落後。
- 政治人物：OGE 278e／278-T（總統）與眾議院 STOCK Act 申報只有金額區間，沒有 API，這裡人工整理、附原始連結。
"""
from __future__ import annotations

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEC_HEADERS = {"User-Agent": "tw-stock-dashboard research a1113359@mail.nuk.edu.tw"}

# (CIK, 人物, 機構)
GURUS = [
    (1067983, "華倫・巴菲特 Warren Buffett", "Berkshire Hathaway"),
    (1336528, "比爾・艾克曼 Bill Ackman", "Pershing Square"),
    (1697748, "凱西・伍德 Cathie Wood", "ARK Investment"),
    (1536411, "史丹利・朱肯米勒 Stanley Druckenmiller", "Duquesne Family Office"),
    (1656456, "大衛・泰珀 David Tepper", "Appaloosa"),
    (1029160, "喬治・索羅斯 George Soros", "Soros Fund Management"),
    (1350694, "瑞・達利歐 Ray Dalio", "Bridgewater Associates"),
]

# 台股相關的美國掛牌標的（CUSIP 前 6 碼）
TAIWAN_RELATED = {"874039": "台積電 ADR", "45104G": "日月光 ADR", "904767": "聯電 ADR", "21871N": "中華電 ADR"}

# 政治人物：申報只有金額區間；以下整理自公開申報與報導，日期為申報涵蓋期間。
POLITICIANS = [
    {
        "name": "唐納・川普 Donald J. Trump",
        "role": "美國總統",
        "as_of": "2026 年度申報（2026-07-01）＋ 278-T 交易申報至 2026-07-31（2026-09-22 申報）",
        "summary": (
            "帳戶由第三方全權委託經理人操作，申報顯示交易極為頻繁：2025-01 至 2026-07 共約 3.1 萬筆交易"
            "（買進約 2.16 萬、賣出約 0.96 萬），以區間中值估計成交約 32 億美元。部位以市政債、美國公債、"
            "公司債與大型權值股為主；2026 年第一季明顯加碼科技股。"
        ),
        "highlights": [
            ("買進", "NVDA 輝達", "$1M–$5M", "2026-02-10"),
            ("買進", "AVGO 博通", "$1M–$5M", "2026-02-10"),
            ("買進", "NOW ServiceNow", "$1M–$5M", "2026-02-10"),
            ("買進", "BA 波音", "$1M–$5M", "2026-02-10"),
            ("買進", "AAPL 蘋果", "$1M–$5M", "2026-03-02"),
            ("買進", "VOO 標普 500 ETF", "$1M–$5M", "2026-03-02"),
            ("買進", "ORCL 甲骨文", "$1M–$5M", "2026-03-17"),
            ("買進", "MSFT 微軟、AMZN 亞馬遜", "$1M–$5M", "2026-03-19"),
            ("賣出", "美國國庫券（2026-04 到期）", "$500K–$5M", "2026-03"),
            ("買進", "AVGO、CSCO、TXN、XOM、HD、PG、VZ 等", "$250K–$500K", "2026-07"),
        ],
        "sources": [
            ("OGE 2026 年度財產申報（278e）", "https://extapps2.oge.gov/201/Presiden.nsf/PAS+Index/69AEAA9D7455ACD585258E27002DDEE1/$FILE/Donald-J-Trump-2026-278ANNUAL.pdf"),
            ("Open Cabinet 交易彙整", "https://open-cabinet.org/officials/trump-donald-j"),
        ],
    },
    {
        "name": "南西・裴洛西 Nancy Pelosi",
        "role": "美國眾議員（配偶 Paul Pelosi 操作）",
        "as_of": "眾議院 STOCK Act 交易申報，至 2026-08",
        "summary": (
            "交易多由配偶執行，常用深價內買權（LEAPS）布局大型科技股，是市場追蹤度最高的國會議員之一。"
            "已宣布不再競選 2026 年連任。"
        ),
        "highlights": [
            ("賣出", "AAPL 蘋果、NVDA 輝達（年底節稅）", "—", "2025-12"),
            ("買進", "INTC 英特爾", "$1M–$5M", "2026-05-29"),
            ("買進", "UBER Uber", "$500K–$1M", "2026-05-29"),
            ("買進", "BE Bloom Energy 5,000 股＋100 口買權（履約價 $100）", "$500K–$1M", "2026-07-28"),
        ],
        "sources": [
            ("眾議院財產申報查詢", "https://disclosures-clerk.house.gov/FinancialDisclosure"),
            ("GuruFocus 交易彙整", "https://gurufocus.com/politician/7/nancy-pelosi"),
        ],
    },
]


def _get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _latest_13f_filings(cik: int, n: int = 2) -> list[dict]:
    """最新 n 筆 13F-HR（不含修正案 13F-HR/A，修正案常只補部分持股）。"""
    data = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    r = data["filings"]["recent"]
    out = []
    for form, filed, acc, period in zip(r["form"], r["filingDate"], r["accessionNumber"], r["reportDate"]):
        if form == "13F-HR":
            out.append({"filed": filed, "accession": acc, "period": period})
            if len(out) >= n:
                break
    return out


def _parse_info_table(cik: int, accession: str) -> dict[str, dict]:
    """讀取 13F 資訊表，依 CUSIP 合併（同一檔常拆成多列）。value 自 2023 起為美元。"""
    folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
    items = json.loads(_get(f"{folder}/index.json"))["directory"]["item"]
    xml_name = next(
        (i["name"] for i in items if i["name"].lower().endswith(".xml") and i["name"].lower() != "primary_doc.xml"),
        None,
    )
    if not xml_name:
        return {}
    root = ET.fromstring(_get(f"{folder}/{xml_name}"))
    holdings: dict[str, dict] = {}
    for row in root.iter():
        if not row.tag.endswith("infoTable"):
            continue
        fields = {child.tag.split("}")[-1]: child for child in row}
        cusip = (fields.get("cusip").text or "").strip() if fields.get("cusip") is not None else ""
        if not cusip:
            continue
        put_call = fields.get("putCall")
        if put_call is not None and (put_call.text or "").strip():
            cusip = f"{cusip}-{put_call.text.strip().upper()}"
        shares = 0.0
        amt = fields.get("shrsOrPrnAmt")
        if amt is not None:
            for c in amt:
                if c.tag.endswith("sshPrnamt"):
                    shares = float(c.text or 0)
        item = holdings.setdefault(cusip, {
            "issuer": (fields["nameOfIssuer"].text or "").strip() if "nameOfIssuer" in fields else cusip,
            "title": (fields["titleOfClass"].text or "").strip() if "titleOfClass" in fields else "",
            "put_call": (put_call.text or "").strip().upper() if put_call is not None else "",
            "value": 0.0,
            "shares": 0.0,
        })
        item["value"] += float(fields["value"].text or 0) if "value" in fields else 0.0
        item["shares"] += shares
    # 少數機構仍沿用舊制以「千美元」申報：股票部位隱含股價中位數低於 1 美元就視為千元單位。
    prices = sorted(h["value"] / h["shares"] for h in holdings.values() if h["shares"] > 0 and not h["put_call"])
    if prices and prices[len(prices) // 2] < 1:
        for h in holdings.values():
            h["value"] *= 1000
    return holdings


def _change_label(cur: dict | None, prev: dict | None) -> str:
    if prev is None:
        return "新進"
    if cur is None:
        return "出清"
    if prev["shares"] <= 0:
        return "—"
    pct = (cur["shares"] - prev["shares"]) / prev["shares"] * 100
    if abs(pct) < 0.5:
        return "持平"
    return f"加碼 {pct:+.0f}%" if pct > 0 else f"減碼 {pct:+.0f}%"


def fetch_guru_portfolio(cik: int, person: str, firm: str, top_n: int = 10) -> dict:
    filings = _latest_13f_filings(cik)
    if not filings:
        return {"cik": cik, "person": person, "firm": firm, "error": "查無 13F-HR 申報"}
    cur = _parse_info_table(cik, filings[0]["accession"])
    prev = _parse_info_table(cik, filings[1]["accession"]) if len(filings) > 1 else {}
    total = sum(h["value"] for h in cur.values()) or 1.0
    rows = []
    for cusip, h in sorted(cur.items(), key=lambda kv: kv[1]["value"], reverse=True):
        rows.append({
            "cusip": cusip,
            "issuer": h["issuer"],
            "put_call": h["put_call"],
            "value": h["value"],
            "weight": h["value"] / total * 100,
            "shares": h["shares"],
            "change": _change_label(h, prev.get(cusip)) if prev else "—",
            "taiwan": TAIWAN_RELATED.get(cusip[:6], ""),
        })
    sold_out = [
        {"issuer": h["issuer"], "value": h["value"]}
        for cusip, h in sorted(prev.items(), key=lambda kv: kv[1]["value"], reverse=True)
        if cusip not in cur
    ]
    return {
        "cik": cik,
        "person": person,
        "firm": firm,
        "period": filings[0]["period"],
        "filed": filings[0]["filed"],
        "accession": filings[0]["accession"],
        "prev_period": filings[1]["period"] if len(filings) > 1 else "",
        "total_value": total,
        "positions": len(cur),
        "top": rows[:top_n],
        "taiwan_rows": [r for r in rows if r["taiwan"]],
        "new_buys": [r for r in rows if r["change"] == "新進"][:5],
        "sold_out": sold_out[:5],
        "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR",
    }


def fetch_all_gurus(cache_path: Path | None = None, max_age_hours: float = 12) -> dict:
    """抓全部大師持股；成功就寫本機快取，SEC 連不上時讀快取。"""
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if time.time() - float(cached.get("fetched_ts", 0)) < max_age_hours * 3600:
                return cached
        except Exception:
            cached = None
    else:
        cached = None

    def one(g):
        try:
            return fetch_guru_portfolio(*g)
        except Exception as exc:
            return {"cik": g[0], "person": g[1], "firm": g[2], "error": f"{type(exc).__name__}: {exc}"}

    # SEC 限制每秒 10 次請求；4 條執行緒、每位約 6 次請求，不會超過。
    with ThreadPoolExecutor(max_workers=4) as pool:
        gurus = list(pool.map(one, GURUS))

    result = {"fetched_ts": time.time(), "gurus": gurus}
    ok = [g for g in gurus if not g.get("error")]
    if cache_path and ok:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    if not ok and cached:
        cached["stale"] = True
        return cached
    return result


if __name__ == "__main__":
    data = fetch_all_gurus()
    for g in data["gurus"]:
        if g.get("error"):
            print(g["person"], "ERROR", g["error"])
            continue
        print(f"{g['person']}｜{g['period']}｜${g['total_value']/1e9:.1f}B｜{g['positions']} 檔")
        for r in g["top"][:5]:
            print(f"   {r['issuer'][:28]:28} {r['weight']:5.1f}%  {r['change']}  {r['taiwan']}")
