# Fetch daily OHLCV for the 17-symbol weather-regime panel from Yahoo Finance v8 chart API.
# Pure stdlib. End date frozen at 2026-07-26 for reproducibility. Fetched 2026-07-26.
import urllib.request as u, json, datetime as dt, os, time, csv
OUT="data/panel"; os.makedirs(OUT, exist_ok=True)
P2=int(dt.datetime(2026,7,27,tzinfo=dt.UTC).timestamp())   # include 26 Jul 2026
FETCH_DATE="2026-07-26"
PANEL=[
 ("US500","^GSPC","Index"),("US100","^NDX","Index"),("US2000","^RUT","Index"),("US30","^DJI","Index"),
 ("EURUSD","EURUSD=X","FX"),("GBPUSD","GBPUSD=X","FX"),("USDJPY","USDJPY=X","FX"),("AUDUSD","AUDUSD=X","FX"),
 ("NZDUSD","NZDUSD=X","FX"),("USDCHF","USDCHF=X","FX"),("USDCAD","USDCAD=X","FX"),
 ("BTCUSD","BTC-USD","Crypto"),("ETHUSD","ETH-USD","Crypto"),
 ("XAUUSD","GC=F","Commodity"),("XAGUSD","SI=F","Commodity"),("UKOIL","BZ=F","Commodity"),("USOIL","CL=F","Commodity"),
]
def fetch(tk):
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}?period1=0&period2={P2}&interval=1d"
    req=u.Request(url, headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    j=json.loads(u.urlopen(req,timeout=40).read()); r=j["chart"]["result"][0]
    ts=r["timestamp"]; q=r["indicators"]["quote"][0]
    o=q.get("open");h=q.get("high");l=q.get("low");c=q.get("close");v=q.get("volume")
    rows=[]
    for i,t in enumerate(ts):
        cl=c[i] if c else None
        if cl is None: continue
        d=dt.datetime.fromtimestamp(t,dt.UTC).date().isoformat()
        rows.append([d, o[i] if o else "", h[i] if h else "", l[i] if l else "", cl, v[i] if v else ""])
    return rows
man=[]
for name,tk,cls in PANEL:
    try:
        rows=fetch(tk)
        with open(f"{OUT}/{name}.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["Date","Open","High","Low","Close","Volume"]); w.writerows(rows)
        man.append((name,tk,cls,len(rows),rows[0][0],rows[-1][0]))
        print(f"{name:7s} {tk:9s} {cls:9s} rows={len(rows):6d}  {rows[0][0]} -> {rows[-1][0]}")
    except Exception as e:
        print(f"{name:7s} {tk:9s} FAIL {repr(e)[:80]}"); man.append((name,tk,cls,0,"",""))
    time.sleep(0.3)
with open(f"{OUT}/_manifest.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["symbol","yahoo_ticker","class","rows","start","end","fetch_date","source"])
    for m in man: w.writerow(list(m)+[FETCH_DATE,"Yahoo Finance v8 chart, interval=1d"])
print("DONE ->", OUT)
