import os, io, base64
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, request
from anthropic import Anthropic
import yfinance as yf
from web_config import WebConfig

app = Flask(__name__)
cfg = WebConfig()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", cfg.ANTHROPIC_API_KEY))

SYMBOL_MAP = {
    "XAUUSD": {"yf": "GC=F",    "label": "Gold / USD"},
    "BTCUSD": {"yf": "BTC-USD", "label": "Bitcoin / USD"},
}
TF_MAP = {
    "M1":  {"interval": "1m",  "period": "1d",  "label": "1分足"},
    "M5":  {"interval": "5m",  "period": "5d",  "label": "5分足"},
    "M15": {"interval": "15m", "period": "5d",  "label": "15分足"},
    "M30": {"interval": "30m", "period": "10d", "label": "30分足"},
    "H4":  {"interval": "1h",  "period": "60d", "label": "4時間足"},
}

def fetch_ohlcv(symbol_key, tf_key):
    yf_sym = SYMBOL_MAP[symbol_key]["yf"]
    tf = TF_MAP[tf_key]
    df = yf.Ticker(yf_sym).history(interval=tf["interval"], period=tf["period"])
    if df.empty:
        raise RuntimeError(f"データ取得失敗: {yf_sym}")
    return df[["Open","High","Low","Close","Volume"]].tail(150)

def add_indicators(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df = df.copy()
    df["RSI"] = 100 - 100/(1+gain/loss)
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    ma20 = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["BB_mid"] = ma20
    df["BB_upper"] = ma20 + 2*std
    df["BB_lower"] = ma20 - 2*std
    df["MA20"] = ma20
    df["MA200"] = c.rolling(200).mean()
    return df

def render_chart_png(df, symbol, tf_key):
    df = add_indicators(df).dropna(subset=["BB_mid"])
    rsi_val = df["RSI"].iloc[-1]
    atr_val = df["ATR"].iloc[-1]
    close = df["Close"].iloc[-1]
    apds = [
        mpf.make_addplot(df["BB_upper"],color="gray",width=0.8,linestyle="--"),
        mpf.make_addplot(df["BB_lower"],color="gray",width=0.8,linestyle="--"),
        mpf.make_addplot(df["BB_mid"],color="white",width=0.8),
        mpf.make_addplot(df["MA20"],color="cyan",width=1.0),
    ]
    if df["MA200"].notna().any():
        apds.append(mpf.make_addplot(df["MA200"],color="red",width=1.2))
    fig = mpf.figure(style="nightclouds",figsize=(10,7))
    ax_main = fig.add_axes([0.06,0.40,0.90,0.55])
    ax_rsi  = fig.add_axes([0.06,0.22,0.90,0.16])
    ax_atr  = fig.add_axes([0.06,0.05,0.90,0.14])
    mpf.plot(df,type="candle",ax=ax_main,addplot=apds,volume=False,
             axtitle=f"{symbol} {TF_MAP[tf_key]['label']}  現在値:{close:.2f}  RSI:{rsi_val:.1f}  ATR:{atr_val:.2f}",
             returnfig=False)
    ax_rsi.plot(df.index,df["RSI"],color="#4488ff",linewidth=1)
    ax_rsi.axhline(70,color="red",linestyle="--",linewidth=0.7)
    ax_rsi.axhline(30,color="green",linestyle="--",linewidth=0.7)
    ax_rsi.set_ylim(0,100); ax_rsi.set_ylabel("RSI",color="white",fontsize=8)
    ax_rsi.tick_params(colors="white",labelsize=7); ax_rsi.set_facecolor("#0d0d0d")
    ax_atr.plot(df.index,df["ATR"],color="#ff44aa",linewidth=1)
    ax_atr.set_ylabel("ATR",color="white",fontsize=8)
    ax_atr.tick_params(colors="white",labelsize=7); ax_atr.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")
    buf = io.BytesIO()
    plt.savefig(buf,format="png",dpi=110,bbox_inches="tight",facecolor="#0d0d0d")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

SYSTEM_PROMPT = """あなたはプロのFXトレーダー兼テクニカルアナリストです。
XAUUSDまたはBTCUSDのマルチタイムフレームチャートを分析し、具体的なトレード指示を日本語で出力してください。

## 出力フォーマット（必ずMarkdownで）

### 📊 マルチタイムフレーム分析
各時間足（M1/M5/M15/M30/H4）のトレンド・RSI・BBの状態を簡潔に

### 🎯 エントリー判断
- **方向**: ロング / ショート / 見送り
- **根拠**: 最重要理由を3つ箇条書き

### 📍 注文設定
| 項目 | 価格 |
|---|---|
| エントリー（指値） | |
| ストップロス (SL) | |
| テイクプロフィット1 (TP1) | |
| テイクプロフィット2 (TP2) | |
| リスクリワード比 | 1:○.○ |

### ⚠️ 無効化条件
このシナリオが崩れる条件

### 📈 一言サマリー
"""

def analyze(symbol_key, img_b64_list):
    content = [{"type":"text","text":f"{symbol_key} のM1/M5/M15/M30/H4チャートです。トレード分析をしてください。"}]
    labels = ["M1","M5","M15","M30","H4"]
    for i,b64 in enumerate(img_b64_list):
        lbl = labels[i] if i < len(labels) else f"TF{i+1}"
        content.append({"type":"text","text":f"\n--- {lbl} ---"})
        content.append({"type":"image","source":{"type":"base64","media_type":"image/png","data":b64}})
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user","content":content}]
    )
    return msg.content[0].text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    symbol_key = request.json.get("symbol","XAUUSD").upper()
    if symbol_key not in SYMBOL_MAP:
        return jsonify({"error":f"不明なシンボル: {symbol_key}"}),400
    charts, errors = [], []
    for tf_key in ["M1","M5","M15","M30","H4"]:
        try:
            df = fetch_ohlcv(symbol_key, tf_key)
            b64 = render_chart_png(df, symbol_key, tf_key)
            charts.append({"tf":tf_key,"label":TF_MAP[tf_key]["label"],"img":b64})
        except Exception as e:
            errors.append(f"{tf_key}: {e}")
    if len(charts) < 2:
        return jsonify({"error":"チャート取得失敗: "+", ".join(errors)}),500
    analysis = analyze(symbol_key, [c["img"] for c in charts])
    return jsonify({
        "symbol":symbol_key,
        "label":SYMBOL_MAP[symbol_key]["label"],
        "timestamp":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "charts":charts,
        "analysis":analysis,
        "errors":errors,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False)
