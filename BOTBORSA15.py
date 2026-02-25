#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOTBORSA15.py
Borsa İstanbul (BIST) Teknik Analiz Telegram Botu (Yalnızca yfinance)

Özellikler:
- BIST hisseleri için (GARAN, SISE, THYAO...) Yahoo Finance üzerinden veri (yfinance)
- RSI, MACD, Bollinger, SMA 20/50, EMA 200, ATR, ADX
- Destek/Direnç + Basit AL/SAT/BEKLE sinyali
- Grafik çıktısı (PNG) + Telegram'a gönderim

Kurulum:
  pip install python-telegram-bot==20.* yfinance pandas numpy matplotlib

Çalıştırma:
  export TELEGRAM_BOT_TOKEN="TOKENIN"
  python BOTBORSA15.py

Not:
- BIST hisseleri yfinance'ta .IS uzantısıyla bulunur (örn: SISE.IS).
- Bu botta kullanıcı "SISE" yazarsa otomatik "SISE.IS" kullanılır.
"""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)

# ===================== AYARLAR =====================

POPULAR_STOCKS = ["GARAN", "THYAO", "SISE", "EREGL", "BIMAS", "AKBNK", "ISCTR", "KCHOL", "SAHOL", "TOASO"]

# Grafik çıktıları için klasör
OUTPUT_DIR = os.environ.get("BOTBORSA_OUTPUT_DIR")
if not OUTPUT_DIR:
    OUTPUT_DIR = os.path.join(os.getcwd(), "botborsa_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== VERİ ÇEKME =====================

def _to_yf_ticker(stock_code: str) -> str:
    code = stock_code.strip().upper()
    # Kullanıcı "SISE.IS" yazarsa elleme, "SISE" yazarsa ".IS" ekle
    if code.endswith(".IS"):
        return code
    # Bazıları "SISE:IS" vb yazabilir, normalize et
    code = code.replace(":IS", "").replace(".TR", "").replace(".TI", "")
    return f"{code}.IS"

def fetch_from_yfinance(stock_code: str, period: str = "6mo", interval: str = "1d", max_retries: int = 3):
    """
    yfinance ile veri çek.
    """
    try:
        import yfinance as yf
    except Exception as e:
        return None, f"yfinance import edilemedi: {e}"

    ticker = _to_yf_ticker(stock_code)

    last_err = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(1.5)

            obj = yf.Ticker(ticker)
            df = obj.history(period=period, interval=interval, auto_adjust=False)

            if df is None or df.empty:
                last_err = "Boş veri döndü"
                continue

            # Bazı ortamlarda MultiIndex gelebiliyor
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standart isimler
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(set(df.columns)):
                last_err = f"Beklenen kolonlar yok: {df.columns}"
                continue

            df = df.dropna()
            if len(df) < 60:
                # 200 EMA için biraz daha fazla veri iyi olur ama minimumu düşük tutuyoruz
                # Yine de analizin sağlıklı olabilmesi için >= 30
                if len(df) < 30:
                    last_err = "Yeterli veri yok (en az 30 gün gerekli)"
                    continue

            return df, None

        except Exception as e:
            last_err = str(e)

    return None, f"yfinance hata: {last_err or 'bilinmeyen'}"


# ===================== İNDİKATÖRLER =====================

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = data["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
    sma = data["Close"].rolling(window=period).mean()
    std = data["Close"].rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
    return data["Close"].rolling(window=period).mean()

def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
    return data["Close"].ewm(span=period, adjust=False).mean()

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.fillna(method="bfill")

def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = data["High"].diff()
    down_move = -data["Low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (data["High"] - data["Low"]),
        (data["High"] - data["Close"].shift()).abs(),
        (data["Low"] - data["Close"].shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=data.index).ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=data.index).ewm(alpha=1/period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period).mean()
    return adx.fillna(method="bfill")

def find_support_resistance(data: pd.DataFrame, lookback: int = 80):
    recent = data.tail(lookback).copy()
    w = 5
    highs = []
    lows = []

    for i in range(w, len(recent) - w):
        if (recent["High"].iloc[i] >= recent["High"].iloc[i-w:i]).all() and (recent["High"].iloc[i] >= recent["High"].iloc[i+1:i+w+1]).all():
            highs.append(recent["High"].iloc[i])
        if (recent["Low"].iloc[i] <= recent["Low"].iloc[i-w:i]).all() and (recent["Low"].iloc[i] <= recent["Low"].iloc[i+1:i+w+1]).all():
            lows.append(recent["Low"].iloc[i])

    current = float(data["Close"].iloc[-1])
    resistances = sorted([h for h in highs if h > current])
    supports = sorted([l for l in lows if l < current], reverse=True)

    r1 = resistances[0] if len(resistances) > 0 else current * 1.05
    r2 = resistances[1] if len(resistances) > 1 else r1 * 1.03
    s1 = supports[0] if len(supports) > 0 else current * 0.95
    s2 = supports[1] if len(supports) > 1 else s1 * 0.97
    return s1, s2, r1, r2

def generate_signal(data: pd.DataFrame):
    rsi = data["RSI"].iloc[-1]
    macd = data["MACD"]
    sig = data["MACD_Signal"]

    prev_macd, prev_sig = macd.iloc[-2], sig.iloc[-2]
    curr_macd, curr_sig = macd.iloc[-1], sig.iloc[-1]

    macd_cross_up = (prev_macd < prev_sig) and (curr_macd > curr_sig)
    macd_cross_down = (prev_macd > prev_sig) and (curr_macd < curr_sig)

    signal = "BEKLE"
    emoji = "⏳"

    if rsi < 30:
        signal, emoji = "AL", "🟢"
    elif rsi > 70:
        signal, emoji = "SAT", "🔴"

    if macd_cross_up:
        signal, emoji = ("GÜÇLÜ AL" if signal == "AL" else "AL (MACD)"), "🟢"
    elif macd_cross_down:
        signal, emoji = ("GÜÇLÜ SAT" if signal == "SAT" else "SAT (MACD)"), "🔴"

    return signal, emoji, float(rsi), macd_cross_up, macd_cross_down


# ===================== GRAFİK =====================

def create_chart(data: pd.DataFrame, stock_code: str, s1, s2, r1, r2, stop, target, signal: str):
    df = data.tail(90).copy()
    df = df.reset_index()

    fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0d1117")
    ax1.tick_params(colors="white", labelsize=9)
    for sp in ax1.spines.values():
        sp.set_color("#30363d")

    # Mum
    for i in range(len(df)):
        o = float(df.loc[i, "Open"])
        h = float(df.loc[i, "High"])
        l = float(df.loc[i, "Low"])
        c = float(df.loc[i, "Close"])
        up = c >= o
        color = "#3fb950" if up else "#f85149"
        rect = Rectangle((i - 0.35, min(o, c)), 0.7, abs(c - o), facecolor=color, edgecolor=color, alpha=0.9, linewidth=1)
        ax1.add_patch(rect)
        ax1.plot([i, i], [l, h], color=color, linewidth=1.2)

    ax1.plot(df.index, df["SMA20"], linewidth=1.6, label="SMA 20", alpha=0.9)
    ax1.plot(df.index, df["SMA50"], linewidth=1.6, label="SMA 50", alpha=0.9)
    ax1.plot(df.index, df["EMA200"], linewidth=2.3, label="EMA 200", alpha=0.9)

    # Bollinger
    ax1.plot(df.index, df["BB_Upper"], linewidth=1, linestyle="--", alpha=0.5)
    ax1.plot(df.index, df["BB_Middle"], linewidth=1, alpha=0.25)
    ax1.plot(df.index, df["BB_Lower"], linewidth=1, linestyle="--", alpha=0.5)
    ax1.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.04)

    current_price = float(df["Close"].iloc[-1])

    # seviyeler
    ax1.axhline(r2, linestyle="--", linewidth=2, alpha=0.7)
    ax1.axhline(r1, linestyle="-", linewidth=2.5, alpha=0.9)
    ax1.axhline(s1, linestyle="-", linewidth=2.5, alpha=0.9)
    ax1.axhline(s2, linestyle="--", linewidth=2, alpha=0.7)
    ax1.axhline(current_price, linestyle="-", linewidth=2, alpha=0.9)
    ax1.axhline(stop, linestyle="-.", linewidth=2, alpha=0.85)
    ax1.axhline(target, linestyle="-.", linewidth=2, alpha=0.85)

    ax1.set_title(f"{stock_code} - Teknik Analiz | Sinyal: {signal}", color="white", fontsize=15, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.15)
    ax1.set_xlim(-2, len(df) + 10)
    ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9, framealpha=0.9)
    ax1.set_xticks([])

    # Hacim
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor("#0d1117")
    ax2.tick_params(colors="white", labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color("#30363d")

    colors_vol = ["#3fb950" if df.loc[i, "Close"] >= df.loc[i, "Open"] else "#f85149" for i in range(len(df))]
    ax2.bar(df.index, df["Volume"].values / 1e6, alpha=0.7, width=0.8, color=colors_vol)
    ax2.grid(True, alpha=0.15)
    ax2.set_ylabel("Hacim (M)", color="white", fontsize=9)
    ax2.set_xticks([])

    # RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor("#0d1117")
    ax3.tick_params(colors="white", labelsize=8)
    for sp in ax3.spines.values():
        sp.set_color("#30363d")

    ax3.plot(df.index, df["RSI"], linewidth=1.7, label="RSI")
    ax3.axhline(70, linestyle="--", linewidth=1.3, alpha=0.8)
    ax3.axhline(30, linestyle="--", linewidth=1.3, alpha=0.8)
    ax3.axhline(50, linestyle="-", linewidth=0.6, alpha=0.5)
    ax3.fill_between(df.index, 30, 70, alpha=0.05)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.15)
    ax3.set_ylabel("RSI (14)", color="white", fontsize=9)

    # X etiketleri
    n_ticks = 6
    tick_positions = np.linspace(0, len(df)-1, n_ticks, dtype=int)
    labels = []
    for p in tick_positions:
        try:
            dt = df.loc[p, "Date"]
            if hasattr(dt, "strftime"):
                labels.append(dt.strftime("%d.%m"))
            else:
                labels.append(str(dt)[:10])
        except Exception:
            labels.append("")
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(labels, color="white", fontsize=8)

    fig.text(
        0.5, 0.01,
        f"Güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        ha="center", va="bottom", fontsize=10, color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#30363d", alpha=0.95, linewidth=1.5)
    )

    out_path = os.path.join(OUTPUT_DIR, f"{stock_code}_{int(time.time())}.png")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117", edgecolor="none", pad_inches=0.25)
    plt.close(fig)
    return out_path


# ===================== ANALİZ =====================

def analyze_stock(stock_code: str):
    df, err = fetch_from_yfinance(stock_code)
    if df is None:
        return None, f"❌ {stock_code}: Veri çekilemedi!\n\nHata: {err}\n\n💡 Örnek: GARAN, THYAO, SISE, EREGL, BIMAS"

    # İndikatörler
    df = df.copy()
    df["RSI"] = calculate_rsi(df)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(df)
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = calculate_bollinger_bands(df)
    df["SMA20"] = calculate_sma(df, 20)
    df["SMA50"] = calculate_sma(df, 50)
    df["EMA200"] = calculate_ema(df, 200)
    df["ATR"] = calculate_atr(df)
    df["ADX"] = calculate_adx(df)

    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2])
    daily_change = ((current_price - prev_price) / prev_price) * 100.0

    s1, s2, r1, r2 = find_support_resistance(df)
    signal, emoji, rsi_val, macd_up, macd_down = generate_signal(df)

    atr = float(df["ATR"].iloc[-1])
    adx = float(df["ADX"].iloc[-1])

    # stop/target (basit ATR)
    if emoji == "🔴":
        stop = current_price + 2 * atr
        target = current_price - 3 * atr
    else:
        stop = current_price - 2 * atr
        target = current_price + 3 * atr

    # trend
    sma20 = float(df["SMA20"].iloc[-1])
    sma50 = float(df["SMA50"].iloc[-1])
    if sma20 > sma50:
        trend = "Yükselen"
    elif sma20 < sma50:
        trend = "Düşen"
    else:
        trend = "Yatay"

    chart_path = create_chart(df, stock_code.upper().strip().replace(".IS",""), s1, s2, r1, r2, stop, target, f"{emoji} {signal}")

    report = f"""📊 *{stock_code.upper().strip().replace('.IS','')} TEKNİK ANALİZ RAPORU*

📡 Veri Kaynağı: `YFINANCE (Yahoo)`

💰 *FİYAT*
• Güncel: `{current_price:.2f} TL`
• Günlük: `{'+' if daily_change >= 0 else ''}{daily_change:.2f}%`

📈 *İNDİKATÖRLER*
• RSI (14): `{rsi_val:.2f}` {'⬇️ Aşırı Satım' if rsi_val < 30 else '⬆️ Aşırı Alım' if rsi_val > 70 else '⚖️ Nötr'}
• MACD: `{'🟢 Yukarı Kesişim' if macd_up else '🔴 Aşağı Kesişim' if macd_down else '⏳ Beklemede'}`
• ADX (14): `{adx:.2f}` {'💪 Güçlü Trend' if adx > 25 else '😴 Zayıf Trend'}
• ATR (14): `{atr:.2f}`

📉 *DESTEK / DİRENÇ*
• R2: `{r2:.2f}`
• R1: `{r1:.2f}`
• S1: `{s1:.2f}`
• S2: `{s2:.2f}`

🎯 *TREND*
• `{trend}`

⚡ *SİNYAL*
• `{emoji} {signal}`

💡 *İŞLEM SENARYOSU (ATR)*
• Giriş: `{current_price:.2f}`
• Stop: `{stop:.2f}` (Risk: {abs((stop-current_price)/current_price*100):.1f}%)
• Hedef: `{target:.2f}` (Potansiyel: {abs((target-current_price)/current_price*100):.1f}%)

⚠️ *Uyarı*: Yatırım tavsiyesi değildir."""
    return chart_path, report


# ===================== TELEGRAM =====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """🤖 *BOTBORSA15 - BIST Teknik Analiz Botu*

✅ Veri kaynağı: *yfinance (Yahoo)*

📝 *Kullanım:*
• Hisse kodunu yaz: `SISE`, `GARAN`, `THYAO`
• /grafik SISE - Sadece grafik
• /analiz GARAN - Grafik + rapor
• /populer - Popüler hisseler

⚠️ Yatırım tavsiyesi değildir."""
    kb = [
        [InlineKeyboardButton("🔥 Popüler Hisseler", callback_data="popular")],
        [InlineKeyboardButton("❓ Yardım", callback_data="help")]
    ]
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """❓ *YARDIM*

• /start - Başlat
• /help - Yardım
• /grafik [KOD] - Grafik
• /analiz [KOD] - Grafik + detay rapor
• /populer - Popüler hisseler

📌 Örnek: `SISE`, `GARAN`, `THYAO`, `EREGL`, `BIMAS`

Not: BIST hisselerinde yfinance otomatik `.IS` ekler.
⚠️ Yatırım tavsiyesi değildir."""
    await update.message.reply_text(text, parse_mode="Markdown")

async def grafik_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Hisse kodu yaz.\nÖrnek: `/grafik SISE`", parse_mode="Markdown")
        return

    code = context.args[0].strip().upper()
    msg = await update.message.reply_text(f"⏳ {code} grafiği hazırlanıyor...")
    chart_path, report_or_err = analyze_stock(code)
    await msg.delete()

    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            await update.message.reply_photo(photo=f, caption=f"📊 {code} Grafiği", parse_mode="Markdown")
    else:
        await update.message.reply_text(report_or_err)

async def analiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Hisse kodu yaz.\nÖrnek: `/analiz GARAN`", parse_mode="Markdown")
        return

    code = context.args[0].strip().upper()
    msg = await update.message.reply_text(f"⏳ {code} analiz ediliyor...")
    chart_path, report_or_err = analyze_stock(code)
    await msg.delete()

    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            await update.message.reply_photo(photo=f, caption=report_or_err, parse_mode="Markdown")
    else:
        await update.message.reply_text(report_or_err)

async def populer_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Popüler hisseler taranıyor...")
    results = []
    for code in POPULAR_STOCKS[:5]:
        try:
            chart_path, report = analyze_stock(code)
            if chart_path:
                results.append((code, chart_path, report))
        except Exception:
            pass
        time.sleep(0.8)

    await msg.delete()

    if not results:
        await update.message.reply_text("❌ Veri çekilemedi. İnternet/yfinance kontrol et.")
        return

    summary = "📌 *POPÜLER HİSSELER (ÖZET)*\n\n"
    for code, _, report in results:
        lines = report.splitlines()
        price = next((l for l in lines if "Güncel" in l), None)
        sig = next((l for l in lines if "⚡" in l), None)
        if price:
            summary += f"*{code}*\n{price}\n\n"
        else:
            summary += f"*{code}*\n\n"

    await update.message.reply_text(summary, parse_mode="Markdown")

    # İlk hissenin grafiğini de gönder
    code, chart_path, report = results[0]
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            await update.message.reply_photo(photo=f, caption=report, parse_mode="Markdown")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    code = (update.message.text or "").strip().upper()
    if not code:
        return
    msg = await update.message.reply_text(f"⏳ {code} analiz ediliyor...")
    chart_path, report_or_err = analyze_stock(code)
    await msg.delete()

    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            await update.message.reply_photo(photo=f, caption=report_or_err, parse_mode="Markdown")
    else:
        await update.message.reply_text(report_or_err)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "popular":
        text = "📈 *POPÜLER HİSSELER*\n\n" + "\n".join([f"• `{x}`" for x in POPULAR_STOCKS]) + "\n\nHisse kodunu yazıp analiz edebilirsin."
        await q.edit_message_text(text, parse_mode="Markdown")
    elif q.data == "help":
        await q.edit_message_text(
            "1) Hisse yaz: `SISE`\n2) veya /analiz SISE\n3) /grafik SISE\n\n⚠️ Yatırım tavsiyesi değildir.",
            parse_mode="Markdown"
        )

def main():
    token = "8632592009:AAHXTFu0XHy0jIJdZP9BUlTKUXf3eDsDoYs"
    
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("grafik", grafik_command))
    app.add_handler(CommandHandler("analiz", analiz_command))
    app.add_handler(CommandHandler("populer", populer_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 BOTBORSA15 çalışıyor...")
    print(f"📁 Grafik klasörü: {OUTPUT_DIR}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
