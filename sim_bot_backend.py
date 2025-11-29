import math
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import ccxt
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ==========================
# GENEL AYARLAR
# ==========================

# ccxt'de desteklenen herhangi bir borsa:
# "binance", "bybit", "okx", "kucoin", ...
EXCHANGE_ID = "binance"

# Sadece public endpoint kullanıyoruz -> API key YOK
exchange_class = getattr(ccxt, EXCHANGE_ID)
exchange = exchange_class({"enableRateLimit": True})

app = FastAPI(title="Sim Trading Bot Backend (Paper Trading)", version="1.0.0")


# ==========================
# VERİ MODELİ
# ==========================

class BotConfig(BaseModel):
    symbol: str = Field("BTC/USDT", description="İşlem yapılacak parite (ör: BTC/USDT)")
    start_usdt: float = Field(1000.0, description="Başlangıç USDT bakiyesi")
    grid_width_pct: float = Field(0.05, description="Merkez fiyatın +/- yüzdesi (örn: 0.05 = %5)")
    num_grids: int = Field(10, description="Grid basamak sayısı")
    usdt_per_order: float = Field(50.0, description="Her seviyede kullanılacak USDT miktarı")
    take_profit_pct: float = Field(0.006, description="Her işlem için kâr marjı (örn: 0.006 = %0.6)")
    stop_loss_pct: float = Field(0.01, description="Her işlem için zarar kesme oranı (örn: 0.01 = %1)")
    poll_interval_sec: int = Field(5, description="Fiyat kontrol aralığı (saniye)")


class GridLevelState(BaseModel):
    level_price: float
    holding: bool
    entry_price: Optional[float]
    amount: float


class BotState(BaseModel):
    running: bool
    symbol: str
    usdt_balance: float
    coin_balance: float
    realized_pnl_usdt: float
    last_price: Optional[float]
    last_update_utc: Optional[str]
    config: BotConfig
    grid_levels: List[GridLevelState]


# ==========================
# GRID BOT IMPLEMENTASYONU
# ==========================

class GridLevel:
    """Her grid seviyesi için dahili durum."""

    def __init__(self, level_price: float):
        self.level_price = level_price
        self.holding: bool = False
        self.entry_price: Optional[float] = None
        self.amount: float = 0.0

    def to_state(self) -> GridLevelState:
        return GridLevelState(
            level_price=self.level_price,
            holding=self.holding,
            entry_price=self.entry_price,
            amount=self.amount,
        )


class SimGridBot:
    """
    Gerçek emir YOK, sadece simülasyon.
    - Belirlenen sembol için public API'den canlı fiyat çeker.
    - Grid seviyelerine göre:
        * Fiyat grid seviyesinin altına inerse -> AL (simüle)
        * Fiyat entry * (1 + take_profit_pct) üzerine çıkarsa -> TP ile SAT (simüle)
        * Fiyat entry * (1 - stop_loss_pct) altına inerse -> SL ile SAT (simüle)
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.symbol = config.symbol
        self.usdt_balance: float = config.start_usdt
        self.coin_balance: float = 0.0
        self.realized_pnl_usdt: float = 0.0

        self.levels: List[GridLevel] = []
        self.last_price: Optional[float] = None
        self.last_update_utc: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False

        # Precision bilgisi
        markets = exchange.load_markets()
        if self.symbol not in markets:
            raise ValueError(f"{EXCHANGE_ID} üzerinde {self.symbol} marketi bulunamadı.")

        market = markets[self.symbol]
        self.amount_prec = market.get("precision", {}).get("amount", 6)
        self.price_prec = market.get("precision", {}).get("price", 2)
        self.min_amount = market.get("limits", {}).get("amount", {}).get("min", 0.0)

    # ---------- Yardımcı ----------

    def _round_amount(self, amount: float) -> float:
        factor = 10 ** self.amount_prec
        return math.floor(amount * factor) / factor

    def _round_price(self, price: float) -> float:
        factor = 10 ** self.price_prec
        return math.floor(price * factor) / factor

    def _fetch_price(self) -> float:
        ticker = exchange.fetch_ticker(self.symbol)
        return float(ticker["last"])

    def _build_grid(self):
        center = self._fetch_price()
        low = center * (1 - self.config.grid_width_pct)
        high = center * (1 + self.config.grid_width_pct)
        step = (high - low) / self.config.num_grids

        levels: List[GridLevel] = []
        for i in range(self.config.num_grids + 1):
            lvl_price = low + i * step
            lvl_price = self._round_price(lvl_price)
            levels.append(GridLevel(lvl_price))

        self.levels = levels
        self.last_price = center
        self.last_update_utc = datetime.utcnow().isoformat()
        print(f"[GRID] {self.symbol} grid oluşturuldu. Merkez: {center:.4f}")
        for lvl in levels:
            print(f"  - Level: {lvl.level_price:.4f}")

    # ---------- Al / Sat simülasyon ----------

    def _handle_buy(self, level: GridLevel, price: float):
        if level.holding:
            return

        # Bu seviyeden ne kadar alacağız?
        size_usdt = min(self.config.usdt_per_order, self.usdt_balance)
        if size_usdt < 5:  # minimum bir sınır
            return

        raw_amount = size_usdt / price
        amount = self._round_amount(raw_amount)
        if amount <= 0 or amount < self.min_amount:
            return

        cost = amount * price
        # Fee'yi simülasyon için kaba tahmin (isteğe göre 0 da yapabilirsin)
        fee = cost * 0.001

        self.usdt_balance -= (cost + fee)
        self.coin_balance += amount

        level.holding = True
        level.entry_price = price
        level.amount = amount

        print(
            f"[BUY] {self.symbol} | Fiyat={price:.4f}, amount={amount:.6f}, "
            f"USDT_kalan={self.usdt_balance:.2f}"
        )

    def _handle_sell(self, level: GridLevel, price: float, reason: str):
        if not level.holding or level.amount <= 0:
            return

        entry = level.entry_price
        amount = level.amount
        if entry is None:
            return

        gross_buy = entry * amount
        gross_sell = price * amount
        buy_fee = gross_buy * 0.001
        sell_fee = gross_sell * 0.001

        pnl = (gross_sell - sell_fee) - (gross_buy + buy_fee)
        self.realized_pnl_usdt += pnl
        self.usdt_balance += (gross_sell - sell_fee)
        self.coin_balance -= amount

        print(
            f"[SELL-{reason}] {self.symbol} | Entry={entry:.4f}, Price={price:.4f}, "
            f"amount={amount:.6f}, TradePnL={pnl:.4f}, TotalPnL={self.realized_pnl_usdt:.4f}"
        )

        # Level sıfırla
        level.holding = False
        level.entry_price = None
        level.amount = 0.0

    # ---------- Ana loop ----------

    def _loop(self):
        self._build_grid()
        self._running = True
        print(f"[BOT] {self.symbol} için simülasyon başlatıldı.")

        while not self._stop_event.is_set():
            try:
                price = self._fetch_price()
                now_utc = datetime.utcnow().isoformat()

                with self._lock:
                    self.last_price = price
                    self.last_update_utc = now_utc

                    # Her level için BUY / SELL koşullarını kontrol et
                    for level in self.levels:
                        # BUY: fiyat bu grid seviyesine eşit/altına indiğinde
                        if (not level.holding) and price <= level.level_price:
                            self._handle_buy(level, price)

                        # SELL: pozisyon varsa TP/SL
                        if level.holding and level.entry_price is not None:
                            tp_level = level.entry_price * (1 + self.config.take_profit_pct)
                            sl_level = level.entry_price * (1 - self.config.stop_loss_pct)

                            if price >= tp_level:
                                self._handle_sell(level, price, reason="TP")
                            elif price <= sl_level:
                                self._handle_sell(level, price, reason="SL")

                time.sleep(self.config.poll_interval_sec)

            except Exception as e:
                print(f"[ERROR] Loop hatası: {e}")
                time.sleep(3)

        self._running = False
        print(f"[BOT] {self.symbol} için simülasyon durduruldu.")

    def start(self):
        if self._thread and self._thread.is_alive():
            print("[BOT] Zaten çalışıyor.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        return self._running

    def get_state(self) -> BotState:
        with self._lock:
            return BotState(
                running=self._running,
                symbol=self.symbol,
                usdt_balance=self.usdt_balance,
                coin_balance=self.coin_balance,
                realized_pnl_usdt=self.realized_pnl_usdt,
                last_price=self.last_price,
                last_update_utc=self.last_update_utc,
                config=self.config,
                grid_levels=[lvl.to_state() for lvl in self.levels],
            )


# ==========================
# GLOBAL BOT INSTANCE
# ==========================

bot_instance: Optional[SimGridBot] = None


# ==========================
# API ENDPOINT'LERİ
# ==========================

@app.get("/prices")
def get_all_prices():
    """
    Tüm market fiyatlarını döndürür.
    Mobil uygulama buradan liste çekip, kullanıcıya coin seçtirebilir.
    """
    tickers = exchange.fetch_tickers()
    result = {}
    for sym, t in tickers.items():
        try:
            result[sym] = float(t["last"])
        except Exception:
            continue
    return result


@app.post("/bot/start", response_model=BotState)
def start_bot(config: BotConfig):
    """
    Simülasyon botunu verilen parametrelerle başlat.
    Eğer daha önce çalışan bir bot varsa durdurup yenisini başlatır.
    """
    global bot_instance

    # Önce varsa eski botu durdur
    if bot_instance is not None:
        bot_instance.stop()

    bot_instance = SimGridBot(config)
    bot_instance.start()
    # Grid hemen oluşsun diye biraz bekleyelim
    time.sleep(1)

    return bot_instance.get_state()


@app.get("/bot/status", response_model=BotState)
def bot_status():
    """
    Botun anlık durumunu döndürür.
    - running
    - usdt_balance
    - coin_balance
    - realized_pnl
    - last_price
    - grid seviyeleri vs.
    """
    if bot_instance is None:
        raise RuntimeError("Bot başlatılmamış. Önce /bot/start çağır.")

    return bot_instance.get_state()


@app.post("/bot/stop")
def stop_bot():
    """
    Botu durdurur.
    """
    global bot_instance
    if bot_instance is None:
        return {"status": "no_bot", "message": "Zaten aktif bot yok."}
    bot_instance.stop()
    state = bot_instance.get_state()
    bot_instance = None
    return {"status": "stopped", "final_state": state}

