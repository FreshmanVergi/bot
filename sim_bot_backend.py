import math
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================
# COINGECKO AYARLARI
# ==========================

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

app = FastAPI(
    title="Sim Trading Bot Backend (CoinGecko, No API Key)",
    version="1.0.0"
)


# ==========================
# VERİ MODELLERİ
# ==========================

class BotConfig(BaseModel):
    symbol: str = Field("BTC/USDT", description="İşlem yapılacak parite (BTC/USDT, BTCUSDT, BTC vs.)")
    start_usdt: float = Field(1000.0, description="Başlangıç USDT (USD) bakiyesi")
    grid_width_pct: float = Field(0.05, description="Merkez fiyatın +/- yüzdesi (örn: 0.05 = %5)")
    num_grids: int = Field(10, description="Grid basamak sayısı")
    usdt_per_order: float = Field(50.0, description="Her seviyede kullanılacak USDT miktarı")
    take_profit_pct: float = Field(0.006, description="Her işlem için kâr marjı (0.006 = %0.6)")
    stop_loss_pct: float = Field(0.01, description="Her işlem için zarar kesme oranı (0.01 = %1)")
    poll_interval_sec: int = Field(5, description="Fiyat kontrol aralığı (saniye)")


class GridLevelState(BaseModel):
    level_price: float
    holding: bool
    entry_price: Optional[float]
    amount: float


class BotState(BaseModel):
    running: bool
    symbol: str
    base_symbol: str
    coingecko_id: str
    usdt_balance: float
    coin_balance: float
    realized_pnl_usdt: float
    last_price: Optional[float]
    last_update_utc: Optional[str]
    config: BotConfig
    grid_levels: List[GridLevelState]


# ==========================
# YARDIMCI FONKSİYONLAR
# ==========================

def normalize_base_symbol(symbol: str) -> str:
    """
    'BTC/USDT', 'BTCUSDT', 'btcusd', 'BTC' -> 'BTC'
    """
    s = symbol.upper()
    if "/" in s:
        base = s.split("/")[0]
    elif s.endswith("USDT"):
        base = s[:-4]
    elif s.endswith("USD"):
        base = s[:-3]
    else:
        base = s
    return base


def fetch_coingecko_markets(vs_currency: str = "usd", per_page: int = 250, page: int = 1) -> List[dict]:
    url = f"{COINGECKO_BASE_URL}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false"
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"CoinGecko markets hatası: {resp.status_code} {resp.text}")
    return resp.json()


def find_coingecko_coin_id(base_symbol: str) -> str:
    """
    CoinGecko'da 'btc' -> 'bitcoin' gibi id'yi bul.
    Sembol eşleşmesiyle arıyoruz.
    """
    sym_lower = base_symbol.lower()
    markets = fetch_coingecko_markets()
    for coin in markets:
        if coin.get("symbol", "").lower() == sym_lower:
            return coin["id"]
    raise ValueError(f"CoinGecko'da '{base_symbol}' için coin bulunamadı.")


def fetch_price_from_coingecko(coin_id: str) -> float:
    url = f"{COINGECKO_BASE_URL}/simple/price"
    params = {
        "ids": coin_id,
        "vs_currencies": "usd",
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"CoinGecko price hatası: {resp.status_code} {resp.text}")
    data = resp.json()
    if coin_id not in data or "usd" not in data[coin_id]:
        raise RuntimeError(f"CoinGecko price verisi eksik: {data}")
    return float(data[coin_id]["usd"])


# ==========================
# GRID BOT İÇ YAPISI
# ==========================

class GridLevel:
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
    CoinGecko fiyatlarıyla çalışan, tamamen simülasyon grid botu.
    USDT ~ USD varsayıyoruz.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.symbol = config.symbol
        self.base_symbol = normalize_base_symbol(config.symbol)

        # CoinGecko id'yi bul
        self.coingecko_id = find_coingecko_coin_id(self.base_symbol)

        # Bakiyeler
        self.usdt_balance: float = config.start_usdt
        self.coin_balance: float = 0.0
        self.realized_pnl_usdt: float = 0.0

        # Grid
        self.levels: List[GridLevel] = []

        # Son durum
        self.last_price: Optional[float] = None
        self.last_update_utc: Optional[str] = None

        # Thread / sync
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False

        # Precision ve min_amount simülasyon için sabit (coin mantığı)
        self.amount_prec = 8
        self.min_amount = 10 ** -self.amount_prec  # çok küçük bir miktar

    # ---------- Yardımcı ----------

    def _round_amount(self, amount: float) -> float:
        factor = 10 ** self.amount_prec
        return math.floor(amount * factor) / factor

    def _fetch_price(self) -> float:
        return fetch_price_from_coingecko(self.coingecko_id)

    def _build_grid(self):
        center = self._fetch_price()
        low = center * (1 - self.config.grid_width_pct)
        high = center * (1 + self.config.grid_width_pct)
        step = (high - low) / self.config.num_grids

        levels: List[GridLevel] = []
        for i in range(self.config.num_grids + 1):
            lvl_price = low + i * step
            levels.append(GridLevel(lvl_price))

        with self._lock:
            self.levels = levels
            self.last_price = center
            self.last_update_utc = datetime.utcnow().isoformat()

        print(f"[GRID] {self.symbol} ({self.coingecko_id}) grid oluşturuldu. Merkez: {center:.4f}")
        for lvl in levels:
            print(f"  - Level: {lvl.level_price:.4f}")

    # ---------- BUY / SELL simülasyon ----------

    def _handle_buy(self, level: GridLevel, price: float):
        if level.holding:
            return

        size_usdt = min(self.config.usdt_per_order, self.usdt_balance)
        if size_usdt < 5:
            return

        raw_amount = size_usdt / price
        amount = self._round_amount(raw_amount)
        if amount <= 0 or amount < self.min_amount:
            return

        cost = amount * price
        fee = cost * 0.001  # %0.1 fee simülasyonu

        self.usdt_balance -= (cost + fee)
        self.coin_balance += amount

        level.holding = True
        level.entry_price = price
        level.amount = amount

        print(
            f"[BUY] {self.symbol} | Fiyat={price:.4f}, amount={amount:.8f}, "
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
            f"amount={amount:.8f}, TradePnL={pnl:.4f}, TotalPnL={self.realized_pnl_usdt:.4f}"
        )

        level.holding = False
        level.entry_price = None
        level.amount = 0.0

    # ---------- Ana loop ----------

    def _loop(self):
        self._build_grid()
        self._running = True
        print(f"[BOT] {self.symbol} ({self.coingecko_id}) için simülasyon başlatıldı.")

        while not self._stop_event.is_set():
            try:
                price = self._fetch_price()
                now_utc = datetime.utcnow().isoformat()

                with self._lock:
                    self.last_price = price
                    self.last_update_utc = now_utc

                    for level in self.levels:
                        # BUY
                        if (not level.holding) and price <= level.level_price:
                            self._handle_buy(level, price)

                        # SELL (TP / SL)
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
                base_symbol=self.base_symbol,
                coingecko_id=self.coingecko_id,
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

@app.get("/")
def root():
    return {
        "message": "Sim trading bot backend (CoinGecko) is running.",
        "endpoints": ["/prices", "/bot/start", "/bot/status", "/bot/stop"]
    }


@app.get("/prices")
def get_all_prices():
    """
    CoinGecko'dan top N coin fiyatlarını getirir.
    Key = base symbol (BTC, ETH, SOL...), value = USD fiyatı
    """
    try:
        markets = fetch_coingecko_markets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result: Dict[str, float] = {}
    for coin in markets:
        symbol = coin.get("symbol", "").upper()  # "btc"
        price = coin.get("current_price", None)
        if price is None:
            continue
        result[symbol] = float(price)
    return result


@app.post("/bot/start", response_model=BotState)
def start_bot(config: BotConfig):
    """
    Simülasyon botunu verilen parametrelerle başlat.
    Önce varsa eski botu durdurur.
    """
    global bot_instance

    try:
        new_bot = SimGridBot(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # eski botu durdur
    if bot_instance is not None:
        bot_instance.stop()

    bot_instance = new_bot
    bot_instance.start()
    time.sleep(1)  # grid oluşsun

    return bot_instance.get_state()


@app.get("/bot/status", response_model=BotState)
def bot_status():
    """
    Botun anlık durumunu döndürür.
    """
    if bot_instance is None:
        raise HTTPException(status_code=400, detail="Bot başlatılmamış. Önce /bot/start çağır.")
    return bot_instance.get_state()


@app.post("/bot/stop")
def stop_bot():
    """
    Botu durdurur ve final state'i döndürür.
    """
    global bot_instance
    if bot_instance is None:
        return {"status": "no_bot", "message": "Zaten aktif bot yok."}

    bot_instance.stop()
    state = bot_instance.get_state()
    bot_instance = None
    return {"status": "stopped", "final_state": state}
