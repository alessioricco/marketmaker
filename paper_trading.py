import uuid
from datetime import datetime
import ccxt

class InsufficientBalanceError(Exception):
    pass

class OrderNotFoundError(Exception):
    pass

class AbstractTradingSystem:
    def __init__(self, real_exchange, initial_balance):
        self.real_exchange = real_exchange
        self.balances = {'free': initial_balance.copy(), 'used': {asset: 0.0 for asset in initial_balance}}
        self.orders = []

    async def fetch_balance(self):
        return self.balances

    async def create_limit_buy_order(self, symbol, amount, price):
        raise NotImplementedError

    async def create_limit_sell_order(self, symbol, amount, price):
        raise NotImplementedError

    async def fetch_open_orders(self, symbol=None):
        raise NotImplementedError

    async def cancel_order(self, order_id, symbol):
        raise NotImplementedError

    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=2):
        raise NotImplementedError

    async def fetch_ticker(self, symbol):
        raise NotImplementedError
    
    async def load_markets(self):
        raise NotImplementedError
    
    def set_sandbox_mode(self, enable):
        raise NotImplementedError

class CCXTTrading(AbstractTradingSystem):
    def __init__(self, exchange, initial_balance):
        super().__init__(exchange, initial_balance)

    async def fetch_balance(self):
        return await self.real_exchange.fetch_balance()

    async def create_limit_buy_order(self, symbol, amount, price):
        return await self.real_exchange.create_limit_buy_order(symbol, amount, price)

    async def create_limit_sell_order(self, symbol, amount, price):
        return await self.real_exchange.create_limit_sell_order(symbol, amount, price)

    # async def fetch_open_orders(self, symbol=None):
    #     return await self.real_exchange.fetch_open_orders(symbol)
    async def fetch_open_orders(self, symbol=None):
        print(f"Fetching open orders for symbol={symbol}")
        all_open_orders = []
        since = None  # Starting point for fetching orders
        limit = 100   # Max number of orders per request (if supported by the exchange)

        while True:
            open_orders = await self.real_exchange.fetch_open_orders(symbol=symbol, since=since, limit=limit)
            if not open_orders:
                break  # No more open orders to fetch

            all_open_orders.extend(open_orders)
            since = open_orders[-1]['timestamp'] + 1  # Update 'since' to the timestamp of the last fetched order

            if len(open_orders) < limit:
                break  # Fetched fewer than 'limit' orders, indicating the end of available orders

        print(f"Total open orders fetched: {len(all_open_orders)}")
        return all_open_orders
    
    async def cancel_order(self, order_id, symbol):
        return await self.real_exchange.cancel_order(order_id, symbol)

    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=2):
        return await self.real_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def fetch_ticker(self, symbol):
        return await self.real_exchange.fetch_ticker(symbol)
    
    async def load_markets(self):
        return await self.real_exchange.load_markets()
    
    def set_sandbox_mode(self, enable):
        return  self.real_exchange.set_sandbox_mode(enable)

    def enable_demo_trading(self, enable):
        return  self.real_exchange.enable_demo_trading(enable)

class PaperTrading(AbstractTradingSystem):
    def __init__(self, real_exchange, initial_balance):
        super().__init__(real_exchange, initial_balance)

    async def fetch_balance(self):
        async with self.lock:
            print("Fetching balance...")
            print(f"Current balances: {self.balances}")
            return self.balances

    async def create_limit_buy_order(self, symbol, amount, price):
        async with self.lock:
            print(f"Creating limit buy order: symbol={symbol}, amount={amount}, price={price}")
            base, quote = symbol.split('/')
            cost = amount * price
            print(f"Calculated cost: {cost}")
            if self.balances['free'][quote] < cost:
                raise InsufficientBalanceError("Insufficient balance")
            print("Sufficient balance available")

            order = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().timestamp(),
                'datetime': datetime.now().isoformat(),
                'status': 'open',
                'symbol': symbol,
                'type': 'limit',
                'side': 'buy',
                'price': price,
                'amount': amount,
                'filled': 0,
                'remaining': amount,
                'cost': cost,
                'fee': None,
            }
            self.balances['free'][quote] -= cost
            self.balances['used'][quote] += cost
            self.orders.append(order)
            print(f"Order created: {order}")
            print(f"Updated balances: {self.balances}")
            return order

    async def create_limit_sell_order(self, symbol, amount, price):
        async with self.lock:
            print(f"Creating limit sell order: symbol={symbol}, amount={amount}, price={price}")
            base, quote = symbol.split('/')
            if self.balances['free'][base] < amount:
                raise InsufficientBalanceError("Insufficient balance")
            print("Sufficient balance available")

            order = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().timestamp(),
                'datetime': datetime.now().isoformat(),
                'status': 'open',
                'symbol': symbol,
                'type': 'limit',
                'side': 'sell',
                'price': price,
                'amount': amount,
                'filled': 0,
                'remaining': amount,
                'cost': amount * price,
                'fee': None,
            }
            self.balances['free'][base] -= amount
            self.balances['used'][base] += amount
            self.orders.append(order)
            print(f"Order created: {order}")
            print(f"Updated balances: {self.balances}")
            return order

    async def fetch_open_orders(self, symbol=None):
        async with self.lock:
            print(f"Fetching open orders for symbol={symbol}")
            open_orders = [order for order in self.orders if order['status'] == 'open' and (symbol is None or order['symbol'] == symbol)]
            print(f"Open orders: {open_orders}")
            return open_orders

    async def cancel_order(self, order_id, symbol):
        async with self.lock:
            print(f"Cancelling order: order_id={order_id}, symbol={symbol}")
            order = next((o for o in self.orders if o['id'] == order_id and o['symbol'] == symbol), None)
            if order is None:
                raise OrderNotFoundError("Order not found")
            print("Order found")

            if order['side'] == 'buy':
                quote = symbol.split('/')[1]
                self.balances['free'][quote] += order['cost']
                self.balances['used'][quote] -= order['cost']
            elif order['side'] == 'sell':
                base = symbol.split('/')[0]
                self.balances['free'][base] += order['amount']
                self.balances['used'][base] -= order['amount']

            order['status'] = 'canceled'
            print(f"Order cancelled: {order}")
            print(f"Updated balances: {self.balances}")
            return order

    def set_sandbox_mode(self, enable):
        return self.real_exchange.set_sandbox_mode(enable)

    async def load_markets(self):
        print("Loading markets...")
        self.markets = await self.real_exchange.load_markets()
        print(f"Markets loaded: {self.markets}")        
        return self.markets

    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=2):
        print(f"Fetching OHLCV for symbol={symbol}, timeframe={timeframe}, limit={limit}")
        ohlcv = await self.real_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        print(f"OHLCV data: {ohlcv}")
        return ohlcv

    async def fetch_ticker(self, symbol):
        print(f"Fetching ticker for symbol={symbol}")
        ticker = await self.real_exchange.fetch_ticker(symbol)
        print(f"Ticker data: {ticker}")
        return ticker

    def simulate_order_execution(self, symbol, current_price):
        print(f"Simulating order execution for symbol={symbol}, current_price={current_price}")
        
        market_symbol = self.markets[symbol]
        maker_fee = market_symbol['maker']        
        
        for order in self.orders:
            if order['status'] == 'open' and order['symbol'] == symbol:
                if (order['side'] == 'buy' and current_price <= order['price']) or (order['side'] == 'sell' and current_price >= order['price']):
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['remaining'] = 0
                    base, quote = symbol.split('/')
                    fee = maker_fee * order['cost']  # Example fee calculation (0.1%)
                    if order['side'] == 'buy':
                        self.balances['used'][quote] -= order['cost']
                        self.balances['free'][base] += order['amount']
                        self.balances['free'][quote] -= fee
                    elif order['side'] == 'sell':
                        self.balances['used'][base] -= order['amount']
                        self.balances['free'][quote] += order['cost']
                        self.balances['free'][quote] -= fee
                    print(f"Order executed: {order}")
        print(f"Updated orders: {self.orders}")
        print(f"Updated balances: {self.balances}")
        
def abstractTradingSystemBuilder(trading_system, exchange, initial_balance)->AbstractTradingSystem:
    if trading_system == 'paper':
        return PaperTrading(exchange, initial_balance)
    else:
        return CCXTTrading(exchange, initial_balance)

