import uuid
import ccxt.async_support as ccxt
import asyncio
import logging
from dotenv import load_dotenv
import os
import csv
import ta.volatility
import yaml
from datetime import datetime
import time
import pandas as pd
import numpy as np
from itertools import product
import ta
from paper_trading import AbstractTradingSystem, abstractTradingSystemBuilder

class TradingBot:
    def __init__(self, config):
        load_dotenv()
        self.config = config
        self.tasks = []
    
    def create_tasks(self):
        for market_config in self.config['markets']:
            bot = MarketBot(market_config, self)
            self.tasks.append(bot.trade_market())

    async def run(self):
        self.create_tasks()
        await asyncio.gather(*self.tasks)

class TrendStrategy:
    @staticmethod
    def _calculate_trend_with_vwap(df, epsilon=0.001):
        df = TradingTools.calculate_vwap(df)
        current_price = df['Close'].iloc[-1]
        vwap = df['VWAP'].iloc[-1]

        if abs(current_price - vwap) / vwap < epsilon:
            trend = 'no_trend'
        elif current_price > vwap:
            trend = 'uptrend'
        else:
            trend = 'downtrend'

        # return current_price, vwap, trend
        return trend

    @staticmethod
    def _calculate_median_slope_trend(df, period=10, epsilon=0.001):
        df['Median'] = (df['High'] + df['Low']) / 2
        df['Median_diff'] = df['Median'].diff(period)
        slope = df['Median_diff'].iloc[-1]
        if abs(slope) < epsilon:
            trend = 'no_trend'
        elif slope > 0:
            trend = 'uptrend'
        else:
            trend = 'downtrend'
        return trend

    # @staticmethod
    # def combined_trend_strategy(df, short_period=10, long_period=30, slope_period=10, epsilon=0.001):
    #     long_term_trend = TradingTools.calculate_moving_average_trend(df, short_period, long_period, epsilon)
    #     short_term_trend = TradingTools.calculate_median_slope_trend(df, slope_period, epsilon)
        
    #     if long_term_trend == short_term_trend and long_term_trend != 'no_trend':
    #         combined_trend = long_term_trend
    #     else:
    #         combined_trend = 'no_trend'
        
    #     return combined_trend

    @staticmethod
    def _calculate_moving_average_trend(df, short_period=10, long_period=30, epsilon=0.001):
        df['SMA_short'] = df['Close'].rolling(window=short_period).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_period).mean()
        short_sma = df['SMA_short'].iloc[-1]
        long_sma = df['SMA_long'].iloc[-1]
        
        # Calculate the percentage difference
        percentage_diff = abs(short_sma - long_sma) / long_sma
        
        if percentage_diff < epsilon:
            trend = 'no_trend'
        elif short_sma > long_sma:
            trend = 'uptrend'
        else:
            trend = 'downtrend'
        
        return trend

    @staticmethod
    def _calculate_slope_trend(df, value = 'Close'):
        def calculate_slope(df, period=10):
            x = np.arange(period)
            y = df[value].iloc[-period:]
            slope, _ = np.polyfit(x, y, 1)
            return slope

        slope = calculate_slope(df)
        epsilon_slope = 0.1
        if slope > epsilon_slope:
            trend = "Uptrend"
        elif slope < -epsilon_slope:
            trend = "Downtrend"
        else:
            trend = "No Trend"
        return trend
    
    @staticmethod
    def calculate_trend(df, strategy):
        if strategy == 'moving_average':
            return TrendStrategy._calculate_moving_average_trend(df)
        if strategy == 'slope_close':
            return TrendStrategy._calculate_slope_trend(df)
        if strategy == 'slope_median':
            return TrendStrategy._calculate_median_slope_trend(df)
        if strategy == 'vwap':
            return TrendStrategy._calculate_trend_with_vwap(df)

class TradingTools:
    
    @staticmethod
    def find_support_resistance(df):
        """
        Identify support and resistance levels in a given OHLC dataframe.
        
        Parameters:
        df (pd.DataFrame): A dataframe with columns 'Open', 'High', 'Low', 'Close'
        window (int): The window size to use for identifying local maxima and minima.
        
        Returns:
        dict: A dictionary with support and resistance levels.
        """
                
        def _extract_support_resistance(value):
            resistance_counts = df['High'].value_counts()
            resistances = resistance_counts[resistance_counts > 1].sort_values(ascending=False).index.tolist()
            resistance_frequencies = resistance_counts[resistance_counts > 1].sort_values(ascending=False).tolist()
            resistance_df = pd.DataFrame({'Level': resistances, 'Frequency': resistance_frequencies})
            resistance_df = resistance_df.sort_values(by='Frequency', ascending=False)
            top_frequencies = resistance_df['Frequency'].unique()[:2]
            resistances = resistance_df[resistance_df['Frequency'].isin(top_frequencies)]['Level'].tolist()
            return resistances
        
        resistances = _extract_support_resistance('High')
        supports = _extract_support_resistance('Low')

        return resistances, supports

    @staticmethod
    def calculate_spreads(resistances, supports):
        """
        Calculate all possible (resistance, support) pairs and sort them by the spread.

        Parameters:
        resistances (list): List of resistance levels.
        supports (list): List of support levels.

        Returns:
        pd.DataFrame: A DataFrame with columns 'Resistance', 'Support', and 'Spread', sorted by 'Spread'.
        """
        pairs = list(product(resistances, supports))
        data = [{'Resistance': r, 'Support': s, 'Spread': r-s} for r, s in pairs]
        df = pd.DataFrame(data)
        df = df.sort_values(by='Spread', ascending=False)
        
        return df

    # @staticmethod
    # def calculate_trend(df, short_period=10, long_period=30, epsilon=0.001):
    #     df['SMA_short'] = df['Close'].rolling(window=short_period).mean()
    #     df['SMA_long'] = df['Close'].rolling(window=long_period).mean()
    #     short_sma = df['SMA_short'].iloc[-1]
    #     long_sma = df['SMA_long'].iloc[-1]
        
    #     # Calculate the percentage difference
    #     percentage_diff = abs(short_sma - long_sma) / long_sma
        
    #     if percentage_diff < epsilon:
    #         trend = 'no_trend'
    #     elif short_sma > long_sma:
    #         trend = 'uptrend'
    #     else:
    #         trend = 'downtrend'
        
    #     return trend

    @staticmethod
    def calculate_vwap(df):
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        return df

    # @staticmethod
    # def calculate_trend_with_vwap(df, epsilon=0.001):
    #     df = TradingTools.calculate_vwap(df)
    #     current_price = df['Close'].iloc[-1]
    #     vwap = df['VWAP'].iloc[-1]

    #     if abs(current_price - vwap) / vwap < epsilon:
    #         trend = 'no_trend'
    #     elif current_price > vwap:
    #         trend = 'uptrend'
    #     else:
    #         trend = 'downtrend'

    #     return current_price, vwap, trend

    # @staticmethod
    # def calculate_median_slope_trend(df, period=10, epsilon=0.001):
    #     df['Median'] = (df['High'] + df['Low']) / 2
    #     df['Median_diff'] = df['Median'].diff(period)
    #     slope = df['Median_diff'].iloc[-1]
    #     if abs(slope) < epsilon:
    #         trend = 'no_trend'
    #     elif slope > 0:
    #         trend = 'uptrend'
    #     else:
    #         trend = 'downtrend'
    #     return trend

    # # @staticmethod
    # # def combined_trend_strategy(df, short_period=10, long_period=30, slope_period=10, epsilon=0.001):
    # #     long_term_trend = TradingTools.calculate_moving_average_trend(df, short_period, long_period, epsilon)
    # #     short_term_trend = TradingTools.calculate_median_slope_trend(df, slope_period, epsilon)
        
    # #     if long_term_trend == short_term_trend and long_term_trend != 'no_trend':
    # #         combined_trend = long_term_trend
    # #     else:
    # #         combined_trend = 'no_trend'
        
    # #     return combined_trend

    # @staticmethod
    # def calculate_moving_average_trend(df, short_period=10, long_period=30, epsilon=0.001):
    #     df['SMA_short'] = df['Close'].rolling(window=short_period).mean()
    #     df['SMA_long'] = df['Close'].rolling(window=long_period).mean()
    #     short_sma = df['SMA_short'].iloc[-1]
    #     long_sma = df['SMA_long'].iloc[-1]
        
    #     # Calculate the percentage difference
    #     percentage_diff = abs(short_sma - long_sma) / long_sma
        
    #     if percentage_diff < epsilon:
    #         trend = 'no_trend'
    #     elif short_sma > long_sma:
    #         trend = 'uptrend'
    #     else:
    #         trend = 'downtrend'
        
    #     return trend

    # @staticmethod
    # def calculate_slope_trend(df, value = 'Close'):
    #     def calculate_slope(df, period=10):
    #         x = np.arange(period)
    #         y = df[value].iloc[-period:]
    #         slope, _ = np.polyfit(x, y, 1)
    #         return slope

    #     slope = calculate_slope(df)
    #     epsilon_slope = 0.1
    #     if slope > epsilon_slope:
    #         trend = "Uptrend"
    #     elif slope < -epsilon_slope:
    #         trend = "Downtrend"
    #     else:
    #         trend = "No Trend"
    #     return trend
    
    # @staticmethod
    # def calculate_trend(df, strategy):
    #     if strategy == 'moving_average':
    #         return TradingTools.calculate_moving_average_trend(df)
    #     if strategy == 'slope_close':
    #         return TradingTools.calculate_slope_trend(df)
    #     if strategy == 'slope_median':
    #         return TradingTools.calculate_median_slope_trend(df)
    #     if strategy == 'vwap':
    #         return TradingTools.calculate_trend_with_vwap(df)

    @staticmethod
    def calculate_volatility_trend_indicator(atr, epsilon=0.05):
        # df = TradingTools.calculate_atr(df, period)
        current_atr = atr.iloc[-1]
        previous_atr = atr.iloc[-2]
        
        if abs(current_atr - previous_atr) / previous_atr < epsilon:
            volatility = 'no_volatility'
        elif current_atr > previous_atr:
            volatility = 'growing'
        else:
            volatility = 'lowering'

        return current_atr, previous_atr, volatility

    @staticmethod
    def dynamic_bid_ask_strategy(df, current_price, buy_fee, sell_fee, spread, price_action_trend):
        
        bid_price, ask_price = TradingTools.calculate_bid_ask(current_price, buy_fee, sell_fee, spread)
        delta = current_price - bid_price
        if price_action_trend == 'uptrend':
            # If in an uptrend, places the ask price higher.
            bid_price += delta
            ask_price += delta
        elif price_action_trend == 'downtrend':
            # If in a downtrend, places the bid price lower.
            bid_price -= delta
            ask_price -= delta
        
        return bid_price, ask_price

    @staticmethod
    def calculate_reference_price(df, method='Median', period=14):
        if method == 'SMA':
            reference_price = df['Close'].rolling(window=period).mean().iloc[-1]
        elif method == 'EMA':
            reference_price = df['Close'].ewm(span=period, adjust=False).mean().iloc[-1]
        elif method == 'VWAP':
            reference_price = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum().iloc[-1]
        elif method == 'TWAP':
            reference_price = df['Close'].expanding().mean().iloc[-1]
        elif method == 'Donchian':
            high = df['High'].rolling(window=period).max().iloc[-1]
            low = df['Low'].rolling(window=period).min().iloc[-1]
            reference_price = (high + low) / 2
        elif method == 'Bollinger':
            sma = df['Close'].rolling(window=period).mean().iloc[-1]
            std = df['Close'].rolling(window=period).std().iloc[-1]
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            reference_price = (upper_band + lower_band) / 2
        elif method == 'Median':
            high = df['High'].iloc[-1]
            low = df['Low'].iloc[-1]
            reference_price = (high + low) / 2
        else:
            raise ValueError("Unsupported method")
        
        return reference_price

    @staticmethod
    def calculate_bid_ask(mid_price, buy_fee, sell_fee, spread):        
        bid_price = (mid_price-spread) * (1-buy_fee)
        ask_price = (mid_price+spread) * (1+sell_fee)
        
        return bid_price, ask_price

    @staticmethod
    def calculate_profit(bid_price, ask_price, market_fee):
        buy_price = bid_price * (1 + market_fee)
        sell_price = ask_price * (1 - market_fee)
        profit = sell_price - buy_price
        return profit

    @staticmethod
    def calculate_high_low_volatility(high, low):
        return (high - low) / low * 100
    
    # @staticmethod
    # def calculate_base_spread(base_fee, volatility, profit_margin, k):
    #     return base_fee + (volatility * k) + profit_margin

    @staticmethod
    def calculate_dynamic_spread(base_spread, additional_volatility):
        return base_spread + (additional_volatility / 2)

    @staticmethod
    def calculate_dynamic_spread(reference_price, volatility, volatility_factor=1.0):
        
        # Calculate the dynamic spread as a percentage of the reference price
        spread_percentage = volatility * volatility_factor
        
        # Calculate the spread
        spread = spread_percentage * reference_price / 100
        
        return spread

    @staticmethod
    def calc_trend(slope):
        epsilon_slope = 0.1
        if slope > epsilon_slope:
            trend = "Uptrend"
        elif slope < -epsilon_slope:
            trend = "Downtrend"
        else:
            trend = "No Trend"
        return trend
    
class MarketBot:
    def __init__(self, config, trading_bot: TradingBot):
        self.config = config
        self.trading_bot = trading_bot
        self.exchange = None
        self.initial_balance = None
        
        self.session = str(uuid.uuid4())
        self.log_file = f'log_trading_{self.config["name"]}.csv'
        self.balance_file = f'log_balance_{self.config["name"]}.csv'
        self._initialize_csv_files()        
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        
    def _initialize_csv_files(self):
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Market', 'Order Type', 'Order Price', 'Order Amount', 'Spread', 'Strategy'])

        with open(self.balance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Session', 'Timestamp', 'Symbol', 'Close', 'Balance A', 'Balance B', 'Balance Tot', 'Diff'])

    def log_trade(self, timestamp, market, order_type, order_price, order_amount, spread, strategy):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, market, order_type, order_price, order_amount, spread, strategy])

    def log_balance(self, timestamp, symbol, close, balanceA, balanceB, balanceTot, diff):
        with open(self.balance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.session, timestamp, symbol, close, balanceA, balanceB, balanceTot, diff])

    async def get_latest_ohlcv(self, symbol, timeframe='1m', limit=2, max_retries=3, retry_delay=5):
        for _ in range(max_retries):
            try:
                return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            except ccxt.BaseError as e:
                logging.warning(f"Error fetching OHLCV data: {e}")
                await asyncio.sleep(retry_delay)
        raise Exception("Max retries reached for fetching OHLCV data")

    async def place_limit_order(self, symbol, order_type, price, amount, max_retries=3, retry_delay=5):
        for _ in range(max_retries):
            try:
                if order_type == 'buy':
                    order = await self.exchange.create_limit_buy_order(symbol, amount, price)
                    return order
                elif order_type == 'sell':
                    order = await self.exchange.create_limit_sell_order(symbol, amount, price)
                    return order
            except ccxt.BaseError as e:
                logging.warning(f"Error placing {order_type} order: {e}")
                await asyncio.sleep(retry_delay)
        raise Exception(f"Max retries reached for placing {order_type} order")

    async def place_timed_limit_order(self, symbol, order_type, price, amount, max_order_value_slice=0):

            def index_to_sequence(index):
                if index % 2 == 0:
                    # Even index, return negative sequence
                    return -(index // 2 + 1)
                else:
                    # Odd index, return positive sequence
                    return (index // 2 + 1)

            order_value = price * amount
            if max_order_value_slice == 0 or order_value <= max_order_value_slice:
                order = await self.place_limit_order(symbol, order_type, price, amount)
                return [order]
            else:
                order_slices = []
                i = 0
                remaining_amount = amount
                while remaining_amount > 0:
                    order_amount = min(remaining_amount, max_order_value_slice / price)
                    amount_increment = (self.config["order_size_slice_in_usd"] * index_to_sequence(i))
                    order = await self.place_limit_order(symbol, order_type, price + amount_increment, order_amount)
                    order_slices.append(order)
                    remaining_amount -= order_amount
                    i += 1
                return order_slices

    async def cancel_all_orders(self, symbol, max_retries=3, retry_delay=5):
        for _ in range(max_retries):
            try:
                open_orders = await self.exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    await self.exchange.cancel_order(order['id'], symbol)
                return
            except ccxt.BaseError as e:
                logging.warning(f"Error canceling orders: {e}")
                await asyncio.sleep(retry_delay)
        raise Exception("Max retries reached for canceling orders")

    async def get_balance(self, asset):
        balance = await self.exchange.fetch_balance()
        if asset in balance['free']:
            return balance['free'][asset]
        else:
            raise ValueError(f"Asset {asset} not found in balance")

    async def calculate_order_size(self, symbol, percentage, order_type):
        asset = symbol.split('/')[0]  # Assuming USDT is the quote asset
        balance = await self.get_balance(asset)
        if order_type == 'buy':
            return balance * percentage / 100
        elif order_type == 'sell':
            base_asset = symbol.split('/')[0]  # Assuming BTC is the base asset
            base_balance = await self.get_balance(base_asset)
            return base_balance * percentage / 100
        else:
            raise ValueError("Invalid order type")

    async def get_additional_market_data(self, symbol):
        ticker = await self.exchange.fetch_ticker(symbol)
        volume = ticker['quoteVolume']
        recent_trend = (ticker['close'] - ticker['open']) / ticker['open'] * 100
        return volume, recent_trend

    async def place_stop_loss_order(self, symbol, order_type, stop_price, amount):
        if order_type == 'buy':
            return await self.exchange.create_stop_limit_buy_order(symbol, amount, stop_price, stop_price)
        elif order_type == 'sell':
            return await self.exchange.create_stop_limit_sell_order(symbol, amount, stop_price, stop_price)

    async def dynamic_order_size_strategy(self, symbol, bid_price, ask_price, order_size, timestamp, spread, strategy, volatility_trend):
        
        # SLICING THE ORDER INTO SMALLER ORDERS
        if volatility_trend == 'growing':
            order_max_size_in_usd = self.config['order_max_size_in_usd']
        elif volatility_trend == 'lowering':
            order_max_size_in_usd = self.config['order_max_size_in_usd'] * 1.5
        else:
            order_max_size_in_usd = 0
            
        order_size_buy, order_size_sell = order_size
        
        await self.place_timed_limit_order(symbol, 'buy', bid_price, order_size_buy, order_max_size_in_usd)
        self.log_trade(timestamp, symbol, 'buy', bid_price, order_size_buy, spread, strategy)
        
        await self.place_timed_limit_order(symbol, 'sell', ask_price, order_size_sell, order_max_size_in_usd)
        self.log_trade(timestamp, symbol, 'sell', ask_price, order_size_sell, spread, strategy)

    async def risk_management_strategy(self, symbol, bid_price, ask_price, order_size, timestamp, spread, strategy, stop_loss_percentage):
        order_size_buy, order_size_sell = order_size
        stop_loss_buy = bid_price * (1 - stop_loss_percentage / 100)
        stop_loss_sell = ask_price * (1 + stop_loss_percentage / 100)
        
        await self.place_timed_limit_order(symbol, 'buy', bid_price, order_size)
        self.log_trade(timestamp, symbol, 'buy', bid_price, order_size_buy, spread, strategy)
        
        await self.place_timed_limit_order(symbol, 'sell', ask_price, order_size_sell)
        self.log_trade(timestamp, symbol, 'sell', ask_price, order_size_sell, spread, strategy)
        
        await self.place_stop_loss_order(symbol, 'buy', stop_loss_buy, order_size_buy)
        self.log_trade(timestamp, symbol, 'stop_loss_buy', stop_loss_buy, order_size_buy, spread, strategy)
        
        await self.place_stop_loss_order(symbol, 'sell', stop_loss_sell, order_size_sell)
        self.log_trade(timestamp, symbol, 'stop_loss_sell', stop_loss_sell, order_size_sell, spread, strategy)

    async def calculate_total_balance(self, symbol):
        base_asset = symbol.split('/')[0]
        quote_asset = symbol.split('/')[1]
        base_balance = await self.get_balance(base_asset)
        quote_balance = await self.get_balance(quote_asset)
        
        ticker = await self.exchange.fetch_ticker(symbol)
        base_to_quote_price = ticker['close']
        
        base_balance_in_quote = base_balance * base_to_quote_price
        total_balance_in_quote = base_balance_in_quote + quote_balance
        
        return base_balance, quote_balance, total_balance_in_quote

    async def print_balance(self, symbol, close):
        balance = await self.calculate_total_balance(symbol)
        logging.info(f"Current balance for {symbol}: {balance} - {balance[2]-self.initial_balance[2]}")
        self.trading_bot.log_balance(datetime.now().isoformat(), symbol, close, balance[0], balance[1], balance[2], balance[2]-self.initial_balance[2])
        print()

    async def cancel_old_orders(self, close):
        open_orders = []
        cancelled_orders = []
        not_cancelled_orders = []
        
        # CANCEL ALL THE ORDERS
        if not self.config['apply_order_timeout']:
            await self.cancel_all_orders(self.config['symbol'], max_retries=self.config['max_retries'], retry_delay=self.config['retry_delay'])
            return 0, []

        # CANCEL OLDEST ORDERSj
        open_orders = await self.exchange.fetch_open_orders(self.config['symbol'])
        current_time = datetime.now()
        
        cancelled = 0
        for order in open_orders:
            order_id = order['id']
            order_timestamp = datetime.fromtimestamp(order['timestamp'] / 1000)
            if (current_time - order_timestamp).total_seconds() > self.config['order_timeout']:
                await self.exchange.cancel_order(order_id, self.config['symbol'])
                cancelled += 1
                cancelled_orders.append(order['id'])
                logging.info(f"Cancelled order {order_id} due to timeout for {self.config['symbol']}")
        
        # IF NO ORDERS WERE CANCELLED DUE TO TIMEOUT, CANCEL THE ORDERS BY DISTANCE
        if self.config['cancel_orders_by_distance'] and len(open_orders) >= self.config['max_orders']:
            # cancel the orders with price most far from the current price
            open_orders = sorted(open_orders, key=lambda x: abs(x['price'] - close), reverse=True)
            # I WANT TO KEEP THE MAX ORDERS-2, AND CANCEL THE REST
            diff= 2 + (len(open_orders) - self.config['max_orders'])
            for order in open_orders[:diff]:
                await self.exchange.cancel_order(order['id'], self.config['symbol'])
                cancelled += 1
                cancelled_orders.append(order['id'])
                logging.info(f"Cancelled order {order['id']} due to distance for {self.config['symbol']}")

        not_cancelled_orders = [order for order in open_orders if order['id'] not in cancelled_orders]

        return len(open_orders) - cancelled, not_cancelled_orders

    async def order_strategy(self, bid_price, ask_price, spread, stop_loss_percentage, volatility_trend):
        timestamp = datetime.now().isoformat()
        order_size_buy = await self.calculate_order_size(self.config['symbol'], self.config['order_size'], "buy")
        order_size_sell = await self.calculate_order_size(self.config['symbol'], self.config['order_size'], "sell")
        order_size = (order_size_buy, order_size_sell)
        
        if self.config['strategy'] == 'dynamic_order_size':
            volatility_trend_label = volatility_trend[2]
            await self.dynamic_order_size_strategy(self.config['symbol'], bid_price, ask_price, order_size, timestamp, spread, self.config['strategy'], volatility_trend=volatility_trend_label)

        elif self.config['strategy'] == 'risk_management':
            await self.risk_management_strategy(self.config['symbol'], bid_price, ask_price, order_size, timestamp, spread, self.config['strategy'], stop_loss_percentage)
        
        logging.info(f"Placed buy order at {bid_price} and sell order at {ask_price} for {self.config['symbol']}")

    def reduce_ask_bid(self, orders, ask_bid, descending=True):

        '''
        reduce the number of orders based on resistance and support levels
        it can apply two strategies:
            one filtering the orders where ask or bid are not the same of an already placed order
            one filtering the orders where both ask and bid are not the same of an already placed order
        '''   
        strategy = self.config['reduce_ask_bid_strategy']   
        
        def _at_least_once(order, ask, bid):
            return (order['side'] == 'sell' and order['price'] != ask) or (order['side'] == 'buy' and order['price'] != bid)
        
        def _both(order, ask, bid):
            return order['price'] != ask and order['price'] != bid
        
        strategy_function = _at_least_once if strategy == 'at_least_once' else _both
        
        if len(orders) > 0:
            filtered_ask_bid = [(ask, bid, profit) for order in orders for ask, bid, profit in ask_bid if strategy_function(order, ask, bid)]
        else:
            filtered_ask_bid = ask_bid
    
        filtered_ask_bid = sorted(list(set(filtered_ask_bid)), key=lambda x: x[2], reverse=descending)
        return filtered_ask_bid

    async def res_supp_opportunities(self, df, limit, base_fee, min_profit=1):
        resistances, supports = TradingTools.find_support_resistance(df.tail(limit))
        spreads_df = TradingTools.calculate_spreads(resistances, supports)
        filtered_spread_df = spreads_df[spreads_df['Spread'] > 0]
        # first_three_rows = filtered_spread_df
        # resistances = filtered_spread_df['Resistance'].tolist()
        # supports = filtered_spread_df['Support'].tolist()
        resistance_support_tuples = list(zip(filtered_spread_df['Resistance'].tolist(), filtered_spread_df['Support'].tolist())) 
                    
        # TRANSFORM RESISTANCE AND SUPPORTS IN ASK AND BID
        ask_bid_from_res_supp = []
        for resistance, support in resistance_support_tuples:
            # logging.info(f"Resistance: {resistance} and Support: {support}")
            profit = TradingTools.calculate_profit(support, resistance, base_fee)
            if profit > min_profit:
                # logging.info(f"r: {resistance} s: {support}")
                # logging.info(f"Profitable trade found between support and resistances with profit: {profit}")
                ask_bid_from_res_supp.append((resistance, support, profit))
        return ask_bid_from_res_supp

    async def get_candles(self, limit=10):
        ohlcv = await self.get_latest_ohlcv(self.config['symbol'], "1m", limit=limit, max_retries=self.config['max_retries'], retry_delay=self.config['retry_delay'])
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        return df

    async def latest_candle(self, df):
        latest_candle = df.iloc[-1]
        high = latest_candle['High']
        low = latest_candle['Low']
        open = latest_candle['Open']
        close = latest_candle['Close']
        return high, low, open, close

    async def trade_market(self):
        
        trading_engine = self.config['trading_engine']
        is_paper = trading_engine == 'paper'
        
        exchange = getattr(ccxt, self.config['exchange_id'])({
            'apiKey': self.config['api_key'],
            'secret': self.config['api_secret'],
            'options': {'defaultType': self.config['exchange_type']},
        })
        initial_balance = {'BTC': 1, 'USDT': 100000}
        trading_system = abstractTradingSystemBuilder(trading_engine, exchange, initial_balance)
        
        self.exchange = trading_system
        if self.config['sandbox_mode']:
            self.exchange.set_sandbox_mode(True)
        
        markets = await self.exchange.load_markets()
        market_symbol = markets[self.config['symbol']]
        maker_fee = market_symbol['maker']
        min_profit = self.config["min_profit"]

        volatility_scaling_factor = self.config["volatility_scaling_factor"]
        stop_loss_percentage = self.config["stop_loss_percentage"]

        # ohlcv = await self.get_latest_ohlcv(self.config['symbol'], timeframe='1m', limit=10, max_retries=self.config['max_retries'], retry_delay=self.config['retry_delay'])
        df = await self.get_candles(limit=10)
        high, low, open, close = await self.latest_candle(df)

        volatilities = [TradingTools.calculate_high_low_volatility(row['High'], row['Low']) for _, row in df.iterrows()]
        average_volatility = sum(volatilities) / len(volatilities)
        
        self.initial_balance = await self.calculate_total_balance(self.config['symbol'])
        await self.print_balance(self.config['symbol'], close)        
        print()

        limit=self.config['max_candles']

        symbol = self.config['symbol']
        last_time_checked_simulated_order_execution = time.time()
        
        while True:
            
            try:

                start_time = time.time()
                # TODO: we can read less candles and build the dataframe merging the results 
                df = await self.get_candles(limit=limit)
                high, low, open, close = await self.latest_candle(df)
                
                if is_paper:
                    if time.time() - last_time_checked_simulated_order_execution >= 60:
                        self.exchange.simulate_order_execution(symbol, close)
                        last_time_checked_simulated_order_execution = time.time()

                # INDICATORS
                rsi = ta.momentum.rsi(df['Close'], window=14)
                last_rsi = rsi.iloc[-1]
                if last_rsi > 70:
                    logging.info(f"RSI is overbought for {symbol}")
                elif last_rsi < 30:
                    logging.info(f"RSI is oversold for {symbol}")

                # ORDERS TO CANCEL
                num_orders, not_cancelled_orders = await self.cancel_old_orders(close)
                if self.config['max_orders'] > 0 and num_orders >= self.config['max_orders']:
                    logging.info(f"Reached maximum number of open orders: {self.config['max_orders']} for {symbol}")
                    print()
                    await asyncio.sleep(10)
                    continue

                # SUPPORT AND RESISTANCES
                ask_bid_candidates = await self.res_supp_opportunities(df, limit, maker_fee, min_profit=self.config['min_profit'])
                  
                price_action_trend = TrendStrategy.calculate_trend(df, strategy="slope_close")
                logging.info(f"Current trend: {price_action_trend}")

                 # REFERENCE PRICE
                method = "Median"
                reference_price =  TradingTools.calculate_reference_price(df, method=method, period=14)
                
                # REFERENCE PRICE MUST BE BETWEEN ASK AND BID
                ask_bid_candidates = [(ask, bid, profit) for ask, bid, profit in ask_bid_candidates if not (bid < reference_price < ask)]
                
                # VOLATILITY   
                current_volatility = TradingTools.calculate_high_low_volatility(high, low)
                volatility = (current_volatility + average_volatility) / 2
                average_volatility = volatility
                logging.info(f"Current volatility for {self.config['symbol']}: {volatility}")

                # ATR
                atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
                volatility_trend = TradingTools.calculate_volatility_trend_indicator(atr)
                logging.info(f"ATR: {atr.iloc[-1]} {volatility_trend}")
                
                # SPREAD
                spread = TradingTools.calculate_dynamic_spread(reference_price, volatility, volatility_scaling_factor)
                logging.info(f"Current spread for {self.config['symbol']}: {spread}")
                bid_price, ask_price = TradingTools.dynamic_bid_ask_strategy(df, reference_price, maker_fee, maker_fee, spread, price_action_trend)
                profit = TradingTools.calculate_profit(bid_price, ask_price, maker_fee)
                if profit > min_profit:
                    ask_bid_candidates.append((ask_price, bid_price, profit))
                    
                # PREPARING ORDERS
                ask_bid = self.reduce_ask_bid(not_cancelled_orders, ask_bid_candidates, descending=volatility_trend == 'growing')
                for ask, bid, profit in ask_bid:
                    print(f"{ask:.2f}", f"{bid:.2f}", f"{profit:.2f}")
                
                for ask, bid, profit in ask_bid:
                    await self.order_strategy(bid, ask, spread, stop_loss_percentage, volatility_trend=volatility_trend)
                    # print(f"{ask:.2f}", f"{bid:.2f}", f"{profit:.2f}")
                    num_orders += 2
                    if num_orders >= self.config['max_orders']:
                        break
                    
                await self.print_balance(self.config['symbol'], close)
                
                # elapsed_time = time.time() - start_time
                logging.info(f"Elapsed time: {(time.time() - start_time):.2f} seconds")
                
                print("Waiting for the next iteration...")
                await asyncio.sleep(self.config['polling'])
                
            except Exception as e:
                logging.error(f"An error occurred for {self.config['symbol']}: {e}")
                await asyncio.sleep(self.config['retry_delay'])
            
            finally:

                print()
                
                

        # await self.cancel_all_orders(self.config['symbol'], max_retries=self.config['max_retries'], retry_delay=self.config['retry_delay'])
        # await asyncio.sleep(5)
        # balance = await self.calculate_total_balance(self.config['symbol'])
        # logging.info(f"Final balance for {self.config['symbol']}: {balance} - {balance[2]-self.initial_balance[2]}")
        # print()


def banner():
    print("░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓████████▓▒░ ")
    print("░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░         ░▒▓█▓▒░     ")
    print("░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░         ░▒▓█▓▒░     ")
    print("░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓███████▓▒░░▒▓██████▓▒░    ░▒▓█▓▒░     ")
    print("░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░         ░▒▓█▓▒░     ")
    print("░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░         ░▒▓█▓▒░     ")
    print("░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░  ░▒▓█▓▒░     ")
    print("                                                                                       ")
    print("                                                                                       ")
    print("      ░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░         ")
    print("      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░        ")
    print("      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░        ")
    print("      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓██████▓▒░ ░▒▓███████▓▒░         ")
    print("      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░        ")
    print("      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░        ")
    print("      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░        ")
    print()

if __name__ == "__main__":
    banner()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    trading_bot = TradingBot(config)
    asyncio.run(trading_bot.run())
