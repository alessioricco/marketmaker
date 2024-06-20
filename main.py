from functools import partial
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
            resistance_counts = df[value].value_counts()
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
        if len(df) == 0:
            return df
        
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
        
        atr_increment_percentage = TradingTools.calculate_increment_percentage(current_atr,previous_atr)
        
        if abs(atr_increment_percentage) < epsilon:
            volatility = 'no_volatility'
        elif current_atr > previous_atr:
            volatility = 'growing'
        else:
            volatility = 'lowering'

        return current_atr, previous_atr, atr_increment_percentage, volatility

    # @staticmethod
    # def dynamic_bid_ask_strategy(df, current_price, buy_fee, sell_fee, spread, price_action_trend):
        
    #     bid_price, ask_price = TradingTools.calculate_bid_ask(current_price, buy_fee, sell_fee, spread)
    #     delta = ask_price - bid_price
    #     if price_action_trend == 'uptrend':            
    #         bid_price = current_price
    #         ask_price = current_price + delta
            
    #     elif price_action_trend == 'downtrend':
    #         ask_price = current_price
    #         bid_price = current_price - delta
    #     return bid_price, ask_price

    def oscillating_bid_ask_strategy(df, current_price, buy_fee, sell_fee, spread, oscillator):
        # Calculate the initial bid and ask prices around the current price
        bid_price, ask_price = TradingTools.calculate_bid_ask(current_price, buy_fee, sell_fee, spread)
        
        # Calculate the difference between ask and bid prices
        price_difference = ask_price - bid_price
        
        # Adjust bid and ask prices based on the oscillator value
        # When oscillator is 0, bid price should be the same as the current price, and ask price is shifted
        # When oscillator is 1, ask price should be the same as the current price, and bid price is shifted
        bid_price = current_price - (price_difference * oscillator )
        ask_price = current_price + (price_difference * (1 - oscillator) )
        
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
    
    # @staticmethod
    # def calculate_bid_ask_with_profit(mid_price, buy_fee, sell_fee, profit):
    #     # Calculate half of the desired profit margin to apply it symmetrically
    #     half_profit = profit / 2
        
    #     # Calculate the bid price (adjusting for buy_fee)
    #     bid_price = mid_price * (1 - half_profit) * (1 - buy_fee)
        
    #     # Calculate the ask price (adjusting for sell_fee)
    #     ask_price = mid_price * (1 + half_profit) * (1 + sell_fee)
        
    #     return bid_price, ask_price

    @staticmethod
    def calculate_increment_percentage(current, previous):
        try:
            increment = ((current - previous) / previous) * 100
        except ZeroDivisionError:
            increment = 0
        return increment

    @staticmethod
    def calculate_bid_ask_with_profit(mid_price, buy_fee, sell_fee, profit, oscillator=0.5):
        # Calculate half of the desired profit margin to apply it symmetrically
        # half_profit = profit #/ 2
        
        # Calculate the bid price (adjusting for buy_fee)
        bid_price = mid_price * (1 - profit * oscillator) * (1 - buy_fee)
        
        # Calculate the ask price (adjusting for sell_fee)
        ask_price = mid_price * (1 + profit * (1 - oscillator)) * (1 + sell_fee)
        
        return bid_price, ask_price


    @staticmethod
    def calculate_ask_from_bid(bid_price, buy_fee, sell_fee, spread):
        mid_price = bid_price / (1 - buy_fee) + spread
        ask_price = (mid_price + spread) * (1 + sell_fee)
        
        return ask_price

    @staticmethod
    def calculate_bid_from_ask(ask_price, buy_fee, sell_fee, spread):
        mid_price = ask_price / (1 + sell_fee) - spread
        bid_price = (mid_price - spread) * (1 - buy_fee)
        
        return bid_price

    @staticmethod
    def calculate_ask_from_bid_with_profit(bid_price, buy_fee, sell_fee, profit):
        # Calculate the mid_price from the bid_price by reversing the buy_fee adjustment
        mid_price = bid_price / (1 - buy_fee)
        # Calculate the ask_price by adjusting the mid_price for the desired profit and the sell_fee
        ask_price = mid_price * (1 + profit) * (1 + sell_fee)
        return ask_price

    @staticmethod
    def calculate_bid_from_ask_with_profit(ask_price, buy_fee, sell_fee, profit):
        # Calculate the mid_price from the ask_price by reversing the sell_fee adjustment
        mid_price = ask_price / (1 + sell_fee)
        # Calculate the bid_price by adjusting the mid_price for the desired profit and the buy_fee
        bid_price = mid_price / (1 + profit) * (1 - buy_fee)
        return bid_price


    @staticmethod
    def calculate_bid_ask_from_reference(reference_price, profit, buy_fee, sell_fee):
        bid_price = reference_price * (1 - profit) / (1 + buy_fee)
        ask_price = reference_price * (1 + profit) / (1 - sell_fee)
        
        return bid_price, ask_price

    @staticmethod
    def calculate_profit_percentage_from_bid_ask(bid_price, ask_price, buy_fee, sell_fee):
        # Calcolo del profitto lordo
        gross_profit = ask_price - bid_price
        
        # Calcolo del profitto netto considerando le fee
        net_profit = (ask_price * (1 - sell_fee)) - (bid_price * (1 + buy_fee))
        
        # Calcolo della percentuale di profitto rispetto al costo iniziale (bid price)
        profit_percentage = net_profit / (bid_price * (1 + buy_fee))
        
        return profit_percentage


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



class OrderManager:
    def __init__(self):
        self.orders = {}

    def onCancelled(self, order_id):
        if order_id in self.orders:
            del self.orders[order_id]
            # print(f"Order {order_id} cancelled and removed from the dictionary.")

    def onAdded(self, order, type=None):
        order_id = order['id']
        self.orders[order_id] = {"order": order, "info": {"type": type}}
        # print(f"Order {order_id} added to the dictionary.")

    def getType(self, order_id):
        if order_id in self.orders:
            if "type" in self.orders[order_id]["info"]:
                return self.orders[order_id]["info"]["type"]
        return None


    def filter_perm_orders(self, order_list):
        return [order for order in order_list if not self.getType(order['id']) in ['perm']]
    # self.place_timed_limit_order(symbol, 'buy', bid_price, order_size_buy, order_max_size_in_usd),

    async def onReconcile(self, orders_list, create_simmetrical_order_func=None):
        timestamp = datetime.now().isoformat()
        current_order_ids = {order['id'] for order in orders_list}
        already_filled_orders = {order['order']['id'] for order in self.orders.values() if 'mm_status' in order["info"] and order["info"]['mm_status'] == 'filled'}
        current_orders = set(self.orders.keys()) # for the self.orders that are not yet "filled"
        filled_orders = (current_orders - current_order_ids) - already_filled_orders
        for order_id in current_orders:
            if order_id in current_order_ids:
                self.orders[order_id]["order"] = [order for order in orders_list if order['id'] == order_id][0]
            if order_id in filled_orders:
                
                filled_order = self.orders[order_id]
                filled_order["info"]['mm_status'] = 'filled'
                # TODO: log the trade
                # self.log_trade(filled_order["order"]["datetime"], filled_order["order"]["symbol"], filled_order["order"]["side"], filled_order["order"]["price"], filled_order["order"]["amount"], 0, "")
                
                print(f"Order {order_id} marked as filled.") 
                if "type" in filled_order["info"] and filled_order["info"]["type"] == 'perm':
                    # i filled a perm order, so this was an opposite order that was filled
                    # i must remove the original order
                    original_order_id = filled_order["info"]["original_order_id"]
                    del self.orders[original_order_id]
                    print(f"Original order {original_order_id} removed.")
                    del filled_order
                    print(f"Order {order_id} removed.")
                else:
                    # i must create a new order in the opposite direction, mark it as perm
                    #   for the same amount, but with a different price allowing me to have a profit
                    if not create_simmetrical_order_func:
                        return
                    await create_simmetrical_order_func(filled_order["order"])   
                  
                        

    def reduce_filled(self, maker_fee):
        return 0,0
        filled_orders = [order for order in self.orders.values() if 'mm_status' in order and order['mm_status'] == 'filled']
        buy_orders = [order for order in filled_orders if order['side'] == 'buy']
        sell_orders = [order for order in filled_orders if order['side'] == 'sell']

        matched_orders = []

        total_profit = 0
        for sell_order in sell_orders:
            best_match = None
            best_match_value = float('inf')

            for buy_order in buy_orders:
                # match_value = self.match(sell_order, buy_order)
                match_value = TradingTools.calculate_profit(sell_order, buy_order, maker_fee)
                if 0 < match_value < best_match_value:
                    best_match = buy_order
                    best_match_value = match_value

            if best_match:
                total_profit += best_match_value
                matched_orders.append((best_match['id'], sell_order['id']))
                buy_orders.remove(best_match)

        for buy_id, sell_id in matched_orders:
            del self.orders[buy_id]
            del self.orders[sell_id]
            print(f"Matched and removed orders {buy_id} and {sell_id}.")
        
        filled_orders = [order for order in self.orders.values() if order['status'] == 'filled']
        return total_profit, len(filled_orders)

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
        self.order_manager = OrderManager()      
        
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

    async def place_timed_limit_order(self, symbol, order_type, price, amount, max_order_value_slice=0, order_info=None):

            async def _place_limit_order(self, symbol, order_type, price, amount, max_retries=3, retry_delay=5):
               
                for _ in range(max_retries):
                    logging.info(f"Order {symbol}:{order_type} amount:{amount} price:{price}")
                    try:
                        if order_type == 'buy':
                            order = await self.exchange.create_limit_buy_order(symbol, amount, price)
                            self.order_manager.onAdded(order, order_info)
                            # self.log_trade(timestamp, symbol, 'buy', price, amount, 0, "")
                            return order
                        elif order_type == 'sell':
                            order = await self.exchange.create_limit_sell_order(symbol, amount, price)
                            self.order_manager.onAdded(order, order_info)
                            return order
                    except ccxt.BaseError as e:
                        logging.warning(f"Error placing {order_type} amount:{amount} price:{price} order: {e}")
                        return None
                        # await asyncio.sleep(retry_delay)
                return None
                # raise Exception(f"Max retries reached for placing {order_type} order")


            def index_to_sequence(index):
                if index % 2 == 0:
                    # Even index, return negative sequence
                    return -(index // 2 + 1)
                else:
                    # Odd index, return positive sequence
                    return (index // 2 + 1)

            #TODO: CHECK IF THE AMOUNT IS VALID
            order_value = price * amount
            if max_order_value_slice == 0 or order_value <= max_order_value_slice:
                order = await _place_limit_order(self, symbol, order_type, price, amount)
                if not order:
                    return []
                # self.order_manager.onAdded(order)
                return [order]
            else:
                order_slices = []
                i = 0
                remaining_amount = amount
                while remaining_amount > 0:
                    # TODO: CHECK IF THE ORDER SIZE SLICE IS VALID (MINIMUM ORDER SIZE)
                    order_amount = min(remaining_amount, max_order_value_slice / price)
                    # logging.info(f"Placing {symbol}:{order_type} order slice {i} with amount {order_amount}")
                    amount_increment = (self.config["order_size_slice_in_usd"] * index_to_sequence(i))
                    order = await _place_limit_order(self, symbol, order_type, price + amount_increment, order_amount)
                    if not order:
                        break
                    # self.order_manager.onAdded(order)
                    order_slices.append(order)
                    remaining_amount -= order_amount
                    i += 1
                return order_slices

    # async def cancel_all_orders(self, symbol, max_retries=3, retry_delay=5):
    #     for _ in range(max_retries):
    #         try:
    #             open_orders = await self.exchange.fetch_open_orders(symbol)
    #             await self.order_manager.onReconcile(open_orders)
    #             for order in open_orders:
    #                 await self.exchange.cancel_order(order['id'], symbol)
    #                 self.order_manager.onCancelled(order['id'])
    #             return
    #         except ccxt.BaseError as e:
    #             logging.warning(f"Error canceling orders: {e}")
    #             await asyncio.sleep(retry_delay)
    #     raise Exception("Max retries reached for canceling orders")

    # async def get_balance(self, asset, cache=True):
    #     if not cache:
    #         self.balance = await self.exchange.fetch_balance()
    #     if asset in self.balance['free']:
    #         return self.balance['free'][asset], self.balance['used'][asset]
    #     else:
    #         raise ValueError(f"Asset {asset} not found in balance")

    async def calculate_order_size(self, symbol, percentage, order_type):
        asset = symbol.split('/')[0]
        balance_free, balance_used = await self.get_balance(asset)
        
        if order_type not in ['buy', 'sell']:
            raise ValueError("Invalid order type")
        
        return balance_free * percentage / 100

    async def create_symmetrical_order(self, fee, profit_percentage, order):
        # if place_timed_limit_order:
        if not order["price"] or not order["amount"]:
            logging.warning(f"Invalid order info: {order}")
            return None
        
        if order["side"] == 'sell':
            bid_price = TradingTools.calculate_bid_from_ask_with_profit(order["price"], fee, fee, profit_percentage)
            await self.place_timed_limit_order(order["symbol"], 
                                    'buy', 
                                    bid_price, 
                                    order["amount"], 
                                    order_info={"type": "perm", "original_order_id": order["id"]})
        else:
            ask_price = TradingTools.calculate_ask_from_bid_with_profit(order["price"], fee, fee, profit_percentage)
            await self.place_timed_limit_order(order["symbol"], 
                                    'sell', 
                                    ask_price, 
                                    order["amount"], 
                                    order_info={"type": "perm", "original_order_id": order["id"]})

    async def place_stop_loss_order(self, symbol, order_type, stop_price, amount):
        if order_type == 'buy':
            return await self.exchange.create_stop_limit_buy_order(symbol, amount, stop_price, stop_price)
        elif order_type == 'sell':
            return await self.exchange.create_stop_limit_sell_order(symbol, amount, stop_price, stop_price)

    async def dynamic_order_size_strategy(self, symbol, bid_price, ask_price, order_size, volatility_trend):
        
        # SLICING THE ORDER INTO SMALLER ORDERS
        if self.config['order_layering']:
            if volatility_trend == 'growing':
                order_max_size_in_usd = self.config['order_max_size_in_usd']
            elif volatility_trend == 'lowering':
                order_max_size_in_usd = 0 #self.config['order_max_size_in_usd'] * 2
            else:
                order_max_size_in_usd = 0
        else:
            order_max_size_in_usd = 0
        
        order_max_size_in_usd = 0
            
        order_size_buy, order_size_sell = order_size
        
        await asyncio.gather(
            self.place_timed_limit_order(symbol, 'buy', bid_price, order_size_buy, order_max_size_in_usd),
            self.place_timed_limit_order(symbol, 'sell', ask_price, order_size_sell, order_max_size_in_usd),            
        )
        # self.log_trade(timestamp, symbol, 'buy', bid_price, order_size_buy, 0, strategy)
        # self.log_trade(timestamp, symbol, 'sell', ask_price, order_size_sell, 0, strategy)

    async def get_balance(self, asset, cache=True):
        # if not cache:
        #     self.balance = await self.exchange.fetch_balance()
        if asset in self.balance['free']:
            return self.balance['free'][asset], self.balance['used'][asset]
        else:
            raise ValueError(f"Asset {asset} not found in balance")

    async def calculate_total_balance(self, symbol):
        base_asset = symbol.split('/')[0]
        quote_asset = symbol.split('/')[1]
        base_balance_free,base_balance_used = await self.get_balance(base_asset)
        quote_balance_free,quote_balance_used = await self.get_balance(quote_asset)
        
        ticker = await self.exchange.fetch_ticker(symbol)
        base_to_quote_price = ticker['close']
        
        base_balance_in_quote = (base_balance_free + base_balance_used) * base_to_quote_price
        total_balance_in_quote = base_balance_in_quote + (quote_balance_free + quote_balance_used)
        
        return  (base_balance_free + base_balance_used), (quote_balance_free + quote_balance_used), total_balance_in_quote

    async def print_balance(self, symbol, close):
        balance = await self.calculate_total_balance(symbol)
        logging.info(f"Current balance for {symbol}: {balance} :: {balance[2]-self.initial_balance[2]}")
        if self.last_balance and (self.last_balance[0] != balance[0] or self.last_balance[1] != balance[1]):
            self.log_balance(datetime.now().isoformat(), symbol, close, balance[0], balance[1], balance[2], balance[2]-self.initial_balance[2])
        print()
        return balance

    async def cancel_old_orders(self, close, create_simmetrical_order_func, strategy = None):
        
        async def cancel_all_orders(self, symbol, max_retries=3, retry_delay=5):
            for _ in range(max_retries):
                try:
                    open_orders = await self.exchange.fetch_open_orders(symbol)
                    await self.order_manager.onReconcile(open_orders)
                    for order in open_orders:
                        await self.exchange.cancel_order(order['id'], symbol)
                        self.order_manager.onCancelled(order['id'])
                    return
                except ccxt.BaseError as e:
                    logging.warning(f"Error canceling orders: {e}")
                    await asyncio.sleep(retry_delay)
            raise Exception("Max retries reached for canceling orders")        
        
        if not strategy:
            strategy = self.config["cancel_orders"]
        
        open_orders = []
        cancelled_orders = []
        not_cancelled_orders = []
        
        # CANCEL ALL THE ORDERS
        # if not self.config['apply_order_timeout']:
        if strategy == "all":
            await cancel_all_orders(self, self.config['symbol'], max_retries=self.config['max_retries'], retry_delay=self.config['retry_delay'])
            return 0, []

        if strategy == "distance":
            open_orders = await self.exchange.fetch_open_orders(self.config['symbol'])
            # filtering the orders that cannot be deleted: are the orders with the same id in   
            # open_orders = [order for order in open_orders if not self.order_manager.getType(order['id']) in ['perm']]
            open_orders = self.order_manager.filter_perm_orders(open_orders)
            
            logging.info(f"Open orders: {len(open_orders)}")
            await self.order_manager.onReconcile(open_orders, create_simmetrical_order_func)
            if len(open_orders) == 0:
                return 0, []
            
            open_orders = sorted(open_orders, key=lambda x: abs(x['price'] - close), reverse=True)
            num_orders = len(open_orders)
            if num_orders < self.config['max_orders']:
                return num_orders, open_orders
                
            diff = 2 + max(0,num_orders - self.config['max_orders'])
            for order in open_orders[:diff]:
                await self.exchange.cancel_order(order['id'], self.config['symbol'])
                cancelled_orders.append(order['id'])
                self.order_manager.onCancelled(order['id'])
                logging.info(f"Cancelled order {order['id']} due to distance for {self.config['symbol']}")
            not_cancelled_orders = [order for order in open_orders if order['id'] not in cancelled_orders]
            # TODO: those orders must be filtered by the ones that are not perm
            return num_orders - diff, not_cancelled_orders

        if strategy == "oldest":
            # CANCEL OLDEST ORDERSj
            open_orders = await self.exchange.fetch_open_orders(self.config['symbol'])
            open_orders = self.order_manager.filter_perm_orders(open_orders)
            
            logging.info(f"Open orders: {len(open_orders)}")
            await self.order_manager.onReconcile(open_orders, create_simmetrical_order_func)
            if len(open_orders) == 0:
                return 0, []
            
            current_time = datetime.now()
            cancelled = 0
            for order in open_orders:
                order_id = order['id']
                order_timestamp = datetime.fromtimestamp(order['timestamp'] / 1000)
                if (current_time - order_timestamp).total_seconds() > self.config['order_timeout']:
                    await self.exchange.cancel_order(order_id, self.config['symbol'])
                    cancelled += 1
                    cancelled_orders.append(order['id'])
                    self.order_manager.onCancelled(order['id'])
                    logging.info(f"Cancelled order {order_id} due to timeout for {self.config['symbol']}")

            # TODO: those orders must be filtered by the ones that are not perm
            not_cancelled_orders = [order for order in open_orders if order['id'] not in cancelled_orders]

            return len(open_orders) - cancelled, not_cancelled_orders

    async def order_strategy(self, bid_price, ask_price, volatility_trend_label):
        order_size_buy = await self.calculate_order_size(self.config['symbol'], self.config['order_size'], "buy")
        order_size_sell = await self.calculate_order_size(self.config['symbol'], self.config['order_size'], "sell")
        order_size = (order_size_buy, order_size_sell)
        # compare sizes with the current balance and then take a decision
        await self.dynamic_order_size_strategy(self.config['symbol'], bid_price, ask_price, order_size,  volatility_trend=volatility_trend_label)

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

    async def res_supp_opportunities(self, df, limit, base_fee, desired_profit):
        resistances, supports = TradingTools.find_support_resistance(df.tail(limit))
        spreads_df = TradingTools.calculate_spreads(resistances, supports)
        if len(spreads_df) == 0:
            return []
        filtered_spread_df = spreads_df[spreads_df['Spread'] > 0]
        resistance_support_tuples = list(zip(filtered_spread_df['Resistance'].tolist(), filtered_spread_df['Support'].tolist())) 
                    
        # TRANSFORM RESISTANCE AND SUPPORTS IN ASK AND BID
        ask_bid_from_res_supp = []
        for resistance, support in resistance_support_tuples:
            profit_percentage = TradingTools.calculate_profit_percentage_from_bid_ask(resistance, support, base_fee, base_fee)
            if profit_percentage >= desired_profit:
                ask_bid_from_res_supp.append((resistance, support, profit_percentage))
        return ask_bid_from_res_supp

    async def get_candles(self, limit=10):
        
        async def get_latest_ohlcv(self, symbol, timeframe='1m', limit=2, max_retries=3, retry_delay=5):
            for _ in range(max_retries):
                try:
                    return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                except ccxt.BaseError as e:
                    logging.warning(f"Error fetching OHLCV data: {e}")
                    await asyncio.sleep(retry_delay)
            raise Exception("Max retries reached for fetching OHLCV data")        
        
        
        ohlcv = await get_latest_ohlcv(self, self.config['symbol'], "1m", limit=limit, max_retries=self.config['max_retries'], retry_delay=self.config['retry_delay'])
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        return df

    async def latest_candle(self, df):
        latest_candle = df.iloc[-1]
        high = latest_candle['High']
        low = latest_candle['Low']
        open = latest_candle['Open']
        close = latest_candle['Close']
        return high, low, open, close

    def is_ask_bid_between_reference_price(self, reference_price, ask, bid):
        return  reference_price < ask, bid < reference_price, (bid < reference_price < ask)

    def ask_bid_acceptance(self, reference_price, ask_bid_list, ask, bid, profit) -> list:

        if not self.ask_bid_between_reference_price:
            ask_bid_list.append((ask, bid, profit))
        else:
            ask_ok, bid_ok, askbid_ok = self.is_ask_bid_between_reference_price(reference_price=reference_price, ask=ask, bid=bid)
            if askbid_ok:
                ask_bid_list.append((ask, bid, profit))
            elif self.accept_just_ask_or_bid_orders:
                if ask_ok:
                    ask_bid_list.append((ask, None, None))
                elif bid_ok:
                    ask_bid_list.append((None, bid, None))
        return ask_bid_list

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
        
        if self.config['demo_trading']:
            self.exchange.enable_demo_trading(True)
        
        markets = await self.exchange.load_markets()
        market_symbol = markets[self.config['symbol']]
        self.maker_fee = market_symbol['maker']
        desired_profit = self.config["desired_profit_percentage"] / 100

        self.method = self.config['reference_price_method']
        self.ask_bid_between_reference_price = self.config['ask_bid_between_reference_price']
        self.accept_just_ask_or_bid_orders = self.config['accept_just_ask_or_bid_orders']

        # ohlcv = await self.get_latest_ohlcv(self.config['symbol'], timeframe='1m', limit=10, max_retries=self.config['max_retries'], retry_delay=self.config['retry_delay'])
        df = await self.get_candles(limit=10)
        high, low, open, close = await self.latest_candle(df)

        self.symbol = self.config['symbol']
        self.balance = await self.exchange.fetch_balance()
        
        self.initial_balance = await self.calculate_total_balance(self.config['symbol'])
        self.last_balance = None
        self.last_balance = await self.print_balance(self.config['symbol'], close)        
        print()

        limit=self.config['max_candles']

        symbol = self.config['symbol']
        last_time_checked_simulated_order_execution = time.time()
        
        create_simmetrical_order_func = partial(self.create_symmetrical_order, self.maker_fee, desired_profit)
        
        while True:
            
            try:

                start_time = time.time()
                self.balance = await self.exchange.fetch_balance()
                # await self.get_balance(self.config['symbol'], cache=False)
                
                # TODO: we can read less candles and build the dataframe merging the results 
                df = await self.get_candles(limit=limit)
                high, low, open, close = await self.latest_candle(df)
                
                if is_paper:
                    if time.time() - last_time_checked_simulated_order_execution >= 60:
                        self.exchange.simulate_order_execution(symbol, close)
                        last_time_checked_simulated_order_execution = time.time()

                # ORDERS TO CANCEL
                num_orders, not_cancelled_orders = await self.cancel_old_orders(close,create_simmetrical_order_func)
                # orders_profit, num_orders = self.order_manager.reduce_filled(self.maker_fee)
                # logging.info(f"Orders profit: {orders_profit} - Number of orders: {num_orders}")
                if self.config['max_orders'] > 0 and num_orders >= self.config['max_orders']:
                    num_orders, not_cancelled_orders = await self.cancel_old_orders(close,None,strategy='distance')
                    if self.config['max_orders'] > 0 and num_orders >= self.config['max_orders']:
                        logging.info(f"Reached maximum number of open orders: {self.config['max_orders']} for {symbol}")
                        print()
                        await asyncio.sleep(10)
                        continue

                 # REFERENCE PRICE
                # method = "Median"

                reference_price =  TradingTools.calculate_reference_price(df, method=self.method, period=14)
                
                # SUPPORT AND RESISTANCES
                _ask_bid_candidates = await self.res_supp_opportunities(df, limit, self.maker_fee, desired_profit)
                # REFERENCE PRICE MUST BE BETWEEN ASK AND BID
                ask_bid_candidates = []
                for ask, bid, profit in _ask_bid_candidates:
                    ask_bid_candidates = self.ask_bid_acceptance(reference_price, ask_bid_candidates, ask, bid, profit)
                    # if not self.ask_bid_between_reference_price:
                    #     ask_bid_candidates.append((ask, bid, profit))
                    # else:
                    #     # ask_bid_candidates = [(ask, bid, profit) for ask, bid, profit in ask_bid_candidates if not (bid < reference_price < ask)]
                    #     ask_ok, bid_ok, askbid_ok = self.is_ask_bid_between_reference_price(reference_price=reference_price, ask=ask, bid=bid)
                    #     if askbid_ok:
                    #         ask_bid_candidates.append((ask, bid, profit))
                    #     elif self.accept_just_ask_or_bid_orders
                    #         if ask_ok:
                    #             ask_bid_candidates.append((ask, None, None))
                    #         elif bid_ok:
                    #             ask_bid_candidates.append((None, bid, None))
                
                # VOLATILITY   
                # current_volatility = TradingTools.calculate_high_low_volatility(high, low)
                # volatility = (current_volatility + average_volatility) / 2
                # average_volatility = volatility
                # logging.info(f"Current volatility for {self.config['symbol']}: {volatility}")

                # ATR
                atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
                _,_,atr_increment_percentage,volatility_trend = TradingTools.calculate_volatility_trend_indicator(atr)
                # atr_increment_percentage = TradingTools.calculate_increment_percentage(current_atr,previous_atr)
                logging.info(f"ATR: {atr.iloc[-1]:.2f} {volatility_trend} {atr_increment_percentage:.2f}%")

                # RSI
                rsi = ta.momentum.rsi(df['Close'], window=14)
                self.last_rsi = rsi.iloc[-1]
                last_rsi_info = ""
                if self.last_rsi > 80:
                    last_rsi_info = "overbought"
                elif self.last_rsi < 20:
                    last_rsi_info = "oversold"
                logging.info(f"RSI: {self.last_rsi:.2f} - {last_rsi_info}")
                
                oscillator = self.last_rsi/100
                
                bid_price, ask_price = TradingTools.calculate_bid_ask_with_profit(reference_price, self.maker_fee, self.maker_fee, desired_profit, oscillator)
                profit_percentage = TradingTools.calculate_profit_percentage_from_bid_ask(bid_price, ask_price, self.maker_fee, self.maker_fee)
                if profit_percentage >= desired_profit:
                    ask_bid_candidates.append((ask_price, bid_price, profit_percentage))
                
                # PREPARING ORDERS
                ask_bid = self.reduce_ask_bid(not_cancelled_orders, ask_bid_candidates, descending=volatility_trend == 'growing')
                for ask, bid, profit in ask_bid:
                    print(f"{ask:.2f}", f"{bid:.2f}", f"{profit:.2f}")
                   
                # NO OPPORTUNITIES
                profit_to_achieve = desired_profit
                while len(ask_bid) == 0:
                    bid_price, ask_price = TradingTools.calculate_bid_ask_with_profit(reference_price, self.maker_fee, self.maker_fee, desired_profit, oscillator)

                    profit_percentage = TradingTools.calculate_profit_percentage_from_bid_ask(bid_price, ask_price, self.maker_fee, self.maker_fee)
                    if profit_percentage >= desired_profit:
                        ask_bid.append((ask_price, bid_price, profit_percentage))
                    desired_profit += 0.005
                    
                # PERFORMING ORDERS
                for ask, bid, profit in ask_bid:
                    
                    profit_percentage = TradingTools.calculate_profit_percentage_from_bid_ask(bid_price, ask_price, self.maker_fee, self.maker_fee)
                    if profit_percentage < profit_to_achieve:
                        continue
                    
                    await self.order_strategy(bid, ask, volatility_trend)
                    # print(f"{ask:.2f}", f"{bid:.2f}", f"{profit:.2f}")
                    num_orders += 2
                    if num_orders >= self.config['max_orders']:
                        break
                    
                self.last_balance = await self.print_balance(self.config['symbol'], close)
                
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
