import numpy as np
import random
import time
import logging
import matplotlib.pyplot as plt


def new_order(transaction_type, price=None, mu_seller_addition=40):
    if price is None:
        price = (
            np.floor(np.random.normal(100, 20, 1)[0])  # Change
            if transaction_type == 'Buy'
            else np.floor(np.random.normal(100 + mu_seller_addition, 20, 1)[0])  # Change
        )
    volume = random.randint(1, 11)  # Change
    timestamp = time.time()
    order = {
        'price': price,
        'volume': volume,
        'type': transaction_type,
        'timestamp': timestamp,
    }
    return order


def add_new_order_to_book(order):
    global buy_orders, sell_orders
    price, transaction_type = order['price'], order['type']
    if transaction_type == 'Buy':
        buy_orders.append(order)
    elif transaction_type == 'Sell':
        sell_orders.append(order)

    buy_orders = sorted(sorted(buy_orders, key=lambda x: x['timestamp']), key=lambda x: x['price'], reverse=True)
    sell_orders = sorted(sorted(sell_orders, key=lambda x: x['timestamp']), key=lambda x: x['price'])
    return buy_orders, sell_orders


def transaction(spread_limit):
    global buy_orders, sell_orders, price_history, volume_history
    while (
            ((len(buy_orders) > 0 and len(sell_orders) > 0) and
             (sell_orders[0]['price'] - buy_orders[0]['price']) <= spread_limit)):

        buy_price, buy_volume = buy_orders[0]['price'], buy_orders[0]['volume']
        sell_price, sell_volume = sell_orders[0]['price'], sell_orders[0]['volume']
        spread = sell_price - buy_price

        trade_volume = min(buy_volume, sell_volume)
        buy_orders[0]['volume'] -= trade_volume
        sell_orders[0]['volume'] -= trade_volume
        logging.debug(f"BUY: {buy_volume} at {buy_price}. SELL: {sell_volume} at {sell_price}. SPREAD: {spread}")

        # Append new price and volume to histories
        price_history.append((buy_price + sell_price) / 2)
        volume_history.append(trade_volume)

        # Delete filled orders
        if buy_orders[0]['volume'] == 0:
            del buy_orders[0]
        if sell_orders[0]['volume'] == 0:
            del sell_orders[0]


class Player1:
    def __init__(self, id, w, price_fundamental, n_lookback, noise_std):  # Have weights be a vector
        self.id = id
        self.weights_history = {"Fundamental": [], "Technical": []}  # , "Noise": []}
        self.w = w  # Fundamental, technical, noise
        self.v1 = np.log(price_fundamental / price_history[-1])
        self.v2 = np.log(price_history[-1] / price_history[-1 - n_lookback])
        self.v3 = float(np.random.normal(0, noise_std, 1))
        self.v = np.array([self.v1, self.v2])  # , self.v3])

    def set_expected_return(self, w, v):
        expected_return = v.dot(w) / sum(w)
        self.expected_return = expected_return

    def set_expected_price(self, expected_return):
        self.expected_price = price_history[-1] * np.exp(expected_return)

    def create_order(self, price_std):
        price_order = np.random.normal(self.expected_price, price_std)
        # Place sell order
        if price_order > self.expected_price:
            sell_order = new_order(transaction_type='Sell', price=price_order)
            add_new_order_to_book(sell_order)
        # Place buy order
        elif price_order < self.expected_price:
            buy_order = new_order(transaction_type='Buy', price=price_order)
            add_new_order_to_book(buy_order)

    def update_weights_with_gradient(self, w, v, scalar=0.01):
        gradient = np.array([
            price_history[-1] - v[0],
            price_history[-1] - v[1],
            # price_history[-1] - v[2]
        ]) * price_history[-2]

        # Determine sign of the gradient
        cost = (price_history[-1] - self.expected_price) ** 2

        new_w = w + scalar * gradient
        new_expected_return = v.dot(new_w) / sum(new_w)
        new_expected_price = price_history[-1] * np.exp(new_expected_return)
        new_cost = (price_history[-1] - new_expected_price) ** 2

        # Compare new cost to old cost. Update weights accordingly
        if new_cost < cost:
            self.w = new_w
        else:
            self.w = w - scalar * gradient

        # Record weights history
        for n in range(2):  # range(3):
            self.weights_history[["Fundamental", "Technical"][n]].append(self.w[n] / sum(self.w))


def init_agents(population_size, price_fundamental, n_lookback, noise_std):
    return [
        Player1(id=id, w=np.random.rand(2, ), price_fundamental=price_fundamental, n_lookback=n_lookback,
                noise_std=noise_std)
        for id in range(population_size)
    ]


def run_market(population_size, price_fundamental, n_lookback, noise_std, n_transactions):
    # Initialise agents
    agents = init_agents(population_size, price_fundamental, n_lookback, noise_std)

    # Loop through N transactions. Each loop has:
    # (1) agents making predictions and placing orders,
    # (2) orders executing through the order book which sets a new price,
    # (3) agents updating their weights based on this new price
    for _ in range(n_transactions):
        for a in agents:
            a.set_expected_return(w=a.w, v=a.v)
            a.set_expected_price(expected_return=a.expected_return)
            a.create_order(price_std=price_std)
            # print(f"Fundamental: {a.w[0]} Technical: {a.w[1]} Noise: {a.w[2]}")

        # Clear the order book and determine a new price
        transaction(spread_limit=spread_limit)

        # Update agents' weights
        for a in agents:
            a.update_weights_with_gradient(w=a.w, v=a.v, scalar=0.0001)
            print(f"Agent {a.id} --> Fundamental: {a.w[0]:.2f} Technical: {a.w[1]:.2f}")  # Noise: {a.w[2]:.2f}")
    return agents


# Settings
buy_orders, sell_orders, volume_history = [], [], []
price_fundamental = int(np.random.normal(100, 10, 1))  # Set price around 100
n_lookback = 1  # Lookback period for technical traders
price_history = [price_fundamental] * (n_lookback + 1)  # Initiate a price history that's long enough
noise_std = 0.1  # Standard deviation of noise parameter; Larger = More noise
price_std = 1  # Standard deviation of order price; Says how much the order price can deviate from the expected price
spread_limit = 0.5  # Spread limit below which trades can occur
n_transactions = 100
population_size = 100

# Fire!
agents = run_market(population_size, price_fundamental, n_lookback, noise_std, n_transactions)

# Plot weights
for agent in agents:
    plt.plot(agent.weights_history["Fundamental"], 'g')
    plt.plot(agent.weights_history["Technical"], 'b')
    # plt.plot(agent.weights_history["Noise"], 'r')
    # plt.plot(agent.weights_history["Technical"])
