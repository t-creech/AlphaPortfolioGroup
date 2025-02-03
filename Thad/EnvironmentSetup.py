import numpy as np
import pandas as pd

class MarketEnvironment:
    def __init__(self, historical_data, lookback_window=12):
        """
        Initialize the market environment.

        :param historical_data: DataFrame with columns ['Date', 'Asset', 'Feature1', ..., 'FeatureN', 'Return']
        :param lookback_window: Number of months to look back for state representation.
        """
        self.historical_data = historical_data
        self.lookback_window = lookback_window
        self.assets = historical_data['Asset'].unique()
        self.dates = sorted(historical_data['Date'].unique())
        self.current_index = lookback_window

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.current_index = self.lookback_window
        state = self._get_state()
        return state

    def step(self, action):
        """
        Take a step in the environment.

        :param action: Portfolio weights for each asset at the current time step.
        :return: Tuple (state, reward, done, info)
        """
        if self.current_index >= len(self.dates) - 1:
            raise ValueError("Environment has reached the end of the dataset. Reset the environment.")

        # Calculate reward based on the action (portfolio returns)
        returns = self._get_returns()
        reward = np.dot(action, returns)

        # Move to the next time step
        self.current_index += 1
        state = self._get_state()
        done = self.current_index >= len(self.dates) - 1

        info = {
            "date": self.dates[self.current_index],
            "returns": returns,
            "portfolio_weights": action
        }
        return state, reward, done, info

    def _get_state(self):
        """
        Get the current state representation.

        :return: Dictionary with asset features for the lookback window.
        """
        current_date = self.dates[self.current_index]
        start_date = self.dates[self.current_index - self.lookback_window]

        state_data = self.historical_data[
            (self.historical_data['Date'] >= start_date) &
            (self.historical_data['Date'] < current_date)
        ]

        state = {
            asset: state_data[state_data['Asset'] == asset].iloc[:, 2:-1].values
            for asset in self.assets
        }
        return state

    def _get_returns(self):
        """
        Get the returns for the current date.

        :return: Array of returns for each asset.
        """
        current_date = self.dates[self.current_index]
        returns = self.historical_data[
            self.historical_data['Date'] == current_date
        ].set_index('Asset')['Return'].reindex(self.assets).fillna(0).values
        return returns

# Example usage:
# Assuming historical_data is a DataFrame with columns ['Date', 'Asset', 'Feature1', ..., 'FeatureN', 'Return']
# historical_data = pd.read_csv('historical_data.csv')
# env = MarketEnvironment(historical_data)
# state = env.reset()
# action = np.random.uniform(0, 1, len(env.assets))
# action /= np.sum(action)  # Normalize to sum to 1
# next_state, reward, done, info = env.step(action)