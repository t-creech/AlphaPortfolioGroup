from imports import *

# ------------------------ DATA PIPELINE -----------------------------
class AlphaPortfolioData(Dataset):
    def __init__(self, lookback=12, T=12, num_training_blocks=4, test_year_start = 2015):
        """
        Initializes the dataset.
        
        For each asset, we load historical data from CRSP and Compustat via WRDS.
        
        We then build sequential episodes for RL. Each episode has T time steps.
        At each time step t:
          - The state is the historical window of length 'lookback' for all assets.
            (Shape: [num_assets, lookback, num_features])
          - The forward (one‐month) return for each asset is extracted.
            (Shape: [num_assets])
          - A mask indicates whether the asset has complete data.
        
        Overall, each episode is composed of:
          - state_seq: (T, num_assets, lookback, num_features)
          - fwd_seq: (T, num_assets)
          - mask_seq: (T, num_assets)
        """
        super().__init__()
        self.lookback = lookback
        self.T = T
        self.num_training_blocks = num_training_blocks
        self.train_sequences = []
        self.val_sequences = []
        self.test_sequences = []
        self.train_future_returns = []
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        self.merged = self._load_wrds_data()
        self.merged["date"] = pd.to_datetime(self.merged["date"])
        self.test_year_start = test_year_start
        self.unique_permnos = sorted(self.merged['permno'].unique())
        self.global_max_assets = len(self.unique_permnos)
        self.permno_to_idx = {permno: idx for idx, permno in enumerate(self.unique_permnos)}
        self.train_data, self.val_data, self.test_data = self._split_data()
        self.normalized_train_data, normalization_params, self.num_features = self._normalize_training_data()
        self.normalized_val_data = self._normalize_val_data(normalization_params)
        self.normalized_test_data = self._normalize_test_data(normalization_params)
        logger.info(f"Dataset initialized: {len(self.normalized_train_data)} training rounds created.")
        logger.info(f"Dataset initialized: {len(self.normalized_val_data)} validation rounds created.")
        logger.info(f"Dataset initialized: {len(self.normalized_test_data)} test rounds created.")
        for i in range(len(self.normalized_train_data)):
            logger.info(f"Training round {i} contains {len(self.normalized_train_data[i])} training samples.")
            logger.info(f"Training round {i} contains data from {self.normalized_train_data[i]['date'].dt.year.min()} to {self.normalized_train_data[i]['date'].dt.year.max()}.")
            sequences, future_returns, masks = self._train_create_sequences(self.normalized_train_data[i])
            self.train_sequences.append(sequences)
            self.train_future_returns.append(future_returns)
            self.train_masks.append(masks)
        for i in range(len(self.normalized_val_data)):
            logger.info(f"Validation round {i} contains {len(self.normalized_val_data[i])} training samples.")
            logger.info(f"Validation round {i} contains data from {self.normalized_val_data[i]['date'].dt.year.min()} to {self.normalized_val_data[i]['date'].dt.year.max()}.")
            sequences, masks = self._test_create_sequences(self.normalized_val_data[i])
            self.val_sequences.append(sequences)
            self.val_masks.append(masks)
        sequences, masks = self._test_create_sequences(self.normalized_test_data)
        logger.info(f"Test round contains {len(self.normalized_test_data)} training samples.")
        logger.info(f"Test round contains data from {self.normalized_test_data['date'].dt.year.min()} to {self.normalized_test_data['date'].dt.year.max()}.")
        self.test_sequences = sequences
        self.test_masks = masks

    def _load_wrds_data(self):
        first_data = pd.read_csv('Data/cleanedFinalData_1975-1984.csv')
        second_data = pd.read_csv('Data/cleanedFinalData_1985-1994.csv')
        third_data = pd.read_csv('Data/cleanedFinalData_1995-2004.csv')
        fourth_data = pd.read_csv('Data/cleanedFinalData_2005-2014.csv')
        fifth_data = pd.read_csv('Data/cleanedFinalData_2015-2024.csv')
        merged = pd.concat([first_data, second_data, third_data, fourth_data, fifth_data])
        return merged
      
    def _split_data(self):
        """
        Splits the merged data into training and testing sets.
        """
        data = self.merged
        first_year = data['date'].dt.year.min()
        last_year = self.test_year_start
        training_block_length = (last_year - first_year) // self.num_training_blocks
        train_data = []
        val_data = []
        test_data = data[data['date'].dt.year.isin(range(self.test_year_start, data['date'].dt.year.max() + 1))]
        for i in range(self.num_training_blocks):
            train_data.append(data[data['date'].dt.year.isin(range(first_year + i * training_block_length, first_year + (i+1) * training_block_length))])
            val_data.append(data[data['date'].dt.year.isin(range(first_year + (i+1) * training_block_length, first_year + ((i+1) * training_block_length) + 5))])
        return train_data, val_data, test_data
      
      
    def _normalize_training_data(self):
      """
      Normalizes the training data and returns the normalized data and the normalization parameters.
      """
      data = self.train_data
      normalization_params = {}
      features_to_not_normalize = ['permno', 'ret', 'date']
      for i in range(len(data)):
          data[i] = data[i].drop(columns=['gvkey', 'year', 'rdq'])
          num_features = len(data[i].columns) - 1
          normalization_params[i] = {}
          for column in data[i].columns:
              if column not in features_to_not_normalize:
                  mean = data[i][column].mean()
                  std = data[i][column].std()
                  data[i][column] = (data[i][column] - mean) / std
                  # Add normalization parameters to dictionary for ith block
                  normalization_params[i][column] = (mean, std)
      return data, normalization_params, num_features
    
    def _normalize_val_data(self, normalization_params):
      """
      Normalizes the validation and test data using the normalization parameters.
      """
      data = self.val_data
      features_to_not_normalize = ['permno', 'ret', 'date']
      for i in range(len(data)):
          data[i] = data[i].drop(columns=['gvkey', 'year', 'rdq'])
          this_set_normalization_params = normalization_params[i]
          for column in data[i].columns:
              if column not in features_to_not_normalize:
                  mean, std = this_set_normalization_params[column]
                  data[i][column] = (data[i][column] - mean) / std
      return data
    
    def _normalize_test_data(self, normalization_params):
      """
      Normalizes the validation and test data using the normalization parameters.
      """
      data = self.test_data
      features_to_not_normalize = ['permno', 'ret', 'date']
      test_set_normalization_params = normalization_params[self.num_training_blocks - 1]
      data = data.drop(columns=['gvkey', 'year', 'rdq'])
      for column in data.columns:
          if column not in features_to_not_normalize:
              mean, std = test_set_normalization_params[column]
              data[column] = (data[column] - mean) / std
      return data

    def _train_create_sequences(self, data):
        """
        Creates sequential episodes for RL.
        
        For each episode, we use a sliding window over the sorted unique dates.
        Let T = self.T (number of rebalancing steps per episode).
        For an episode starting at index i, for each time step t (0 <= t < T):
          - The state is the data from date index i+t to i+t+lookback-1.
          - The one-month forward return is taken from date index i+t+lookback.
        
        This yields:
          - state_seq: (T, num_assets, lookback, num_features)
          - fwd_seq:   (T, num_assets)
          - mask_seq:  (T, num_assets)
        """
        lookback = self.lookback
        T = self.T
        unique_dates = pd.to_datetime(data['date'].unique())
        unique_dates_sorted = np.sort(unique_dates)
        num_features = self.num_features  # Here we use: 'permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq'
        
        episodes_states = []
        episodes_fwd = []
        episodes_masks = []
        num_episodes = len(unique_dates_sorted) - (2 * lookback) + 1
        logger.info(f"Creating {num_episodes} sequential episodes (T = {T} time steps each).")
        for start_idx in tqdm(range(num_episodes), desc="Creating sequential episodes"):
            episode_states = []  # shape: (T, global_max_assets, lookback, num_features)
            episode_fwd = []     # shape: (T, global_max_assets)
            episode_masks = []   # shape: (T, global_max_assets)
            for t in range(T):
                state_start = start_idx + t
                state_end = state_start + lookback
                fwd_index = state_end
                step_states = np.zeros((self.global_max_assets, lookback, num_features))
                step_fwd = np.zeros((self.global_max_assets,))
                step_mask = np.zeros((self.global_max_assets,), dtype=bool)
                for permno in self.unique_permnos:
                    idx = self.permno_to_idx[permno]
                    hist_data = data[
                        (data['permno'] == permno) &
                        (data['date'] >= unique_dates_sorted[state_start]) &
                        (data['date'] < unique_dates_sorted[state_end])
                    ].sort_values('date')
                    fwd_data = data[
                        (data['permno'] == permno) &
                        (data['date'] == unique_dates_sorted[fwd_index])
                    ]
                    if len(hist_data) == lookback and len(fwd_data) == 1:
                        features_list = [col for col in hist_data.columns if col not in ['date']]
                        features = hist_data[features_list].values
                        step_states[idx] = features
                        step_fwd[idx] = fwd_data['ret'].values[0]
                        step_mask[idx] = True
                episode_states.append(step_states)
                episode_fwd.append(step_fwd)
                episode_masks.append(step_mask)
            episode_states = np.array(episode_states)   # (T, global_max_assets, lookback, num_features)
            episode_fwd = np.array(episode_fwd)           # (T, global_max_assets)
            episode_masks = np.array(episode_masks)       # (T, global_max_assets)
            episodes_states.append(episode_states)
            episodes_fwd.append(episode_fwd)
            episodes_masks.append(episode_masks)
        sequences_tensor = torch.tensor(np.array(episodes_states), dtype=torch.float32)
        future_returns_tensor = torch.tensor(np.array(episodes_fwd), dtype=torch.float32)
        masks_tensor = torch.tensor(np.array(episodes_masks), dtype=torch.bool)
        logger.info(f"Created sequences tensor shape: {sequences_tensor.shape}")
        logger.info(f"Created future_returns tensor shape: {future_returns_tensor.shape}")
        logger.info(f"Created masks tensor shape: {masks_tensor.shape}")
        return sequences_tensor, future_returns_tensor, masks_tensor
      
    def _test_create_sequences(self, data):
        """
        Creates sequential episodes for RL.
        
        For each episode, we use a sliding window over the sorted unique dates.
        Let
        T = self.T (number of rebalancing steps per episode).
        For an episode starting at index i, for each time step t (0 <= t < T):
          - The state is the data from date index i+t to i+t+lookback-1.
          - The one-month forward return is taken from date index i+t+lookback.
        """
        lookback = self.lookback
        T = self.T
        unique_dates = pd.to_datetime(data['date'].unique())
        unique_dates_sorted = np.sort(unique_dates)
        num_features = self.num_features  # Here we use: 'permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq'
        
        episodes_states = []
        episodes_masks = []
        num_episodes = len(unique_dates_sorted) - (2 * lookback) + 1
        logger.info(f"Creating {num_episodes} sequential episodes (T = {T} time steps each).")
        for start_idx in tqdm(range(num_episodes), desc="Creating sequential episodes"):
            episode_states = []  # shape: (T, global_max_assets, lookback, num_features)
            episode_masks = []   # shape: (T, global_max_assets)
            for t in range(T):
                state_start = start_idx + t
                state_end = state_start + lookback
                fwd_index = state_end
                step_states = np.zeros((self.global_max_assets, lookback, num_features))
                step_fwd = np.zeros((self.global_max_assets,))
                step_mask = np.zeros((self.global_max_assets,), dtype=bool)
                for permno in self.unique_permnos:
                    idx = self.permno_to_idx[permno]
                    hist_data = data[
                        (data['permno'] == permno) &
                        (data['date'] >= unique_dates_sorted[state_start]) &
                        (data['date'] < unique_dates_sorted[state_end])
                    ].sort_values('date')
                    fwd_data = data[
                        (data['permno'] == permno) &
                        (data['date'] == unique_dates_sorted[fwd_index])
                    ]
                    if len(hist_data) == lookback and len(fwd_data) == 1:
                        features_list = [col for col in hist_data.columns if col not in ['date']]
                        features = hist_data[features_list].values
                        step_states[idx] = features
                        step_fwd[idx] = fwd_data['ret'].values[0]
                        step_mask[idx] = True
                episode_states.append(step_states)
                episode_masks.append(step_mask)
            episode_states = np.array(episode_states)   # (T, global_max_assets, lookback, num_features)
            episode_masks = np.array(episode_masks)       # (T, global_max_assets)
            episodes_states.append(episode_states)
            episodes_masks.append(episode_masks)
        sequences_tensor = torch.tensor(np.array(episodes_states), dtype=torch.float32)
        masks_tensor = torch.tensor(np.array(episodes_masks), dtype=torch.bool)
        logger.info(f"Created sequences tensor shape: {sequences_tensor.shape}")
        logger.info(f"Created masks tensor shape: {masks_tensor.shape}")
        return sequences_tensor, masks_tensor


class RoundDataset(Dataset):
    def __init__(self, sequences, future_returns, masks):
        self.sequences = sequences
        self.future_returns = future_returns
        self.masks = masks

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        # If future_returns is not needed (e.g., for validation) you can return None or adjust accordingly.
        return self.sequences[idx], self.future_returns[idx] if self.future_returns is not None else None, self.masks[idx]
