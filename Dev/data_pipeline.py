from imports import *

# ------------------------ DATA PIPELINE -----------------------------
class AlphaPortfolioData(Dataset):
    def __init__(self, start_year=2014, end_year=2020, final_year=2016, lookback=12, T=12):
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
        self.merged, self.final_data = self._load_wrds_data(start_year, end_year, final_year)
        self.unique_permnos = sorted(self.final_data['permno'].unique())
        self.global_max_assets = len(self.unique_permnos)
        self.permno_to_idx = {permno: idx for idx, permno in enumerate(self.unique_permnos)}
        self.sequences, self.future_returns, self.masks = self._create_sequences()
        logger.info(f"Dataset initialized: {len(self.sequences)} episodes created.")
        if len(self.sequences) > 0:
            logger.info(f"Example episode state shape: {self.sequences[0].shape} "
                        f"(T, num_assets, lookback, num_features)")
            logger.info(f"Example episode future returns shape: {self.future_returns[0].shape} "
                        f"(T, num_assets)")
            logger.info(f"Example episode mask shape: {self.masks[0].shape} "
                        f"(T, num_assets)")

    def _load_wrds_data(self, start_year, end_year, final_year):
        permno_list = []
        combined_data = pd.DataFrame()
        for year in range(start_year, end_year+1):
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            crsp_query = f"""
                SELECT a.permno, a.date, a.ret, a.prc, a.shrout, 
                       a.vol, a.cfacshr, a.altprc, a.retx
                FROM crsp.msf AS a
                WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
                  AND a.permno IN (
                      SELECT permno FROM crsp.msenames 
                      WHERE exchcd BETWEEN 1 AND 3  
                        AND shrcd IN (10, 11)
                  )
                """
            crsp_data = db.raw_sql(crsp_query)
            query_ticker = """
                SELECT permno, namedt, nameenddt, ticker
                FROM crsp.stocknames
            """
            stocknames = db.raw_sql(query_ticker)
            crsp_data = crsp_data.merge(stocknames.drop_duplicates(subset=['permno']), on='permno', how='left')
            crsp_data = crsp_data.dropna(subset=['ticker'])
            crsp_data['mktcap'] = (crsp_data['prc'].abs() * crsp_data['shrout'] * 1000) / 1e6
            crsp_data['year'] = pd.to_datetime(crsp_data['date']).dt.year
            crsp_data = crsp_data.dropna(subset=['mktcap'])
            top_50_permnos_by_year = crsp_data.groupby('permno')['mktcap'].agg(['max']).reset_index()\
                                     .sort_values(by='max', ascending=False).head(50)['permno'].unique()
            permno_list.extend(top_50_permnos_by_year)
            combined_data = pd.concat([combined_data, crsp_data[crsp_data['permno'].isin(permno_list)]], axis=0)
        combined_data = combined_data[['permno', 'ticker', 'date', 'ret', 'prc', 'shrout', 'vol', 'mktcap', 'year']]
        combined_data['date'] = pd.to_datetime(combined_data['date'])

        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-31'
        fund_query = f"""
            SELECT gvkey, datadate, rdq, saleq
            FROM comp.fundq
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
              AND datadate BETWEEN '{start_date}' AND '{end_date}'
              AND rdq IS NOT NULL
            """
        fund = db.raw_sql(fund_query)
        fund['rdq'] = pd.to_datetime(fund['rdq'])
        fund['datadate'] = pd.to_datetime(fund['datadate'])
        link_query = """
            SELECT lpermno AS permno, gvkey, linkdt, linkenddt
            FROM crsp.ccmxpf_linktable
            WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
            """
        link = db.raw_sql(link_query)
        fund = pd.merge(fund, link, on='gvkey', how='left')
        fund = fund.dropna(subset=['permno'])
        combined_data_sorted = combined_data.sort_values('date')
        fund_sorted = fund.sort_values('rdq')
        fund_sorted['permno'] = fund_sorted['permno'].astype(int)
        merged = pd.merge_asof(
            combined_data_sorted,
            fund_sorted,
            left_on='date',
            right_on='rdq',
            by='permno',
            direction='backward'
        )
        merged = merged.sort_values(by='date')
        merged = merged[['permno', 'ticker', 'date', 'ret', 'prc','vol', 'mktcap', 
                         'gvkey', 'rdq', 'saleq']]
        merged = merged.ffill()
        unique_dates = merged['date'].unique()
        date_mapping = {date: i for i, date in enumerate(sorted(unique_dates))}
        merged['date_mapped'] = merged['date'].map(date_mapping)
        merged['year'] = pd.to_datetime(merged['date']).dt.year
        final_data = merged[merged['year'] >= final_year]
        logger.info(f"Data loaded: merged shape {merged.shape}, final_data shape {final_data.shape}")
        return merged, final_data

    def _create_sequences(self):
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
        data = self.final_data
        lookback = self.lookback
        T = self.T
        unique_dates = pd.to_datetime(data['date'].unique())
        unique_dates_sorted = np.sort(unique_dates)
        num_features = 6  # Here we use: 'permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq'
        
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
                        features = hist_data[['permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq']].values
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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.future_returns[idx], self.masks[idx]