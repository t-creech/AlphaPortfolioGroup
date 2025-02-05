import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import wrds
import math
import gym
from gym import spaces
from torch.optim.optimizer import Optimizer

# To this:
import torch.optim as optim

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db = wrds.Connection()

class AlphaPortfolioData(Dataset):
    def __init__(self, start_year=2014, end_year=2020, final_year=2016, lookback=12, G=2):
        super().__init__()
        self.lookback = lookback
        self.G = G
        self.merged, self.final_data = self._load_wrds_data(start_year, end_year, final_year)
        self.unique_permnos = sorted(self.final_data['permno'].unique())
        self.global_max_assets = len(self.unique_permnos)
        self.permno_to_idx = {permno: idx for idx, permno in enumerate(self.unique_permnos)}
        self.sequences, self.future_returns, self.masks = self._create_sequences()

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

            crsp_data['mktcap'] = (crsp_data['prc'].abs() * crsp_data['shrout'] * 1000) / 1e6  # In millions
            crsp_data['year'] = pd.to_datetime(crsp_data['date']).dt.year
            crsp_data = crsp_data.dropna(subset=['mktcap'])
            
            top_50_permnos_by_year = crsp_data.groupby('permno')['mktcap'].agg(['max']).reset_index().sort_values(by='max', ascending=False).head(50)['permno'].unique()
            permno_list.extend(top_50_permnos_by_year)
            
            combined_data = pd.concat([combined_data, crsp_data[crsp_data['permno'].isin(permno_list)]], axis=0)

        combined_data = combined_data[['permno', 'ticker', 'date', 'ret', 'prc', 'shrout', 'vol', 'mktcap', 'year']]
        combined_data['date'] = pd.to_datetime(combined_data['date'])

        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-31'

        # Query Compustat quarterly data with release dates (rdq)
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

        # Link Compustat GVKEY to CRSP PERMNO
        link_query = """
            SELECT lpermno AS permno, gvkey, linkdt, linkenddt
            FROM crsp.ccmxpf_linktable
            WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
        """
        link = db.raw_sql(link_query)
        fund = pd.merge(fund, link, on='gvkey', how='left')
        fund = fund.dropna(subset=['permno'])

        # Sort both datasets by date
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
        # merged = merged.dropna(subset=['rdq', 'ticker'])
        merged = merged.sort_values(by='date')
        merged = merged[['permno', 'ticker', 'date', 'ret', 'prc','vol', 'mktcap', 'gvkey', 'rdq', 'saleq']]
        merged = merged.ffill()

        unique_dates = merged['date'].unique()
        date_mapping = {date: i for i, date in enumerate(sorted(unique_dates))}
        merged['date_mapped'] = merged['date'].map(date_mapping)

        merged['year'] = pd.to_datetime(merged['date']).dt.year
        final_data = merged[merged['year'] >= final_year]

        
        return merged, final_data


    def _create_sequences(self):
        data = self.final_data
        lookback = self.lookback
        unique_dates = pd.to_datetime(data['date'].unique())
        unique_dates_sorted = np.sort(unique_dates)
        num_features = 6  # Based on []'permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq']

        sequences = []
        future_returns = []
        masks = []

        for start_idx in tqdm(range(len(unique_dates_sorted) - 2 * lookback+1)):
            hist_start = unique_dates_sorted[start_idx]
            hist_end = unique_dates_sorted[start_idx + lookback - 1]
            future_start = unique_dates_sorted[start_idx + lookback]
            future_end = unique_dates_sorted[start_idx + 2 * lookback-1]

            print(f'Hist start: {hist_start}, Hist end: {hist_end}, Future start: {future_start}, Future end: {future_end}')

            # Initialize batch arrays with zeros
            batch_features = np.zeros((self.global_max_assets, lookback, num_features))
            batch_returns = np.zeros((self.global_max_assets, lookback))
            batch_mask = np.zeros(self.global_max_assets, dtype=bool)

            for permno in self.unique_permnos:
                idx = self.permno_to_idx[permno]

                # Historical data for the current window
                hist_data = data[
                    (data['permno'] == permno) &
                    (data['date'] >= hist_start) &
                    (data['date'] <= hist_end)
                ].sort_values('date')

                # Future returns for the next window
                future_data = data[
                    (data['permno'] == permno) &
                    (data['date'] >= future_start) &
                    (data['date'] <= future_end)
                ]['ret'].values

                # Check if both periods have complete data
                if len(hist_data) == lookback and len(future_data) == lookback:
                    features = hist_data[['permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq']].values
                    batch_features[idx] = features
                    batch_returns[idx] = future_data
                    batch_mask[idx] = True

            sequences.append(batch_features)
            future_returns.append(batch_returns)
            masks.append(batch_mask)

        # Convert to tensors
        sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        future_returns_tensor = torch.tensor(np.array(future_returns), dtype=torch.float32)
        masks_tensor = torch.tensor(np.array(masks), dtype=torch.bool)

        return sequences_tensor, future_returns_tensor, masks_tensor

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.future_returns[idx], self.masks[idx]