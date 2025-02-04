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

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db = wrds.Connection()

class AlphaPortfolioData(Dataset):
    
    def __init__(self, start_year=2014, end_year=2020, final_year=2016, lookback=12, G=2):
        super().__init__()
        self.lookback = lookback
        self.G = G  # Number of assets to long/short
        self.merged, self.final_data = self._load_wrds_data(start_year, end_year, final_year)
        # self.sequences, self.future_returns, self.masks = self._create_sequences()
        # self._validate_data_shapes()

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
        data = self.data
        lookback = self.lookback
        unique_dates = pd.to_datetime(data['date'].unique())
        unique_assets = data['permno'].unique()
        global_max_assets  = data.groupby('year')['permno'].nunique().iloc[-1]
        
        sequences = []
        future_returns = []
        masks = []
        min_assets = 2 * self.G
        batch_info = []

        # First pass: collect valid batches
        for date_idx in range(len(unique_dates) - 2 * lookback):
            hist_start = unique_dates[date_idx]
            hist_end = unique_dates[date_idx + lookback - 1]
            future_start = unique_dates[date_idx + lookback]
            future_end = unique_dates[date_idx + 2 * lookback - 1]

            batch_assets = []
            hist_features = []
            fwd_returns = []
            
            for asset in unique_assets:
                asset_hist = data[
                    (data['permno'] == asset) & 
                    (data['date'].between(hist_start, hist_end))
                ].sort_values('date')
                
                asset_future = data[
                    (data['permno'] == asset) & 
                    (data['date'].between(future_start, future_end))
                ]['ret'].values
                
                if len(asset_hist) == lookback and len(asset_future) == lookback:
                    features = asset_hist[['ret', 'prc', 'vol', 'mktcap', 'saleq']].values
                    hist_features.append(features)
                    fwd_returns.append(asset_future)
                    batch_assets.append(asset)

            if len(hist_features) >= min_assets:
                batch_info.append({
                    'features': np.stack(hist_features),
                    'returns': np.stack(fwd_returns),
                    'num_assets': len(hist_features)
                })

        # Find global max assets across all valid batches
        if not batch_info:
            return torch.empty(0), torch.empty(0), torch.empty(0)
        
        global_max_assets = max(b['num_assets'] for b in batch_info)
        features_dim = batch_info[0]['features'].shape[-1]

        # Second pass: pad to global max
        for batch in batch_info:
            num_assets = batch['num_assets']
            
            # Features: (assets, lookback, features)
            padded_features = np.zeros((global_max_assets, lookback, features_dim))
            padded_features[:num_assets] = batch['features']
            
            # Returns: (assets, lookback)
            padded_returns = np.zeros((global_max_assets, lookback))  # Fix 1: 2D padding
            padded_returns[:num_assets] = batch['returns']
            
            # Mask: (assets,)
            mask = np.zeros(global_max_assets, dtype=bool)
            mask[:num_assets] = True

            sequences.append(padded_features)
            future_returns.append(padded_returns)
            masks.append(mask)

        return (
            torch.as_tensor(np.array(sequences), dtype=torch.float32),  # (time, assets, lookback, features)
            torch.as_tensor(np.array(future_returns), dtype=torch.float32),  # (time, assets, lookback)
            torch.as_tensor(np.array(masks), dtype=torch.bool)  # (time, assets)
        )