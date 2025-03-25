# data_loader.py
from datetime import datetime
import functools
import time
import traceback
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import os

class FeatureScaler:
    """Feature scaler for data normalization."""
    
    def __init__(self, cycle_id: Optional[int] = None, experiment_id: Optional[str] = None):
        """
        Initialize feature scaler.
        
        Args:
            cycle_id: Optional cycle identifier
            experiment_id: Optional experiment identifier
        """
        self.means = None
        self.stds = None
        self.cycle_id = cycle_id
        self.experiment_id = experiment_id
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> str:
        """
        Fit scaler to data.
        
        Args:
            data: Data to fit
            
        Returns:
            Path to saved scaler
        """
        logging.info(f"Fitting scaler to data with shape {data.shape}")
        
        # Calculate statistics
        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)
        self.stds[self.stds < 1e-8] = 1.0  # Avoid division by zero
        
        self.is_fitted = True
        
        # Save scaler
        return self.save()
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted statistics.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            logging.error("Scaler must be fitted before transform.")
            return None
        
        # Special case handling for the specific shape mismatch we're seeing
        if len(data.shape) == 5 and len(self.means.shape) == 4:
            # This is the case where data has shape like (batch, T, assets, lookback, features) 
            # and means has shape like (T, assets, lookback, features)
            transformed = np.zeros_like(data)
            
            # Handle the common assets
            common_assets = min(data.shape[2], self.means.shape[1])
            
            # For each batch element
            for b in range(data.shape[0]):
                # Transform the common assets
                transformed[b, :, :common_assets, :, :] = (data[b, :, :common_assets, :, :] - self.means[:, :common_assets, :, :]) / self.stds[:, :common_assets, :, :]
                
                # For extra assets, use the average statistics
                if data.shape[2] > self.means.shape[1]:
                    avg_means = np.mean(self.means, axis=1, keepdims=True)  # [T, 1, lookback, features]
                    avg_stds = np.mean(self.stds, axis=1, keepdims=True)
                    avg_stds[avg_stds < 1e-8] = 1.0  # Avoid division by zero
                    
                    transformed[b, :, common_assets:, :, :] = (data[b, :, common_assets:, :, :] - avg_means) / avg_stds
            
            return transformed
        
        # Standard case
        try:
            transformed = (data - self.means) / self.stds
        except ValueError as e:
            logging.error(f"Broadcasting error in transform: {e}")
            logging.error(f"Data shape: {data.shape}, Means shape: {self.means.shape}")
            raise
        
        # Handle infinities and NaNs
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        
        return transformed
    
    def save(self) -> str:
        """
        Save scaler to file.
        
        Returns:
            Path to saved scaler
        """
        os.makedirs("scalers", exist_ok=True)
        
        filename = f"scalers/scaler"
        if self.experiment_id:
            filename += f"_{self.experiment_id}"
        if self.cycle_id is not None:
            filename += f"_cycle_{self.cycle_id}"
        filename += ".npz"
        
        np.savez(
            filename,
            means=self.means,
            stds=self.stds
        )
        
        logging.info(f"Scaler saved to {filename}")
        
        return filename
    
    def save_final(self) -> str:
        """
        Save final scaler after full training.
        
        Returns:
            Path to saved final scaler
        """
        os.makedirs("scalers", exist_ok=True)
        
        filename = f"scalers/scaler_final"
        if self.experiment_id:
            filename += f"_{self.experiment_id}"
        filename += ".npz"
        
        np.savez(
            filename,
            means=self.means,
            stds=self.stds
        )
        
        logging.info(f"Final scaler saved to {filename}")
        
        return filename
    
    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        """
        Load scaler from file.
        
        Args:
            path: Path to saved scaler
            
        Returns:
            Loaded scaler
        """
        try:
            data = np.load(path)
            
            scaler = cls()
            scaler.means = data['means']
            scaler.stds = data['stds']
            scaler.is_fitted = True
            
            logging.info(f"Loaded scaler from {path}")
            
            return scaler
        except Exception as e:
            logging.error(f"Failed to load scaler from {path}: {str(e)}")
            return None

def profile_function(func):
    """Decorator to profile a function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"Function {func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

class AlphaPortfolioData(Dataset):
    def __init__(self, data_path: str, start_date: str, end_date: str, T: int, lookback: int, 
                 cycle_id=None, experiment_id=None, scaler_path=None, cache_dir="cached_sequences"):
        """
        Initialize AlphaPortfolioData using pre-processed data file.
        
        Args:
            data_path: Path to the cleaned data file
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            lookback: Number of lookback periods
            cycle_id: Training cycle identifier
            experiment_id: Experiment identifier
            scaler_path: Path to a saved scaler (for validation data)
            cache_dir: Directory to cache processed sequences
        """
        self.lookback = lookback
        self.T = T
        self.start_date = start_date
        self.end_date = end_date
        self.cycle_id = cycle_id
        self.experiment_id = experiment_id
        self.data_path = data_path
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate sequence cache filename
        self.sequence_cache_file = self._get_cache_filename()
        logging.info(f"Cache file path: {self.sequence_cache_file}")
        
        # Load initial data for metadata (asset names, etc.)
        self.data = self._load_data(data_path, start_date, end_date)
        
        # Get unique assets
        self.unique_permnos = sorted(self.data['permno'].unique())
        self.global_max_assets = len(self.unique_permnos)
        self.permno_to_idx = {permno: idx for idx, permno in enumerate(self.unique_permnos)}
        
        # Check if cached sequences exist
        if os.path.exists(self.sequence_cache_file):
            logging.info(f"Found existing cache file: {self.sequence_cache_file}")
            cache_loaded = self._load_cached_sequences()
            
            if not cache_loaded:
                logging.warning("Failed to load from cache - creating sequences from scratch")
                # Create sequences from scratch
                self._create_sequences()
                # Save sequences to cache
                self._save_sequences_to_cache()
        else:
            logging.info(f"No cache file found at {self.sequence_cache_file} - creating sequences from scratch")
            # Create sequences
            self._create_sequences()
            # Save sequences to cache
            self._save_sequences_to_cache()
        
        # Initialize scaler
        if scaler_path:
            # For validation/test data, load existing scaler
            logging.info(f"Loading scaler from {scaler_path} for validation/test data")
            self.scaler = FeatureScaler.load(scaler_path)
            if self.scaler is None:
                raise ValueError(f"Failed to load scaler from {scaler_path}")
                
            # Transform validation/test data using training scaler
            transformed_data = self.scaler.transform(self.sequences.numpy())
            if transformed_data is None:
                raise ValueError("Failed to transform validation/test data")
                
            self.sequences = torch.tensor(transformed_data, dtype=torch.float32)
            logging.info(f"Validation/test data transformed using training scaler")
        else:
            # For training data, create and fit new scaler
            logging.info("Creating and fitting new scaler for training data")
            self.scaler = FeatureScaler(cycle_id=cycle_id, experiment_id=experiment_id)
            self.scaler_path = self.scaler.fit(self.sequences.numpy())
            
            if self.scaler_path is None:
                raise ValueError("Failed to fit and save scaler")
                
            transformed_data = self.scaler.transform(self.sequences.numpy())
            if transformed_data is None:
                raise ValueError("Failed to transform training data")
                
            self.sequences = torch.tensor(transformed_data, dtype=torch.float32)
            logging.info(f"Training data transformed and scaler saved to {self.scaler_path}")
            
        # Verify data integrity after transformation
        if torch.isnan(self.sequences).any():
            raise ValueError("NaN values detected in transformed sequences")
    
    def _get_cache_filename(self) -> str:
        """
        Generate a unique filename for cached sequences based on dataset parameters.
        
        Returns:
            Path to cache file
        """
        # Extract years from dates
        try:
            start_year = self.start_date.split('-')[0]
            end_year = self.end_date.split('-')[0]
        except:
            # Fallback if dates are not in expected format
            start_year = "unknown_start"
            end_year = "unknown_end"
        
        # # Create a hash of the data path to avoid filename issues
        # import hashlib
        # data_hash = hashlib.md5(self.data_path.encode()).hexdigest()[:8]
        
        # Create filename with all relevant parameters
        filename = f"seq_{start_year}_{end_year}_T{self.T}_lb{self.lookback}"
        
        # Add assets count if available
        if hasattr(self, 'global_max_assets'):
            filename += f"_assets{self.global_max_assets}"
        
        # if self.cycle_id is not None:
        #     filename += f"_cycle{self.cycle_id}"
        
        # if self.experiment_id:
        #     filename += f"_{self.experiment_id}"
        
        # filename += f"_{data_hash}.pt"
        
        return os.path.join(self.cache_dir, filename)
    
    def _save_sequences_to_cache(self):
        """Save processed sequences to cache file efficiently."""
        logging.info(f"Saving sequences to cache: {self.sequence_cache_file}")
        
        try:
            # Prepare cache data
            cache_data = {
                'sequences': self.sequences,
                'future_returns': self.future_returns,
                'masks': self.masks,
                'unique_permnos': self.unique_permnos,
                'global_max_assets': self.global_max_assets,
                'metadata': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'T': self.T,
                    'lookback': self.lookback,
                    'sequence_shape': self.sequences.shape,
                    'future_returns_shape': self.future_returns.shape,
                    'masks_shape': self.masks.shape,
                    'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data_path': self.data_path
                }
            }
            
            # Save to cache file
            torch.save(cache_data, self.sequence_cache_file)
            
            # Log success
            logging.info(f"Successfully saved sequences to cache: {self.sequence_cache_file}")
            logging.info(f"Sequence shapes - Sequences: {self.sequences.shape}, "
                       f"Future returns: {self.future_returns.shape}, "
                       f"Masks: {self.masks.shape}")
            
            # Verify file was created
            if not os.path.exists(self.sequence_cache_file):
                logging.error(f"Cache file was not created successfully: {self.sequence_cache_file}")
                return False
                
            # Verify file size
            file_size_mb = os.path.getsize(self.sequence_cache_file) / (1024 * 1024)
            logging.info(f"Cache file size: {file_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving sequences to cache: {str(e)}")
            logging.error(traceback.format_exc())
            return False
    
    def _load_cached_sequences(self):
        """Load sequences from cache file efficiently."""
        logging.info(f"Attempting to load cached sequences from {self.sequence_cache_file}")
        
        try:
            # Check if file exists and has content
            if not os.path.exists(self.sequence_cache_file):
                logging.error(f"Cache file does not exist: {self.sequence_cache_file}")
                return False
                
            file_size = os.path.getsize(self.sequence_cache_file)
            if file_size < 1000:  # Less than 1KB is probably corrupted
                logging.error(f"Cache file appears to be corrupted (size: {file_size} bytes)")
                return False
            
            # Load cache data
            cache_data = torch.load(self.sequence_cache_file)
            
            # Validate cache data structure
            required_keys = ['sequences', 'future_returns', 'masks', 'unique_permnos', 'global_max_assets']
            for key in required_keys:
                if key not in cache_data:
                    logging.error(f"Cache file is missing required key: {key}")
                    return False
            
            # Check if tensor dimensions match expectations
            if len(cache_data['sequences'].shape) != 5:
                logging.error(f"Sequences tensor has unexpected shape: {cache_data['sequences'].shape}")
                return False
                
            if len(cache_data['future_returns'].shape) != 3:
                logging.error(f"Future returns tensor has unexpected shape: {cache_data['future_returns'].shape}")
                return False
                
            if len(cache_data['masks'].shape) != 3:
                logging.error(f"Masks tensor has unexpected shape: {cache_data['masks'].shape}")
                return False
            
            # Load tensors
            self.sequences = cache_data['sequences']
            self.future_returns = cache_data['future_returns']
            self.masks = cache_data['masks']
            
            # Load asset information
            self.unique_permnos = cache_data['unique_permnos']
            self.global_max_assets = cache_data['global_max_assets']
            self.permno_to_idx = {permno: idx for idx, permno in enumerate(self.unique_permnos)}
            
            # Log metadata
            metadata = cache_data.get('metadata', {})
            logging.info(f"Successfully loaded sequences from cache: {self.sequence_cache_file}")
            logging.info(f"Cached sequence info: {metadata}")
            logging.info(f"Loaded shapes - Sequences: {self.sequences.shape}, "
                       f"Future returns: {self.future_returns.shape}, "
                       f"Masks: {self.masks.shape}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading cached sequences: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def _create_sequences(self):
        """Create sequences from raw data - optimized version."""
        logging.info("Creating sequences from raw data (this may take some time)...")
        
        data = self.data
        num_features = self.data.drop(columns=['permno', 'date', 'rdq']).shape[-1]
        unique_dates = np.sort(pd.to_datetime(data['date'].unique()))
        num_episodes = max(1, len(unique_dates) - (2 * self.lookback) + 1)
        
        logging.info(f"Initial setup:")
        logging.info(f"  Total unique dates: {len(unique_dates)}")
        logging.info(f"  Number of features: {num_features}")
        logging.info(f"  Number of episodes: {num_episodes}")
        logging.info(f"  Lookback period: {self.lookback}")
        logging.info(f"  Time steps per episode: {self.T}")
        logging.info(f"  Total unique assets: {len(self.unique_permnos)}")
        
        # Pre-allocate arrays for efficiency
        episodes_states = []
        episodes_fwd = []
        episodes_masks = []
        
        # Group data by permno for faster access
        permno_grouped_data = {}
        for permno in tqdm(self.unique_permnos, desc="Grouping data by asset"):
            permno_grouped_data[permno] = data[data['permno'] == permno]
        
        # Create date lookup dictionary for faster filtering
        date_indices = {date: i for i, date in enumerate(unique_dates)}
        
        for start_idx in tqdm(range(num_episodes), desc="Creating episodes"):
            episode_states = []
            episode_fwd = []
            episode_masks = []
            
            for t in range(self.T):
                state_start = start_idx + t
                state_end = state_start + self.lookback
                fwd_index = state_end
                
                if fwd_index >= len(unique_dates):
                    break
                    
                step_states = np.zeros((self.global_max_assets, self.lookback, num_features))
                step_fwd = np.zeros(self.global_max_assets)
                step_mask = np.zeros(self.global_max_assets, dtype=bool)
                
                # Get date ranges for this step
                hist_dates = unique_dates[state_start:state_end]
                future_date = unique_dates[fwd_index]
                
                # Process all assets efficiently
                for permno, permno_data in permno_grouped_data.items():
                    idx = self.permno_to_idx[permno]
                    
                    # Vectorized filter for historical data
                    hist_mask = permno_data['date'].isin(hist_dates)
                    hist_data = permno_data[hist_mask].sort_values('date')
                    
                    # Vectorized filter for future data
                    fwd_data = permno_data[permno_data['date'] == future_date]
                    
                    if len(hist_data) == self.lookback and len(fwd_data) == 1:
                        features = hist_data.drop(columns=['permno', 'date', 'rdq']).values
                        step_states[idx] = features
                        step_fwd[idx] = fwd_data['ret'].values[0]
                        step_mask[idx] = True
                
                episode_states.append(step_states)
                episode_fwd.append(step_fwd)
                episode_masks.append(step_mask)
            
            if len(episode_states) == self.T:
                episodes_states.append(np.stack(episode_states))
                episodes_fwd.append(np.stack(episode_fwd))
                episodes_masks.append(np.stack(episode_masks))
        
        # Convert to tensors in one operation
        if episodes_states:
            self.sequences = torch.tensor(np.array(episodes_states), dtype=torch.float32)
            self.future_returns = torch.tensor(np.array(episodes_fwd), dtype=torch.float32)
            self.masks = torch.tensor(np.array(episodes_masks), dtype=torch.float32)
        else:
            # Create empty tensors with correct shapes if no episodes were created
            self.sequences = torch.zeros((0, self.T, self.global_max_assets, self.lookback, num_features), dtype=torch.float32)
            self.future_returns = torch.zeros((0, self.T, self.global_max_assets), dtype=torch.float32)
            self.masks = torch.zeros((0, self.T, self.global_max_assets), dtype=torch.float32)
        
        logging.info(f"Finished creating sequences - Shapes:")
        logging.info(f"  Sequences: {self.sequences.shape}")
        logging.info(f"  Future returns: {self.future_returns.shape}")
        logging.info(f"  Masks: {self.masks.shape}")
    
    def get_cache_stats(self):
        """
        Get statistics about cached sequences.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_stats = {
            'cache_file': self.sequence_cache_file,
            'file_exists': os.path.exists(self.sequence_cache_file),
            'file_size_mb': 0
        }
        
        if cache_stats['file_exists']:
            cache_stats['file_size_mb'] = os.path.getsize(self.sequence_cache_file) / (1024 * 1024)
            
            try:
                # Load metadata only
                cache_data = torch.load(self.sequence_cache_file, map_location='cpu')
                metadata = cache_data.get('metadata', {})
                cache_stats['metadata'] = metadata
                
                # Add shapes information
                if 'sequences' in cache_data:
                    cache_stats['sequences_shape'] = tuple(cache_data['sequences'].shape)
                if 'future_returns' in cache_data:
                    cache_stats['future_returns_shape'] = tuple(cache_data['future_returns'].shape)
                if 'masks' in cache_data:
                    cache_stats['masks_shape'] = tuple(cache_data['masks'].shape)
                
            except Exception as e:
                cache_stats['metadata_error'] = str(e)
        
        return cache_stats

    def _load_data(self, data_path: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load data from pre-processed file and filter by date range.
        
        Args:
            data_path: Path to the cleaned data file
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            
        Returns:
            Filtered DataFrame
        """
        # Load data from file
        try:
            data = pd.read_csv(data_path)
            logging.info(f"Loaded data with shape {data.shape}")
        except Exception as e:
            logging.error(f"Failed to load data from {data_path}: {str(e)}")
            raise
        
        # Ensure date column is datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Filter by date range
        filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        logging.info(f"Filtered data to range {start_date} to {end_date}: {filtered_data.shape}")
        
        return filtered_data

    def save_final_scaler(self):
        """Save the final scaler after full training."""
        if hasattr(self, 'scaler'):
            return self.scaler.save_final()
        return None

    def get_scaler_path(self):
        """Get the path to the current scaler."""
        return getattr(self, 'scaler_path', None)
        
    def get_feature_count(self):
        """Return the number of features (excluding permno and date)."""
        return len(self.data.columns) - 3  # Excluding permno, date, and rdq

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.future_returns[idx], self.masks[idx]

    def get_asset_names(self):
        """Get asset names (PERMNOs) for visualization."""
        return [str(p) for p in self.unique_permnos]

    def clean_cache(self, older_than_days=20):
        """
        Clean old cache files.
        
        Args:
            older_than_days: Optional, delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        import glob
        
        cache_files = glob.glob(os.path.join(self.cache_dir, "seq_*.pt"))
        deleted_count = 0
        
        for file in cache_files:
            if older_than_days is not None:
                # Check file age
                file_time = os.path.getmtime(file)
                file_age_days = (time.time() - file_time) / (60 * 60 * 24)
                
                if file_age_days > older_than_days:
                    os.remove(file)
                    deleted_count += 1
                    logging.info(f"Deleted old cache file: {file} (age: {file_age_days:.1f} days)")
            elif file != self.sequence_cache_file:
                # Delete all files except the current one
                os.remove(file)
                deleted_count += 1
                logging.info(f"Deleted cache file: {file}")
        
        return deleted_count
    
def create_data_loaders(config, cycle_params, data_path):
    """
    Create data loaders for training and validation.
    
    Args:
        config: Configuration object
        cycle_params: Parameters for current cycle
        data_path: Path to data file
        
    Returns:
        Dictionary with train and val data loaders
    """
    # Extract parameters
    T = config.config["model"]["T"]
    lookback = config.config["model"]["lookback"]
    experiment_id = config.config["experiment_id"]
    batch_size = config.config["training"]["batch_size"]
    num_workers = config.config["training"]["num_workers"]
    
    # Create training dataset
    train_data = AlphaPortfolioData(
        data_path=data_path,
        start_date=cycle_params["train_start"],
        end_date=cycle_params["train_end"],
        T = T,
        lookback=lookback,
        cycle_id=cycle_params["cycle_idx"],
        experiment_id=experiment_id
    )
    
    # Create validation dataset
    val_data = AlphaPortfolioData(
        data_path=data_path,
        start_date=cycle_params["validate_start"],
        end_date=cycle_params["validate_end"],
        T = T,
        lookback=lookback,
        cycle_id=cycle_params["cycle_idx"],
        experiment_id=experiment_id,
        scaler_path=train_data.get_scaler_path()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_data": train_data,
        "val_data": val_data
    }
