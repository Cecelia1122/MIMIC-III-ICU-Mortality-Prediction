"""
Preprocessing module for MIMIC-III data.
Feature engineering and data preparation for model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import pickle


class MIMICPreprocessor:
    """
    Preprocess MIMIC-III data for model training.
    
    Features:
        - Demographics:  age, gender, ethnicity
        - Vital signs: aggregated statistics (mean, std, min, max, first, last)
        - Lab tests: aggregated statistics
        - Time series: resampled to fixed intervals
    """
    
    VITAL_NAMES = [
        'heart_rate', 'sbp', 'dbp', 'mean_bp', 
        'resp_rate', 'temperature', 'spo2', 'gcs'
    ]
    
    LAB_NAMES = [
        'glucose', 'potassium', 'sodium', 'chloride', 'bicarbonate',
        'bun', 'creatinine', 'hemoglobin', 'hematocrit', 'wbc',
        'platelets', 'lactate'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.fitted = False
    
    def _aggregate_time_series(
        self,
        data: pd.DataFrame,
        value_col: str,
        name_col: str,
        names: List[str]
    ) -> Dict[str, float]:
        """Aggregate time series data into statistics."""
        features = {}
        
        # Handle empty DataFrame
        if data is None or len(data) == 0:
            for name in names:
                features[f'{name}_mean'] = np.nan
                features[f'{name}_std'] = np.nan
                features[f'{name}_min'] = np.nan
                features[f'{name}_max'] = np.nan
                features[f'{name}_first'] = np.nan
                features[f'{name}_last'] = np.nan
            return features
        
        # Check if name_col exists in data
        if name_col not in data.columns:
            # Try to find the correct column name
            possible_cols = ['NAME', 'VITAL_NAME', 'LAB_NAME', 'name', 'vital_name', 'lab_name']
            for col in possible_cols:
                if col in data.columns:
                    name_col = col
                    break
            else:
                # If no matching column found, return NaN features
                for name in names:
                    features[f'{name}_mean'] = np.nan
                    features[f'{name}_std'] = np.nan
                    features[f'{name}_min'] = np. nan
                    features[f'{name}_max'] = np.nan
                    features[f'{name}_first'] = np.nan
                    features[f'{name}_last'] = np.nan
                return features
        
        for name in names:
            subset = data[data[name_col] == name][value_col]
            
            if len(subset) == 0:
                features[f'{name}_mean'] = np.nan
                features[f'{name}_std'] = np. nan
                features[f'{name}_min'] = np.nan
                features[f'{name}_max'] = np.nan
                features[f'{name}_first'] = np.nan
                features[f'{name}_last'] = np. nan
            else:
                features[f'{name}_mean'] = subset.mean()
                features[f'{name}_std'] = subset.std() if len(subset) > 1 else 0
                features[f'{name}_min'] = subset.min()
                features[f'{name}_max'] = subset.max()
                features[f'{name}_first'] = subset.iloc[0]
                features[f'{name}_last'] = subset. iloc[-1]
        
        return features
    
    def _create_time_series(
        self,
        data:  pd.DataFrame,
        value_col: str,
        name_col: str,
        names:  List[str],
        hours: int = 48,
        interval: int = 4
    ) -> np.ndarray:
        """
        Create fixed-length time series by resampling.
        
        Args:
            data: DataFrame with HOURS, name_col, value_col
            value_col: Column with values
            name_col: Column with variable names
            names: List of variable names
            hours: Total hours
            interval: Resampling interval in hours
            
        Returns: 
            Array of shape (num_timesteps, num_features)
        """
        num_timesteps = hours // interval
        num_features = len(names)
        
        ts = np.full((num_timesteps, num_features), np.nan)
        
        # Handle empty DataFrame
        if data is None or len(data) == 0:
            return ts
        
        # Check if name_col exists in data
        if name_col not in data.columns:
            possible_cols = ['NAME', 'VITAL_NAME', 'LAB_NAME', 'name', 'vital_name', 'lab_name']
            for col in possible_cols:
                if col in data.columns:
                    name_col = col
                    break
            else:
                return ts
        
        for j, name in enumerate(names):
            subset = data[data[name_col] == name][['HOURS', value_col]].copy()
            
            if len(subset) == 0:
                continue
            
            # Bin into intervals
            subset['BIN'] = (subset['HOURS'] // interval).astype(int)
            subset = subset[subset['BIN'] < num_timesteps]
            
            # Aggregate within each bin
            binned = subset.groupby('BIN')[value_col].mean()
            
            for bin_idx, value in binned.items():
                if 0 <= bin_idx < num_timesteps:
                    ts[bin_idx, j] = value
        
        return ts
    
    def _extract_demographics(self, cohort_row: pd.Series) -> Dict[str, float]:
        """Extract demographic features from cohort row."""
        features = {}
        
        # Age (normalized)
        features['age'] = cohort_row. get('AGE', np.nan)
        
        # Gender (binary)
        gender = cohort_row.get('GENDER', '')
        features['gender_male'] = 1 if gender == 'M' else 0
        
        # Ethnicity (one-hot simplified)
        ethnicity = str(cohort_row.get('ETHNICITY', '')).upper()
        features['ethnicity_white'] = 1 if 'WHITE' in ethnicity else 0
        features['ethnicity_black'] = 1 if 'BLACK' in ethnicity else 0
        features['ethnicity_asian'] = 1 if 'ASIAN' in ethnicity else 0
        features['ethnicity_hispanic'] = 1 if 'HISPANIC' in ethnicity else 0
        
        # Admission type
        admission_type = str(cohort_row.get('ADMISSION_TYPE', '')).upper()
        features['admission_emergency'] = 1 if 'EMERGENCY' in admission_type else 0
        features['admission_elective'] = 1 if 'ELECTIVE' in admission_type else 0
        
        return features
    
    def preprocess(
        self,
        cohort: pd.DataFrame,
        vitals_dict: Dict[int, pd. DataFrame],
        labs_dict:  Dict[int, pd.DataFrame],
        diagnoses_dict: Dict[int, List[str]],
        hours: int = 48,
        interval: int = 4,
        create_time_series: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np. ndarray]:
        """
        Preprocess all data into model-ready format.
        
        Args:
            cohort:  Cohort DataFrame
            vitals_dict: Dictionary of vital signs
            labs_dict:  Dictionary of lab tests
            diagnoses_dict: Dictionary of diagnoses
            hours: Hours of data
            interval: Time series interval
            create_time_series: Whether to create time series data
            
        Returns:
            Tuple of (static_features, time_series, labels, icustay_ids)
        """
        print("\nPreprocessing data...")
        
        static_features_list = []
        time_series_list = []
        labels = []
        icustay_ids = []
        
        skipped_count = 0
        
        for _, row in tqdm(cohort. iterrows(), total=len(cohort), desc="Processing patients"):
            icustay_id = row['ICUSTAY_ID']
            
            # Skip if missing vitals (but continue with empty data)
            vitals_df = vitals_dict.get(icustay_id, pd.DataFrame())
            labs_df = labs_dict.get(icustay_id, pd.DataFrame())
            
            # Skip only if BOTH vitals and labs are missing
            if len(vitals_df) == 0 and len(labs_df) == 0:
                skipped_count += 1
                continue
            
            # Extract demographics
            demo_features = self._extract_demographics(row)
            
            # Extract vital sign statistics
            # Use 'NAME' as the column name (matching data_extraction.py)
            vital_features = self._aggregate_time_series(
                vitals_df, 'VALUENUM', 'NAME', self. VITAL_NAMES
            )
            
            # Extract lab statistics
            lab_features = self._aggregate_time_series(
                labs_df, 'VALUENUM', 'NAME', self. LAB_NAMES
            )
            
            # Combine static features
            static_features = {**demo_features, **vital_features, **lab_features}
            static_features_list.append(static_features)
            
            # Create time series
            if create_time_series: 
                vital_ts = self._create_time_series(
                    vitals_df, 'VALUENUM', 'NAME', self. VITAL_NAMES,
                    hours=hours, interval=interval
                )
                lab_ts = self._create_time_series(
                    labs_df, 'VALUENUM', 'NAME', self.LAB_NAMES,
                    hours=hours, interval=interval
                )
                ts = np.concatenate([vital_ts, lab_ts], axis=1)
                time_series_list.append(ts)
            
            # Label
            labels.append(row['MORTALITY'])
            icustay_ids.append(icustay_id)
        
        # Convert to arrays
        static_df = pd.DataFrame(static_features_list)
        self.feature_names = list(static_df.columns)
        
        static_array = static_df.values. astype(np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        icustay_ids_array = np.array(icustay_ids)
        
        if create_time_series:
            time_series_array = np. stack(time_series_list).astype(np.float32)
        else:
            time_series_array = np.array([])
        
        print(f"\n  Processed patients: {len(labels_array)}")
        print(f"  Skipped patients (no vitals/labs): {skipped_count}")
        print(f"  Static features:  {static_array.shape}")
        if create_time_series:
            print(f"  Time series shape: {time_series_array. shape}")
        print(f"  Mortality rate: {labels_array.mean():.2%}")
        
        return static_array, time_series_array, labels_array, icustay_ids_array
    
    def impute_and_normalize(
        self,
        static_features: np.ndarray,
        time_series: np.ndarray,
        fit:  bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Impute missing values and normalize features.
        
        Args:
            static_features: Static feature array
            time_series: Time series array
            fit: Whether to fit scaler (True for training data)
            
        Returns: 
            Tuple of (normalized_static, normalized_time_series)
        """
        # Impute static features with median
        static_imputed = static_features.copy()
        for i in range(static_imputed.shape[1]):
            col = static_imputed[:, i]
            mask = np.isnan(col)
            if mask.any():
                median_val = np.nanmedian(col)
                static_imputed[mask, i] = median_val if not np.isnan(median_val) else 0
        
        # Normalize static features
        if fit:
            static_normalized = self.scaler.fit_transform(static_imputed)
            self.fitted = True
        else:
            static_normalized = self.scaler.transform(static_imputed)
        
        # Impute and normalize time series
        if len(time_series) > 0:
            ts_imputed = time_series.copy()
            
            # Forward fill, then backward fill, then zero
            for i in range(ts_imputed.shape[0]):
                for j in range(ts_imputed.shape[2]):
                    col = ts_imputed[i, : , j]
                    
                    # Forward fill
                    mask = np.isnan(col)
                    if mask.all():
                        # All NaN, fill with 0
                        col[: ] = 0
                    elif mask.any():
                        idx = np.where(~mask, np.arange(len(col)), 0)
                        np.maximum.accumulate(idx, out=idx)
                        col[mask] = col[idx[mask]]
                        # Fill any remaining NaN with 0
                        col[np.isnan(col)] = 0
            
            # Normalize time series (per feature)
            ts_normalized = ts_imputed.copy()
            for j in range(ts_normalized.shape[2]):
                feature_data = ts_normalized[:, :, j]. flatten()
                mean = np.nanmean(feature_data)
                std = np.nanstd(feature_data) + 1e-8
                if np.isnan(mean):
                    mean = 0
                ts_normalized[:, :, j] = (ts_normalized[:, :, j] - mean) / std
            
            # Final check:  replace any remaining NaN with 0
            ts_normalized = np.nan_to_num(ts_normalized, nan=0.0)
        else:
            ts_normalized = time_series
        
        return static_normalized. astype(np.float32), ts_normalized.astype(np.float32)
    
    def prepare_dataset(
        self,
        cohort: pd.DataFrame,
        vitals_dict: Dict,
        labs_dict: Dict,
        diagnoses_dict: Dict,
        output_dir: str,
        hours: int = 48,
        interval: int = 4,
        test_size: float = 0.2,
        val_size: float = 0.1
    ):
        """
        Prepare and save train/val/test datasets.
        
        Args:
            cohort:  Cohort DataFrame
            vitals_dict:  Vitals dictionary
            labs_dict:  Labs dictionary
            diagnoses_dict: Diagnoses dictionary
            output_dir: Directory to save processed data
            hours: Hours of data to use
            interval: Time series interval
            test_size: Test set fraction
            val_size: Validation set fraction
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess
        static, time_series, labels, icustay_ids = self.preprocess(
            cohort, vitals_dict, labs_dict, diagnoses_dict,
            hours=hours, interval=interval
        )
        
        # Check if we have enough data
        if len(labels) == 0:
            raise ValueError("No patients with valid data found!")
        
        # Split data
        print("\nSplitting data...")
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=test_size,
            stratify=labels,
            random_state=42
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),
            stratify=labels[train_val_idx],
            random_state=42
        )
        
        print(f"  Train:  {len(train_idx)}")
        print(f"  Val: {len(val_idx)}")
        print(f"  Test: {len(test_idx)}")
        
        # Impute and normalize (fit on train only)
        static_train, ts_train = self.impute_and_normalize(
            static[train_idx], time_series[train_idx], fit=True
        )
        static_val, ts_val = self.impute_and_normalize(
            static[val_idx], time_series[val_idx], fit=False
        )
        static_test, ts_test = self.impute_and_normalize(
            static[test_idx], time_series[test_idx], fit=False
        )
        
        # Save datasets
        print("\nSaving datasets...")
        
        train_data = {
            'static': static_train,
            'time_series': ts_train,
            'labels': labels[train_idx],
            'icustay_ids': icustay_ids[train_idx]
        }
        
        val_data = {
            'static': static_val,
            'time_series':  ts_val,
            'labels': labels[val_idx],
            'icustay_ids':  icustay_ids[val_idx]
        }
        
        test_data = {
            'static': static_test,
            'time_series': ts_test,
            'labels': labels[test_idx],
            'icustay_ids': icustay_ids[test_idx]
        }
        
        with open(output_dir / 'train.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        
        with open(output_dir / 'val.pkl', 'wb') as f:
            pickle.dump(val_data, f)
        
        with open(output_dir / 'test.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        
        # Save feature info
        feature_info = {
            'feature_names': self.feature_names,
            'num_static_features': static_train.shape[1],
            'num_timesteps': ts_train.shape[1] if len(ts_train) > 0 else 0,
            'num_ts_features': ts_train.shape[2] if len(ts_train) > 0 else 0,
            'hours': hours,
            'interval':  interval,
            'train_size': len(train_idx),
            'val_size':  len(val_idx),
            'test_size': len(test_idx),
            'mortality_rate_train': float(labels[train_idx].mean()),
            'mortality_rate_test': float(labels[test_idx].mean()),
        }
        
        with open(output_dir / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Save scaler
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle. dump(self.scaler, f)
        
        print(f"\n✓ Data saved to {output_dir}/")
        
        return feature_info