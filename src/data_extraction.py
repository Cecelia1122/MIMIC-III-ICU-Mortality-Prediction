"""
Data extraction module for MIMIC-III. 
Extracts and joins relevant tables for ICU outcome prediction.
"""

import os
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class MIMICDataExtractor: 
    """
    Extract and process data from MIMIC-III CSV files.
    
    Extracts: 
        - Patient demographics (PATIENTS, ADMISSIONS)
        - ICU stays (ICUSTAYS)
        - Vital signs (CHARTEVENTS)
        - Laboratory tests (LABEVENTS)
        - Diagnoses (DIAGNOSES_ICD)
    """
    
    # Vital signs ItemIDs (CHARTEVENTS)
    VITAL_SIGNS = {
        'heart_rate': [211, 220045],
        'sbp': [51, 442, 455, 6701, 220179, 220050],  # Systolic BP
        'dbp': [8368, 8440, 8441, 8555, 220180, 220051],  # Diastolic BP
        'mean_bp': [456, 52, 6702, 443, 220052, 220181, 225312],
        'resp_rate': [615, 618, 220210, 224690],
        'temperature': [223761, 678, 223762, 676],  # Celsius and Fahrenheit
        'spo2': [646, 220277],  # Oxygen saturation
        'gcs': [198, 226755, 227013],  # Glasgow Coma Scale
    }
    
    # Laboratory tests ItemIDs (LABEVENTS)
    LAB_TESTS = {
        'glucose': [50931, 50809],
        'potassium': [50971, 50822],
        'sodium': [50983, 50824],
        'chloride': [50902, 50806],
        'bicarbonate': [50882, 50803],
        'bun': [51006],  # Blood Urea Nitrogen
        'creatinine': [50912],
        'hemoglobin': [51222, 50811],
        'hematocrit': [51221, 50810],
        'wbc': [51301, 51300],  # White Blood Cells
        'platelets': [51265],
        'lactate': [50813],
    }
    
    def __init__(self, mimic_path: str):
        """
        Initialize extractor.
        
        Args:
            mimic_path: Path to MIMIC-III directory containing CSV. gz files
        """
        self. mimic_path = Path(mimic_path)
        
        if not self.mimic_path. exists():
            raise FileNotFoundError(f"MIMIC-III path not found: {self.mimic_path}")
        
        # Check for required files
        required_files = ['PATIENTS.csv.gz', 'ADMISSIONS.csv.gz', 'ICUSTAYS.csv.gz']
        for f in required_files:
            if not (self.mimic_path / f).exists():
                # Try without . gz
                if not (self.mimic_path / f. replace('.gz', '')).exists():
                    raise FileNotFoundError(f"Required file not found: {f}")
    
    def _load_table(self, table_name: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load a MIMIC-III table."""
        # Try . csv.gz first, then .csv
        gz_path = self.mimic_path / f"{table_name}.csv.gz"
        csv_path = self.mimic_path / f"{table_name}.csv"
        
        if gz_path.exists():
            print(f"  Loading {table_name}.csv.gz...")
            return pd.read_csv(gz_path, usecols=usecols, low_memory=False)
        elif csv_path.exists():
            print(f"  Loading {table_name}.csv...")
            return pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        else:
            raise FileNotFoundError(f"Table not found: {table_name}")
    
    def extract_cohort(
        self,
        min_age: int = 18,
        min_los_hours: float = 24,
        max_los_days: float = 30,
        first_icu_only: bool = True
    ) -> pd.DataFrame:
        """
        Extract ICU stay cohort with inclusion criteria.
        
        Args:
            min_age: Minimum age (default 18)
            min_los_hours: Minimum ICU length of stay in hours
            max_los_days: Maximum ICU length of stay in days
            first_icu_only:  Use only first ICU stay per patient
            
        Returns:
            DataFrame with cohort information
        """
        print("\nExtracting cohort...")
        
        # Load tables
        patients = self._load_table('PATIENTS', usecols=[
            'SUBJECT_ID', 'GENDER', 'DOB', 'DOD'
        ])
        
        admissions = self._load_table('ADMISSIONS', usecols=[
            'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 
            'DEATHTIME', 'ETHNICITY', 'ADMISSION_TYPE', 'HOSPITAL_EXPIRE_FLAG'
        ])
        
        icustays = self._load_table('ICUSTAYS', usecols=[
            'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'FIRST_CAREUNIT',
            'INTIME', 'OUTTIME', 'LOS'
        ])
        
        # Convert datetime columns
        for col in ['DOB', 'DOD']: 
            patients[col] = pd.to_datetime(patients[col], errors='coerce')
        
        for col in ['ADMITTIME', 'DISCHTIME', 'DEATHTIME']: 
            admissions[col] = pd.to_datetime(admissions[col], errors='coerce')
        
        for col in ['INTIME', 'OUTTIME']:
            icustays[col] = pd. to_datetime(icustays[col], errors='coerce')
        
        # Merge tables
        cohort = icustays.merge(admissions, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        cohort = cohort.merge(patients, on='SUBJECT_ID', how='left')
        
        print(f"  Initial ICU stays: {len(cohort)}")
        
        # Calculate age at admission
        #cohort['AGE'] = (cohort['INTIME'] - cohort['DOB']).dt.days / 365.25
        # 1. 识别并修正极端高龄（MIMIC 规定 >89 岁的 DOB 会被设为 ~300年前）
        # 我们先计算年份差，而不是纳秒差，以避免溢出
        cohort['AGE'] = cohort['INTIME'].dt.year - cohort['DOB'].dt.year

        # 2. 修正 MIMIC 的 300 岁逻辑
        # 如果年龄 > 150 (通常是300左右)，按照 MIMIC 官方建议将其统一设为 90 岁
        cohort.loc[cohort['AGE'] > 150, 'AGE'] = 90
        
        # Handle patients > 89 years (MIMIC shifts DOB for privacy)
        #cohort.loc[cohort['AGE'] > 200, 'AGE'] = 91.4
        cohort['AGE'] = cohort['AGE'].astype(float) 
        cohort.loc[cohort['AGE'] > 200, 'AGE'] = 91.4
        # Apply inclusion criteria
        # Age >= 18
        cohort = cohort[cohort['AGE'] >= min_age]
        print(f"  After age filter (>={min_age}): {len(cohort)}")
        
        # ICU LOS >= min_los_hours
        cohort = cohort[cohort['LOS'] * 24 >= min_los_hours]
        print(f"  After min LOS filter (>={min_los_hours}h): {len(cohort)}")
        
        # ICU LOS <= max_los_days
        cohort = cohort[cohort['LOS'] <= max_los_days]
        print(f"  After max LOS filter (<={max_los_days}d): {len(cohort)}")
        
        # First ICU stay only
        if first_icu_only: 
            cohort = cohort.sort_values(['SUBJECT_ID', 'INTIME'])
            cohort = cohort.groupby('SUBJECT_ID').first().reset_index()
            print(f"  After first ICU only: {len(cohort)}")
        
        # Create mortality label
        cohort['MORTALITY'] = cohort['HOSPITAL_EXPIRE_FLAG'].fillna(0).astype(int)
        
        # Calculate actual ICU LOS in hours
        cohort['LOS_HOURS'] = cohort['LOS'] * 24
        
        print(f"\n  Final cohort size: {len(cohort)}")
        print(f"  Mortality rate: {cohort['MORTALITY'].mean():.2%}")
        
        return cohort
    
    def extract_vitals(
        self,
        cohort: pd.DataFrame,
        hours:  int = 48
    ) -> Dict[int, pd.DataFrame]:
        """
        Extract vital signs for first N hours of ICU stay.
        
        Args:
            cohort:  Cohort DataFrame with ICUSTAY_ID and INTIME
            hours: Number of hours to extract
            
        Returns:
            Dictionary mapping ICUSTAY_ID to vital signs DataFrame
        """
        print(f"\nExtracting vital signs (first {hours} hours)...")
        
      # 1. 准备过滤条件
        icustay_ids = set(cohort['ICUSTAY_ID'].values)
        all_itemids = []
        for itemids in self.VITAL_SIGNS.values():
            all_itemids.extend(itemids)
        all_itemids = set(all_itemids)

        # 2. 分块读取 CHARTEVENTS.csv.gz (这是防止死机的关键)
        print("  Loading CHARTEVENTS.csv.gz in chunks...")
        vitals_list = []
        chunksize = 1000000 # 每次读取 100 万行
        
        # 找到文件路径
        gz_path = self.mimic_path / "CHARTEVENTS.csv.gz"
        csv_path = self.mimic_path / "CHARTEVENTS.csv"
        target_path = gz_path if gz_path.exists() else csv_path

        # 使用 tqdm 显示扫描进度
        with tqdm(unit=' lines') as pbar:
            for chunk in pd.read_csv(
                target_path, 
                usecols=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'],
                chunksize=chunksize,
                low_memory=False,
                engine='c'
            ):
                # 立即在块内过滤：只保留目标病人和目标指标
                filtered_chunk = chunk[
                    (chunk['ICUSTAY_ID'].isin(icustay_ids)) & 
                    (chunk['ITEMID'].isin(all_itemids))
                ].dropna(subset=['VALUENUM'])
                
                vitals_list.append(filtered_chunk)
                pbar.update(chunksize)

        # 合并过滤后的结果
        chartevents = pd.concat(vitals_list, axis=0)
        print(f"  Filtered chart events: {len(chartevents)}")
        
        # Convert charttime
        chartevents['CHARTTIME'] = pd.to_datetime(chartevents['CHARTTIME'], errors='coerce')
        
        # Create ITEMID to vital name mapping
        itemid_to_vital = {}
        for vital_name, itemids in self.VITAL_SIGNS.items():
            for itemid in itemids: 
                itemid_to_vital[itemid] = vital_name
        
        chartevents['NAME'] = chartevents['ITEMID'].map(itemid_to_vital)
        
        # Merge with cohort to get INTIME
        chartevents = chartevents.merge(
            cohort[['ICUSTAY_ID', 'INTIME']],
            on='ICUSTAY_ID',
            how='left'
        )
        
        # Calculate hours since ICU admission
        chartevents['HOURS'] = (
            chartevents['CHARTTIME'] - chartevents['INTIME']
        ).dt.total_seconds() / 3600
        
        # Filter to first N hours
        chartevents = chartevents[
            (chartevents['HOURS'] >= 0) & (chartevents['HOURS'] <= hours)
        ]
        
        print(f"  Events in first {hours} hours: {len(chartevents)}")
        
        # Group by ICUSTAY_ID
        vitals_dict = {}
        for icustay_id, group in tqdm(chartevents.groupby('ICUSTAY_ID'), desc="Processing vitals"):
            vitals_dict[icustay_id] = group[['HOURS', 'NAME', 'VALUENUM']].copy()
        
        print(f"  Patients with vitals: {len(vitals_dict)}")
        
        return vitals_dict
    
    def extract_labs(
        self,
        cohort: pd.DataFrame,
        hours: int = 48
    ) -> Dict[int, pd.DataFrame]:
        """
        Extract laboratory tests for first N hours of ICU stay.
        
        Args:
            cohort: Cohort DataFrame
            hours: Number of hours to extract
            
        Returns:
            Dictionary mapping ICUSTAY_ID to lab tests DataFrame
        """
        print(f"\nExtracting lab tests (first {hours} hours)...")
        
        # 1. 准备过滤条件
        hadm_ids = set(cohort['HADM_ID'].values)
        all_itemids = []
        for itemids in self.LAB_TESTS.values():
            all_itemids.extend(itemids)
        all_itemids = set(all_itemids)

        # 2. 分块读取 LABEVENTS
        vitals_list = []
        chunksize = 1000000
        gz_path = self.mimic_path / "LABEVENTS.csv.gz"
        csv_path = self.mimic_path / "LABEVENTS.csv"
        target_path = gz_path if gz_path.exists() else csv_path

        with tqdm(unit=' lines') as pbar:
            for chunk in pd.read_csv(
                target_path, 
                usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'],
                chunksize=chunksize,
                low_memory=False
            ):
                filtered_chunk = chunk[
                    (chunk['HADM_ID'].isin(hadm_ids)) & 
                    (chunk['ITEMID'].isin(all_itemids))
                ].dropna(subset=['VALUENUM'])
                
                vitals_list.append(filtered_chunk)
                pbar.update(chunksize)

        labevents = pd.concat(vitals_list, axis=0)
        
        print(f"  Filtered lab events:  {len(labevents)}")
        
        # Convert charttime
        labevents['CHARTTIME'] = pd.to_datetime(labevents['CHARTTIME'], errors='coerce')
        
        # Create ITEMID to lab name mapping
        itemid_to_lab = {}
        for lab_name, itemids in self. LAB_TESTS.items():
            for itemid in itemids:
                itemid_to_lab[itemid] = lab_name
        
        labevents['NAME'] = labevents['ITEMID'].map(itemid_to_lab)
        
        # Merge with cohort to get INTIME
        labevents = labevents.merge(
            cohort[['HADM_ID', 'ICUSTAY_ID', 'INTIME']],
            on='HADM_ID',
            how='left'
        )
        
        # Calculate hours since ICU admission
        labevents['HOURS'] = (
            labevents['CHARTTIME'] - labevents['INTIME']
        ).dt.total_seconds() / 3600
        
        # Filter to first N hours
        labevents = labevents[
            (labevents['HOURS'] >= 0) & (labevents['HOURS'] <= hours)
        ]
        
        print(f"  Events in first {hours} hours: {len(labevents)}")
        
        # Group by ICUSTAY_ID
        labs_dict = {}
        for icustay_id, group in tqdm(labevents.groupby('ICUSTAY_ID'), desc="Processing labs"):
            labs_dict[icustay_id] = group[['HOURS', 'NAME', 'VALUENUM']].copy()
        
        print(f"  Patients with labs: {len(labs_dict)}")
        
        return labs_dict
    
    def extract_diagnoses(self, cohort: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Extract ICD-9 diagnosis codes. 
        
        Args:
            cohort: Cohort DataFrame
            
        Returns:
            Dictionary mapping ICUSTAY_ID to list of ICD codes
        """
        print("\nExtracting diagnoses...")
        
        diagnoses = self._load_table('DIAGNOSES_ICD', usecols=[
            'HADM_ID', 'ICD9_CODE', 'SEQ_NUM'
        ])
        
        # Filter to cohort
        hadm_ids = set(cohort['HADM_ID'].values)
        diagnoses = diagnoses[diagnoses['HADM_ID'].isin(hadm_ids)]
        
        # Merge to get ICUSTAY_ID
        diagnoses = diagnoses.merge(
            cohort[['HADM_ID', 'ICUSTAY_ID']],
            on='HADM_ID',
            how='left'
        )
        
        # Group by ICUSTAY_ID
        diag_dict = {}
        for icustay_id, group in diagnoses.groupby('ICUSTAY_ID'):
            # Sort by sequence number and get codes
            codes = group.sort_values('SEQ_NUM')['ICD9_CODE'].dropna().tolist()
            diag_dict[icustay_id] = codes
        
        print(f"  Patients with diagnoses: {len(diag_dict)}")
        
        return diag_dict
    
    def extract_all(
        self,
        hours: int = 48,
        min_age: int = 18,
        min_los_hours: float = 24,
        max_los_days: float = 30
    ) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
        """
        Extract all data for ICU outcome prediction.
        
        Args:
            hours: Hours of data to extract
            min_age:  Minimum patient age
            min_los_hours: Minimum ICU LOS in hours
            max_los_days: Maximum ICU LOS in days
            
        Returns: 
            Tuple of (cohort, vitals_dict, labs_dict, diagnoses_dict)
        """
        # Extract cohort
        cohort = self.extract_cohort(
            min_age=min_age,
            min_los_hours=min_los_hours,
            max_los_days=max_los_days
        )
        
        # Extract vitals
        vitals = self.extract_vitals(cohort, hours=hours)
        
        # Extract labs
        labs = self.extract_labs(cohort, hours=hours)
        
        # Extract diagnoses
        diagnoses = self.extract_diagnoses(cohort)
        
        return cohort, vitals, labs, diagnoses