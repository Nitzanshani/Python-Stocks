"""Immutable Phase 3C period boundaries and frozen-spec verification."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd

def canonical_hash(path):
 data=json.loads(Path(path).read_text());canonical=json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=False)
 return hashlib.sha256(canonical.encode()).hexdigest()

def load_frozen_spec(path,expected_hash=None):
 path=Path(path);spec=json.loads(path.read_text());actual=canonical_hash(path)
 if expected_hash and actual!=expected_hash:raise ValueError(f"Frozen specification hash mismatch: {actual} != {expected_hash}")
 spec["_frozen_spec_hash"]=actual;return spec

def period_bounds(spec,period,unlock_holdout=False):
 p=spec["periods"]
 if period=="discovery":return p["discovery_start"],p["discovery_end"]
 if period=="confirmation":return p["confirmation_start"],p["confirmation_end"]
 if period=="holdout":
  if not unlock_holdout:raise PermissionError("Holdout is locked; pass --unlock-holdout explicitly")
  return p["holdout_start"],None
 raise ValueError(f"Unknown period: {period}")

def slice_period(frame,spec,period,unlock_holdout=False,date_column="session_date"):
 start,end=period_bounds(spec,period,unlock_holdout);dates=pd.to_datetime(frame[date_column]).dt.date
 mask=dates>=pd.Timestamp(start).date()
 if end:mask&=dates<=pd.Timestamp(end).date()
 return frame.loc[mask].copy()

def assert_non_overlapping(spec):
 p=spec["periods"]
 assert pd.Timestamp(p["discovery_end"])<pd.Timestamp(p["confirmation_start"])
 assert pd.Timestamp(p["confirmation_end"])<pd.Timestamp(p["holdout_start"])
 return True
