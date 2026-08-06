import unittest
import numpy as np,pandas as pd
from intraday_features import build_intraday_features
from intraday_event_engine import cluster_intraday_events
from intraday_response_engine import measure_intraday_responses
from run_intraday_influence_research import classify_lead_lag
from intraday_predictive import session_walk_forward_splits

def frame_for_sessions(days=12):
 import pandas_market_calendars as mcal
 schedule=mcal.get_calendar("NYSE").schedule("2026-01-02","2026-03-01").head(days);rows=[];index=[]
 for _,session in schedule.iterrows():
  for minute in range(0,390,5):
   timestamp=session.market_open+pd.Timedelta(minutes=minute);price=100+minute/100
   index.append(timestamp);rows.append({"Open":price,"High":price+.1,"Low":price-.1,"Close":price+.02,
    "Adj Close":price+.02,"Volume":1000 if minute==0 else 100,"Dividends":0,"Stock Splits":0})
 return pd.DataFrame(rows,index=pd.DatetimeIndex(index,name="timestamp"))

class Phase3BTests(unittest.TestCase):
 def test_time_of_day_volume_baseline_uses_prior_sessions_only(self):
  raw=frame_for_sessions();features=build_intraday_features(raw,"A","5m",10)
  last=features.session_date.max();opening=features[(features.session_date==last)&(features.minutes_from_open==0)].iloc[0]
  noon=features[(features.session_date==last)&(features.minutes_from_open==180)].iloc[0]
  self.assertAlmostEqual(opening.relative_bar_volume,1.0);self.assertAlmostEqual(noon.relative_bar_volume,1.0)
  changed=raw.copy();last_indices=changed.index.tz_convert("America/New_York").date==last
  positions=np.where(last_indices)[0];changed.iloc[positions[0],changed.columns.get_loc("Volume")]=500
  changed.iloc[positions[36],changed.columns.get_loc("Volume")]=500
  result=build_intraday_features(changed,"A","5m",10);opening=result[(result.session_date==last)&(result.minutes_from_open==0)].iloc[0];noon=result[(result.session_date==last)&(result.minutes_from_open==180)].iloc[0]
  self.assertAlmostEqual(opening.relative_bar_volume,.5);self.assertAlmostEqual(noon.relative_bar_volume,5.0)

 def test_cluster_representative_is_first_not_future_maximum(self):
  times=pd.date_range("2026-01-05 15:00",periods=3,freq="5min",tz="UTC")
  events=pd.DataFrame({"event_id":["first","middle","max"],"ticker":"A","session_date":[times[0].date()]*3,
   "direction":"positive","available_at":times,"return_z_score":[3,4,9],"raw_return":[.01,.02,.08]})
  clusters,reps=cluster_intraday_events(events,15)
  self.assertEqual(reps.iloc[0].event_id,"first");self.assertEqual(reps.iloc[0].maximum_magnitude_event_id,"max")

 def test_ten_minute_lead_lag_and_reversal(self):
  starts=pd.date_range("2026-01-05 14:30",periods=6,freq="5min",tz="UTC");session=starts[0].tz_convert("America/New_York").date()
  residual=pd.DataFrame({"ticker":"B","session_date":session,"bar_start":starts,"bar_end":starts+pd.Timedelta(minutes=5),
   "session_close":pd.Timestamp("2026-01-05 21:00",tz="UTC"),"raw_bar_return":[0,.01,.08,-.04,0,0],
   "market_sector_adjusted_bar_residual":[0,.01,.08,-.04,0,0]})
  event=pd.DataFrame([{"event_id":"e","ticker":"A","session_date":session,"available_at":starts[1],"interval":"5m","direction":"positive","residual_return":.1}])
  response=measure_intraday_responses(event,residual,["B"],[10,15])
  ten=response[response.horizon_minutes==10].iloc[0]
  self.assertEqual(ten.time_to_max_positive,10);self.assertTrue(response[response.horizon_minutes==15].iloc[0].reversal_before_horizon)

 def test_missing_bar_and_session_boundary_are_rejected_without_fill(self):
  starts=pd.to_datetime(["2026-01-05 20:45Z","2026-01-05 20:55Z"]);session=pd.Timestamp(starts[0]).tz_convert("America/New_York").date()
  residual=pd.DataFrame({"ticker":"B","session_date":session,"bar_start":starts,"bar_end":starts+pd.Timedelta(minutes=5),
   "session_close":pd.Timestamp("2026-01-05 21:00Z"),"raw_bar_return":[.01,.01],"market_sector_adjusted_bar_residual":[.01,.01]})
  event=pd.DataFrame([{"event_id":"e","ticker":"A","session_date":session,"available_at":pd.Timestamp("2026-01-05 20:45Z"),"interval":"5m","direction":"positive","residual_return":.1}])
  result=measure_intraday_responses(event,residual,["B"],[10,30])
  self.assertEqual(result[result.horizon_minutes==10].iloc[0].missing_reason,"missing_bar")
  self.assertEqual(result[result.horizon_minutes==30].iloc[0].missing_reason,"session_boundary")

 def test_session_close_response_is_separate_and_causal(self):
  starts=pd.date_range("2026-01-05 20:40",periods=4,freq="5min",tz="UTC");session=starts[0].tz_convert("America/New_York").date()
  residual=pd.DataFrame({"ticker":"B","session_date":session,"bar_start":starts,"bar_end":starts+pd.Timedelta(minutes=5),
   "session_close":pd.Timestamp("2026-01-05 21:00Z"),"raw_bar_return":[.01,.01,.01,.01],"market_sector_adjusted_bar_residual":[.01,.01,.01,.01]})
  event=pd.DataFrame([{"event_id":"e","ticker":"A","session_date":session,"available_at":starts[1],"interval":"5m","direction":"positive","residual_return":.1}])
  result=measure_intraday_responses(event,residual,["B"],["session_close"]).iloc[0]
  self.assertTrue(result.response_available);self.assertEqual(result.horizon_minutes,-1);self.assertEqual(result.response_end,pd.Timestamp("2026-01-05 21:00Z"))

 def test_same_bar_precedes_unrelated_earlier_and_later_target_events(self):
  session=pd.Timestamp("2026-01-05").date();start=pd.Timestamp("2026-01-05 15:00Z");available=start+pd.Timedelta(minutes=5)
  reps=pd.DataFrame([
   {"event_id":"source","ticker":"A","session_date":session,"event_timestamp":start,"available_at":available},
   {"event_id":"old","ticker":"B","session_date":session,"event_timestamp":start-pd.Timedelta(minutes=15),"available_at":start-pd.Timedelta(minutes=10)},
   {"event_id":"same","ticker":"B","session_date":session,"event_timestamp":start,"available_at":available},
   {"event_id":"later","ticker":"B","session_date":session,"event_timestamp":available+pd.Timedelta(minutes=5),"available_at":available+pd.Timedelta(minutes=10)}])
  response=pd.DataFrame([{"event_id":"source","target_ticker":"B","session_date":session}])
  self.assertEqual(classify_lead_lag(response,reps,"5m").iloc[0].lead_lag_classification,"same_bar")

 def test_session_walk_forward_has_no_overlap_and_keeps_embargo(self):
  sessions=pd.date_range("2026-01-02",periods=40,freq="B").date
  folds=list(session_walk_forward_splits(sessions,20,5,5,1))
  self.assertGreaterEqual(len(folds),3)
  for _,train,test in folds:self.assertTrue(set(train).isdisjoint(test));self.assertLess(max(train),min(test))

if __name__=="__main__":unittest.main()
