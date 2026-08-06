import json,tempfile,unittest
from pathlib import Path
import pandas as pd
from phase3c_periods import canonical_hash,load_frozen_spec,period_bounds,slice_period,assert_non_overlapping
from intraday_quality_monitor import require_quality_gate
from phase3c_analysis import relative_rmse_improvement,practical_effect_classification,leave_one_event_out,rolling_stability,generate_placebos,spread_proxy_bps,pending_relationships

BASE=Path(__file__).resolve().parent

class Phase3CFoundationTests(unittest.TestCase):
 def test_frozen_spec_hash_matches_lock(self):
  expected=(BASE/"PHASE3B_FROZEN_SPEC.sha256").read_text().strip()
  self.assertEqual(canonical_hash(BASE/"PHASE3B_FROZEN_SPEC.json"),expected)
  self.assertEqual(load_frozen_spec(BASE/"PHASE3B_FROZEN_SPEC.json",expected)["_frozen_spec_hash"],expected)

 def test_changed_frozen_configuration_is_rejected(self):
  source=json.loads((BASE/"PHASE3B_FROZEN_SPEC.json").read_text());source["fdr"]["alpha"]=.10
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"changed.json";path.write_text(json.dumps(source))
   with self.assertRaises(ValueError):load_frozen_spec(path,(BASE/"PHASE3B_FROZEN_SPEC.sha256").read_text().strip())

 def test_periods_are_disjoint_and_confirmation_only(self):
  spec=load_frozen_spec(BASE/"PHASE3B_FROZEN_SPEC.json");self.assertTrue(assert_non_overlapping(spec))
  frame=pd.DataFrame({"session_date":["2026-08-05","2026-08-06","2026-10-29","2026-10-30"],"value":[1,2,3,4]})
  confirmation=slice_period(frame,spec,"confirmation")
  self.assertEqual(confirmation.value.tolist(),[2,3])

 def test_holdout_requires_explicit_unlock(self):
  spec=load_frozen_spec(BASE/"PHASE3B_FROZEN_SPEC.json")
  with self.assertRaises(PermissionError):period_bounds(spec,"holdout")
  self.assertEqual(period_bounds(spec,"holdout",True)[0],"2026-10-30")

 def test_material_quality_failure_blocks_replication(self):
  bad=pd.DataFrame([{"ticker":"SPY","interval":"5m","quality_gate_passed":False}])
  with self.assertRaises(RuntimeError):require_quality_gate(bad)
  good=bad.assign(quality_gate_passed=True);self.assertTrue(require_quality_gate(good))

 def test_relative_rmse_and_minimum_practical_effect(self):
  self.assertAlmostEqual(relative_rmse_improvement(.01,.009),.1)
  thresholds={"relative_rmse_improvement":.005,"absolute_rmse_improvement":.00001,"effect_bps":1,"effect_to_cost_proxy_ratio":1}
  meaningful={"relative_rmse_improvement":.01,"absolute_rmse_improvement":.00002,"effect_bps":4,"effect_to_cost_proxy_ratio":1.2,"fdr_significant":True}
  tiny={"relative_rmse_improvement":.0001,"absolute_rmse_improvement":.000001,"effect_bps":.1,"effect_to_cost_proxy_ratio":.01,"fdr_significant":True}
  self.assertEqual(practical_effect_classification(meaningful,thresholds),"potentially_meaningful")
  self.assertEqual(practical_effect_classification(tiny,thresholds),"statistically_detectable_but_negligible")

 def test_leave_one_event_out_detects_concentration(self):
  stable=leave_one_event_out([1,1.1,.9,1.05]);concentrated=leave_one_event_out([0,0,0,10])
  self.assertFalse(stable["event_concentrated"]);self.assertTrue(concentrated["event_concentrated"])

 def test_rolling_stability_has_predeclared_windows(self):
  frame=pd.DataFrame({"session_date":pd.date_range("2026-01-01",periods=30,freq="B").date,"effect":range(30)})
  result=rolling_stability(frame,window=20,step=5);self.assertEqual(len(result),3);self.assertTrue((result.sessions==20).all())

 def test_placebos_are_deterministic(self):
  frame=pd.DataFrame({"session_date":list(pd.date_range("2026-01-01",periods=5).date),"x":range(5)})
  a=generate_placebos(frame,7);b=generate_placebos(frame,7)
  self.assertTrue(a["shuffled_source"].equals(b["shuffled_source"]));self.assertTrue(a["future_to_past"].future_to_past_only_diagnostic.all())

 def test_spread_proxy_is_documented_price_range_not_bid_ask(self):
  bars=pd.DataFrame({"High":[101,102,103],"Low":[99,100,101],"Close":[100,101,102],"Volume":[1000,1200,1100]})
  result=spread_proxy_bps(bars);self.assertGreater(result["spread_proxy_bps"],0);self.assertIn("range_proxy_bps",result)

 def test_all_discovery_relationships_are_retained_while_pending(self):
  discovery=pd.DataFrame({"source_ticker":["A","B"],"target_ticker":["B","A"],"interval":["5m","5m"],"horizon_minutes":[5,5],"intraday_status":["candidate","rejected"],"effect_size":[.2,-.1],"prediction_improvement":[.01,-.01]})
  result=pending_relationships(discovery,0);self.assertEqual(len(result),2);self.assertTrue((result.replication_result=="insufficient_data").all())

 def test_fixed_nvda_anet_relationship_definition_is_unchanged(self):
  spec=load_frozen_spec(BASE/"PHASE3B_FROZEN_SPEC.json");fixed=spec["replication"]["fixed_relationship"]
  self.assertEqual(fixed,{"source":"NVDA","target":"ANET","interval":"5m","horizon":5})

if __name__=="__main__":unittest.main()
