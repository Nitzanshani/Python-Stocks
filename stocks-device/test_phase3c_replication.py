import json,tempfile,unittest
from pathlib import Path
import pandas as pd
from phase3c_periods import canonical_hash,load_frozen_spec,period_bounds,slice_period,assert_non_overlapping
from intraday_quality_monitor import require_quality_gate

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

if __name__=="__main__":unittest.main()
