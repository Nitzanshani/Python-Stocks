import unittest

import numpy as np
import pandas as pd

from influence_features import build_feature_panel
from influence_statistics import (apply_fdr_families, benjamini_hochberg,
                                  build_relationships, run_granger_all_directions)
from event_study import select_control_days
from predictive_influence import purged_walk_forward_splits
from residual_returns import build_residual_returns
from response_engine import measure_daily_responses
from test_phase3a_engines import prices


class Phase3AStatisticsTests(unittest.TestCase):
    def test_purged_walk_forward_has_causal_gap(self):
        splits = list(purged_walk_forward_splits(900, 504, 63, 21, 10, 5))
        self.assertGreater(len(splits), 5)
        for _, train, test in splits:
            self.assertLess(train.max(), test.min() - 9)
            self.assertFalse(set(train) & set(test))

    def test_common_market_link_weakens_after_residualization(self):
        rng=np.random.default_rng(10); n=500
        market=rng.normal(0,.015,n); sector=rng.normal(0,.01,n)
        a=.9*market+.2*sector+rng.normal(0,.003,n)
        b=.8*market+.2*sector+rng.normal(0,.003,n)
        panel=build_feature_panel({"SPY":prices(market),"SMH":prices(sector),
                                   "A":prices(a),"B":prices(b)},20)
        residual=build_residual_returns(panel,"SPY","SMH",126,63)
        wide=residual.pivot(index="timestamp",columns="ticker",values=["raw_return","residual_return"])
        raw=wide.raw_return.A.corr(wide.raw_return.B)
        adjusted=wide.residual_return.A.corr(wide.residual_return.B)
        self.assertLess(abs(adjusted), abs(raw)*.25)

    def test_synthetic_lag_two_is_stronger_forward_than_reverse(self):
        rng=np.random.default_rng(12); n=700
        a=rng.normal(0,1,n); b=rng.normal(0,.2,n)
        b[2:]+=1.2*a[:-2]
        dates=pd.bdate_range("2020-01-01",periods=n,tz="UTC")
        rows=[]
        for ticker,values in (("A",a),("B",b)):
            rows += [{"ticker":ticker,"timestamp":d,"residual_return":v}
                     for d,v in zip(dates,values)]
        result=run_granger_all_directions(pd.DataFrame(rows),["A","B"],[1,2,3,5],.05)
        forward=result[(result.source_ticker=="A")&(result.target_ticker=="B")]
        reverse=result[(result.source_ticker=="B")&(result.target_ticker=="A")]
        self.assertEqual(int(forward.loc[forward.raw_p_value.idxmin(),"lag"]),2)
        self.assertLess(forward.raw_p_value.min(), reverse.raw_p_value.min())

    def test_bidirectional_process_reports_both_directions(self):
        rng=np.random.default_rng(3); n=800; a=np.zeros(n); b=np.zeros(n)
        for t in range(1,n):
            a[t]=.35*b[t-1]+rng.normal(0,.3)
            b[t]=.75*a[t-1]+rng.normal(0,.3)
        dates=pd.bdate_range("2020-01-01",periods=n,tz="UTC")
        frame=pd.DataFrame([{"ticker":ticker,"timestamp":date,"residual_return":value}
            for ticker,values in (("A",a),("B",b)) for date,value in zip(dates,values)])
        result=run_granger_all_directions(frame,["A","B"],[1],.05)
        self.assertEqual(len(result),2)
        self.assertTrue(result.fdr_significant.all())

    def test_decay_response_peak_and_horizons(self):
        dates=pd.bdate_range("2026-01-02",periods=6,tz="UTC")
        residual=pd.DataFrame({"ticker":"B","timestamp":dates,
            "raw_return":[0,.10,.06,.03,-.01,0],"market_residual_return":[0,.10,.06,.03,-.01,0],
            "residual_return":[0,.10,.06,.03,-.01,0]})
        event=pd.DataFrame([{"event_id":"e","ticker":"A","event_date":dates[0],
            "direction":"positive","raw_return":.2,"residual_return":.2}])
        response=measure_daily_responses(event,residual,["B"],[1,2,3,5])
        self.assertTrue(response.target_raw_return.iloc[:3].is_monotonic_increasing)
        self.assertEqual(response.loc[response.horizon==3,"time_to_max_positive"].iloc[0],3)

    def test_bh_fdr_limits_random_false_discoveries_and_keeps_families_separate(self):
        rng=np.random.default_rng(99)
        frame=pd.DataFrame({"p":rng.uniform(size=400),"analysis":["event"]*200+["granger"]*200})
        result=apply_fdr_families(frame,"p",["analysis"],.05)
        self.assertLessEqual(int(result.fdr_significant.sum()),5)
        self.assertEqual(set(result.number_of_tests),{200})

    def test_control_matching_records_that_target_future_was_not_used(self):
        dates=pd.bdate_range("2024-01-01",periods=80,tz="UTC")
        rows=[]
        for ticker in ("A","B"):
            for i,date in enumerate(dates):
                rows.append({"ticker":ticker,"timestamp":date,"market_return":.01 if i%2 else -.01,
                    "sector_return":.01 if i%3 else -.01,"residual_return":.001*(i%5)})
        residual=pd.DataFrame(rows); event_date=dates[40]
        events=pd.DataFrame([{"event_id":"e","ticker":"A","event_date":event_date,
                              "event_type":"raw_return"}])
        responses=pd.DataFrame([{"event_id":"e","source_ticker":"A","target_ticker":"B",
            "event_date":event_date,"horizon":2,"response_status":"ok"}])
        controls=select_control_days(events,responses,residual,3)
        self.assertEqual(len(controls),3)
        self.assertFalse(controls.target_future_used_for_matching.any())
        self.assertTrue(controls.matching_features.str.contains("market_volatility").all())

    def test_regime_change_is_unstable_and_small_sample_is_insufficient(self):
        folds=pd.DataFrame({"source_ticker":"A","target_ticker":"B","fold":range(10),
            "baseline_rmse":[1.0]*10,"extended_rmse":[.9]*3+[1.2]*7,
            "rmse_improvement":[.1]*3+[-.2]*7,"direction_improvement":[0]*10})
        event=pd.DataFrame([{"source_ticker":"A","target_ticker":"B","event_count":20,
            "event_type":"raw_return","horizon":2,"abnormal_effect":.1,"effect_size":.25,
            "adjusted_p_value":.2}])
        granger=pd.DataFrame([{"source_ticker":"A","target_ticker":"B","lag":1,
            "adjusted_p_value":.5,"fdr_significant":False}])
        config={"minimum_event_count":5,"minimum_folds":3}
        relationship=build_relationships(event,folds,granger,config).iloc[0]
        self.assertEqual(relationship.relationship_status,"unstable")
        event.loc[0,"event_count"]=2
        relationship=build_relationships(event,folds,granger,config).iloc[0]
        self.assertEqual(relationship.relationship_status,"insufficient_data")


if __name__ == "__main__": unittest.main()
