# Prospective protocol deviations

This is append-only and does not alter the frozen hashes.

1. The UCI download is a wrapper archive containing the declared data archive
   plus unrelated example CSV files. The initial loader tried those wrapper
   CSVs and failed before producing any outcome result. It was corrected to
   open the nested `PRSA2017_Data_20130301-20170228.zip`; the frozen source,
   target, states, metric, model, operators, and criteria are unchanged.
2. The frozen BLS URL used the legacy `special.requests` path. The official
   2024 tables page now links `special-requests/oesm24st.zip`; both official
   endpoints return HTTP 403 from this execution environment. No BLS outcome
   was acquired and no replacement source was introduced. The source remains
   `NOT RUN — SOURCE UNAVAILABLE` as required by the frozen replacement rule.
