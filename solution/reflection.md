# Reflection

**Which fault types were hardest to catch, and why?**

The hardest cases were low-magnitude feature skew and lineage topology faults.
Large freshness, volume, null-rate, schema, runtime, embedding, and corpus-age
faults cross the published limits directly. A subtle feature shift can resemble
the upper tail of healthy traffic, while a missing lineage edge requires knowing
the transform's expected graph rather than treating runtime alone as evidence.
I therefore use the published limits as the primary rules, explicitly validate
both sides of the `stg_orders` lineage, and add a run-local median/MAD detector
for smaller deviations after enough clean history has accumulated.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

The current strategy makes exactly one metered call for each event. That gives
full coverage at 180 credits in practice and about 300 credits on the 200-event
private stream, both within their budgets. I avoided repeat calls because they
return no new evidence and only increase the overage penalty. With another pass,
I would calibrate the robust threshold per metric on several independent clean
streams and make the fallback for previously unseen lineage jobs learn a
job-specific graph more explicitly. This could improve recall on subtle faults
without trading it for a larger false-positive rate.
