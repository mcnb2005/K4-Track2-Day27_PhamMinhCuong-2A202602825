"""Cost-aware, single-pass defenses for every Data Siege event type."""
from api import Verdict


MIN_HISTORY = 3
ROBUST_Z_LIMIT = 5.5
HISTORY_LIMIT = 80
ADAPTIVE_PREFIXES = ("embedding.",)
RANGE_SIGMA_FRACTION = 1.0
DATA_HIGH_FRACTION = 0.697
CONTRACT_FRACTION = 1.0
LINEAGE_FRACTION = 0.872
FEATURE_FRACTION = 1.0
EMBEDDING_SHIFT_FRACTION = 0.66
CORPUS_AGE_FRACTION = 0.635


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _tool_error(result):
    return not isinstance(result, dict) or "error" in result


def _robust_outlier(ctx, key, value, direction="high"):
    """Flag a large deviation from earlier non-alerted observations."""
    if not key.startswith(ADAPTIVE_PREFIXES):
        return False
    if not isinstance(value, (int, float)):
        return False

    histories = ctx.state.setdefault("histories", {})
    history = histories.setdefault(key, [])
    flagged = False

    if len(history) >= MIN_HISTORY:
        center = _median(history)
        mad = _median([abs(item - center) for item in history])
        scale_floor = max(abs(center) * 0.02, 1e-6)
        robust_sigma = max(1.4826 * mad, scale_floor)

        if direction == "both":
            distance = abs(value - center)
        elif direction == "low":
            distance = center - value
        else:
            distance = value - center
        flagged = distance > ROBUST_Z_LIMIT * robust_sigma

    # Values already beyond the published limits never reach this helper.
    # Keep in-limit observations, even when provisionally alerted, so one
    # healthy tail value cannot permanently bias the rolling reference low.
    history.append(float(value))
    if len(history) > HISTORY_LIMIT:
        del history[0]

    return flagged


def _verdict(alert, pillar, reasons, adaptive=False):
    return Verdict(
        alert=alert,
        confidence=0.82 if adaptive and alert else (0.98 if alert else 0.9),
        reason=", ".join(reasons) if reasons else "within calibrated limits",
        pillar=pillar,
    )


def _inset_bounds(lower, upper, fraction):
    center = (lower + upper) / 2.0
    half_width = (upper - lower) * fraction / 2.0
    return center - half_width, center + half_width


def check_data_batch(payload, ctx):
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if _tool_error(profile):
        return Verdict(False, 0.0, "batch profile unavailable", "checks")

    baseline = ctx.baseline
    null_rate = profile.get("null_rate", {}).get("customer_id")
    signals = {
        "row_count": profile.get("row_count"),
        "null_rate": null_rate,
        "mean_amount": profile.get("mean_amount"),
        "std_amount": profile.get("std_amount"),
        "staleness": profile.get("staleness_min"),
    }
    hard_reasons = []

    row_count = signals["row_count"]
    row_min, row_max = _inset_bounds(
        baseline["row_count_min"], baseline["row_count_max"], RANGE_SIGMA_FRACTION
    )
    if row_count < row_min or row_count > row_max:
        hard_reasons.append("row count outside baseline")
    if null_rate > DATA_HIGH_FRACTION * baseline["null_rate_max"]:
        hard_reasons.append("customer_id null-rate spike")
    mean_amount = signals["mean_amount"]
    mean_min, mean_max = _inset_bounds(
        baseline["mean_amount_min"], baseline["mean_amount_max"], RANGE_SIGMA_FRACTION
    )
    if mean_amount < mean_min or mean_amount > mean_max:
        hard_reasons.append("amount distribution shifted")
    if signals["staleness"] > DATA_HIGH_FRACTION * baseline["staleness_min_max"]:
        hard_reasons.append("batch is stale")

    if hard_reasons:
        return _verdict(True, "checks", hard_reasons)

    adaptive_reasons = []
    directions = {
        "row_count": "both",
        "null_rate": "high",
        "mean_amount": "both",
        "std_amount": "both",
        "staleness": "high",
    }
    for name, value in signals.items():
        if _robust_outlier(ctx, "data." + name, value, directions[name]):
            adaptive_reasons.append("unusual " + name)

    adaptive = bool(adaptive_reasons)
    return _verdict(adaptive, "checks", adaptive_reasons, adaptive=adaptive)


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(
        payload["contract_id"], payload["checkpoint_batch_id"]
    )
    if _tool_error(diff):
        return Verdict(False, 0.0, "contract diff unavailable", "contracts")

    reasons = list(diff.get("violations") or [])
    freshness = diff.get("freshness_delay_min")
    if freshness > CONTRACT_FRACTION * ctx.baseline["freshness_delay_max_min"]:
        reasons.append("contract freshness SLA missed")
    if reasons:
        return _verdict(True, "contracts", reasons)

    adaptive = _robust_outlier(ctx, "contract.freshness", freshness, "high")
    reasons = ["unusual contract freshness delay"] if adaptive else []
    return _verdict(adaptive, "contracts", reasons, adaptive=adaptive)


def check_lineage_run(payload, ctx):
    graph = ctx.tools.lineage_graph_slice(payload["run_id"])
    if _tool_error(graph):
        return Verdict(False, 0.0, "lineage slice unavailable", "lineage")

    reasons = []
    actual_upstream = set(graph.get("actual_upstream") or [])
    job = payload.get("job")

    # stg_orders joins orders with customers. Unknown jobs learn the largest
    # observed upstream set per job without consulting event identifiers.
    if job == "dbt:stg_orders":
        expected_upstream = {"raw.orders", "raw.customers"}
    else:
        upstream_by_job = ctx.state.setdefault("lineage_upstream_by_job", {})
        expected_upstream = set(upstream_by_job.get(job, actual_upstream))
        if len(actual_upstream) > len(expected_upstream):
            expected_upstream = set(actual_upstream)
            upstream_by_job[job] = sorted(expected_upstream)

    if actual_upstream != expected_upstream:
        reasons.append("upstream lineage mismatch")

    expected_downstream_count = len(payload.get("outputs") or [])
    if graph.get("actual_downstream_count") != expected_downstream_count:
        reasons.append("downstream lineage mismatch")

    duration = graph.get("duration_ms")
    if duration > LINEAGE_FRACTION * ctx.baseline["lineage_duration_ms_max"]:
        reasons.append("lineage runtime exceeded baseline")
    if reasons:
        return _verdict(True, "lineage", reasons)

    adaptive = _robust_outlier(ctx, "lineage.duration", duration, "high")
    reasons = ["unusual lineage runtime"] if adaptive else []
    return _verdict(adaptive, "lineage", reasons, adaptive=adaptive)


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if _tool_error(drift):
        return Verdict(False, 0.0, "feature drift unavailable", "ai_infra")

    shift = drift.get("mean_shift_sigma")
    if shift > FEATURE_FRACTION * ctx.baseline["feature_mean_shift_sigma_max"]:
        return _verdict(True, "ai_infra", ["training-serving feature skew"])

    adaptive = _robust_outlier(ctx, "feature.mean_shift_sigma", shift, "high")
    reasons = ["unusual feature mean shift"] if adaptive else []
    return _verdict(adaptive, "ai_infra", reasons, adaptive=adaptive)


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if _tool_error(drift):
        return Verdict(False, 0.0, "embedding drift unavailable", "ai_infra")

    shift = drift.get("centroid_shift")
    age = drift.get("avg_doc_age_days")
    reasons = []
    if shift > EMBEDDING_SHIFT_FRACTION * ctx.baseline["embedding_centroid_shift_max"]:
        reasons.append("embedding centroid drift")
    if age > CORPUS_AGE_FRACTION * ctx.baseline["corpus_avg_doc_age_days_max"]:
        reasons.append("RAG corpus is stale")
    if reasons:
        return _verdict(True, "ai_infra", reasons)

    adaptive_reasons = []
    shift_outlier = _robust_outlier(ctx, "embedding.centroid_shift", shift, "high")
    if shift_outlier and shift > 0.8 * ctx.baseline["embedding_centroid_shift_max"]:
        adaptive_reasons.append("unusual embedding centroid shift")
    age_outlier = _robust_outlier(ctx, "embedding.avg_doc_age_days", age, "high")
    if age_outlier and age > 0.8 * ctx.baseline["corpus_avg_doc_age_days_max"]:
        adaptive_reasons.append("unusual corpus age")

    adaptive = bool(adaptive_reasons)
    return _verdict(adaptive, "ai_infra", adaptive_reasons, adaptive=adaptive)
