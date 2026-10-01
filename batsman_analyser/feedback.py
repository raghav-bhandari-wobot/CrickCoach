"""Turn KPIs into a few prioritised, actionable coaching cues."""

RULES = [  # (kpi key, priority, message builder)
    ("head_max_drift_downswing", 1,
     lambda k: f"Head moves {k['value']} TU sideways as the bat comes down. Keep it still and over the line "
               "of the ball - try the 'balance a coin on your head' shadow drill."),
    ("head_over_front_foot", 1,
     lambda k: f"At contact the head is {abs(k['value'])} TU {'outside' if k['value'] > 0 else 'inside'} the front foot. "
               "Lead with the head and let the front foot land under it."),
    ("front_foot_plant_before_impact", 2,
     lambda k: f"Front foot lands {abs(k['value'])} ms after the bat meets the ball. Start the stride earlier "
               "(trigger movement as the bowler loads) so the base is set before the downswing."),
    ("balance_at_impact", 2,
     lambda k: f"Hip centre is {abs(k['value'])} of the base width off-centre at impact - weight is falling "
               f"{'towards off side' if k['value'] > 0 else 'towards leg side'}. Get the weight onto the front knee."),
    ("bat_angle_at_impact", 3,
     lambda k: f"Bat is {k['value']} deg off vertical at impact on a straight-bat shot. Present the full face: "
               "top hand in control, elbow pointing at the bowler."),
    ("bat_path_lateral", 3,
     lambda k: f"Bat toe travels {k['value']} TU sideways during the downswing - the bat is coming across the line. "
               "Bring it down from over the stumps (throw-downs with a stump line on the ground)."),
    ("eye_line_tilt", 4,
     lambda k: f"Eyes are tilted {k['value']} deg at contact. Keep the eye-line level to judge line and length."),
    ("finish_stillness", 4,
     lambda k: f"Head moves {k['value']} TU after the shot - hold the finish for a count of two."),
    ("balance_at_finish", 4,
     lambda k: "Finish is off balance - you are falling away after the shot."),
    ("stance_width", 5,
     lambda k: f"Stance width {k['value']} TU is outside the usual range - feet roughly shoulder width apart."),
]

POSITIVE = {
    "head_max_drift_downswing": "Head stays still through the downswing.",
    "head_over_front_foot": "Head is nicely over the front foot at contact.",
    "front_foot_plant_before_impact": "Front foot is down before contact - good sequencing.",
    "balance_at_impact": "Well balanced at contact.",
    "bat_angle_at_impact": "Straight bat at contact.",
    "finish_stillness": "Holds the finish well.",
}


def coaching_feedback(kpis, max_items=4):
    issues = []
    for key, prio, fn in RULES:
        k = kpis.get(key)
        if not k or k["status"] not in ("bad", "warn") or k["value"] is None:
            continue
        sev = 0 if k["status"] == "bad" else 1
        issues.append((sev, prio, key, k["status"], fn(k), k["reliability"]))
    issues.sort()
    cues = [{"kpi": key, "severity": st, "reliability": rel, "cue": msg}
            for _, _, key, st, msg, rel in issues[:max_items]]
    good = [POSITIVE[k] for k in POSITIVE if kpis.get(k, {}).get("status") == "good"]
    return {"work_on": cues, "went_well": good[:3]}
