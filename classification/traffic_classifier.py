import config


class TrafficClassifier:
    def classify_observation(self, obs):
        """
        obs keys:
            time          str  "HH:MM"
            vehicle_count int  number of vehicles in frame
            light_state   str  'red' | 'green' | 'yellow' | 'unknown'
            motion        bool any movement detected
            speed_kmh     float estimated speed (0 if no radar/optical flow)
            objects       list class names from YOLO
            location      str  road name or GPS string
            vehicle_in_intersection bool
        """
        score = 0
        reasons = []

        # --- Congestion ---
        vc = obs.get("vehicle_count", 0)
        if vc >= config.CONGESTION_VEHICLE_COUNT:
            score += 3
            reasons.append(f"congestion ({vc} vehicles)")
        elif vc >= 2:
            score += 1
            reasons.append("moderate traffic")

        # --- Red-light violation ---
        if obs.get("light_state") == "red" and obs.get("vehicle_in_intersection", False):
            score += 4
            reasons.append("red-light violation")

        # --- Speed violation ---
        spd = obs.get("speed_kmh", 0)
        if spd > config.SPEED_LIMIT_KMH * 1.2:
            score += 3
            reasons.append(f"speeding ({spd:.0f} km/h)")
        elif spd > config.SPEED_LIMIT_KMH:
            score += 1
            reasons.append("slightly over speed limit")

        # --- Wrong-way / unexpected object ---
        if "person" in obs.get("objects", []):
            score += 2
            reasons.append("pedestrian on road")

        if obs.get("motion"):
            score += 1

        # --- Final decision ---
        if score >= 6:
            status = "HIGH_RISK"
        elif score >= 4:
            status = "VIOLATION"
        elif score >= 2:
            status = "CONGESTED"
        else:
            status = "NORMAL"

        return {
            "Status": status,
            "Confidence": f"{min(score * 12, 100)}%",
            "Reason": ", ".join(reasons) if reasons else "clear road",
            "VehicleCount": vc,
            "LightState": obs.get("light_state", "unknown"),
        }
