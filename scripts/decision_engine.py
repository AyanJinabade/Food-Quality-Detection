# ResqFood – Unified Decision Engine

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from infer_raw_efficient import predict as predict_raw
from infer_cooked_efficient import predict as predict_cooked



# RAW FOOD DECISION
def decide_raw(ml_result):
    if (
        ml_result.get("label") == "fresh"
        and ml_result.get("confidence", 0) >= 0.75
    ):
        return "ACCEPT_FOR_DONATION"

    return "REJECT_DONATION"



# COOKED FOOD VISUAL GATE

def cooked_visual_gate(ml_result):
    if ml_result.get("label") == "normal":
        return "CHECK_RULES"

    return "REJECT_DONATION"



# COOKED FOOD RULE ENGINE

def cooked_rules(hours_since_cooked, storage):

    if hours_since_cooked is None or storage is None:
        return "REJECT_DONATION"

    storage = storage.lower()

    if storage == "room" and hours_since_cooked > 4:
        return "REJECT_DONATION"

    if storage == "refrigerated" and hours_since_cooked > 24:
        return "REJECT_DONATION"

    return "ACCEPT_FOR_DONATION"



# UNIFIED DECISION FUNCTION

def unified_decision(
    image_path,
    food_type,
    item_name=None,
    hours_since_cooked=None,
    storage=None,
):

    food_type = food_type.lower()

    try:

        if food_type == "raw":

            ml_result = predict_raw(image_path)
            decision = decide_raw(ml_result)

            return {
                "label": ml_result.get("label"),
                "confidence": ml_result.get("confidence", 0),
                "final_decision": decision,
            }

        elif food_type == "cooked":

            ml_result = predict_cooked(image_path)
            visual_decision = cooked_visual_gate(ml_result)

            if visual_decision == "REJECT_DONATION":
                return {
                    "label": ml_result.get("label"),
                    "confidence": ml_result.get("confidence", 0),
                    "final_decision": "REJECT_DONATION",
                }

            rule_decision = cooked_rules(
                hours_since_cooked,
                storage,
            )

            return {
                "label": ml_result.get("label"),
                "confidence": ml_result.get("confidence", 0),
                "final_decision": rule_decision,
            }

    except Exception as e:
        return {
            "label": "error",
            "confidence": 0.0,
            "final_decision": "REJECT_DONATION",
            "error": str(e),
        }

    return {
        "label": "unknown",
        "confidence": 0.0,
        "final_decision": "REJECT_DONATION",
    }



# Manual Test

if __name__ == "__main__":

    result = unified_decision(
        image_path=r"C:\Users\ilham\Desktop\Ayan\resqfood-food-quality\dataset_cooked\Train\spoiled\IMG-20240914-WA0020 - Copy.jpg",
        food_type="cooked",
        item_name="bread",
        hours_since_cooked=3,
        storage="refrigerated",
    )

    print(result)
