import os
import sys
from typing import Optional, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from infer_raw_efficient import predict as predict_raw
from infer_cooked_efficient import predict as predict_cooked

RAW_CONF_THRESHOLD = 0.75
ROOM_TEMP_LIMIT = 4
REFRIG_LIMIT = 24

def safe_ml_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure ML output is always valid"""
    return {
        "label": result.get("label", "unknown"),
        "confidence": float(result.get("confidence", 0) or 0)
    }

def decide_raw(ml_result: Dict[str, Any]) -> Dict[str, Any]:
    label = ml_result["label"]
    confidence = ml_result["confidence"]

    if label == "fresh" and confidence >= RAW_CONF_THRESHOLD:
        return {"decision": "ACCEPT_FOR_DONATION", "reason": "Fresh & high confidence"}

    return {"decision": "REJECT_DONATION", "reason": "Low freshness or confidence"}

def cooked_visual_gate(ml_result: Dict[str, Any]) -> Dict[str, Any]:
    if ml_result["label"] == "normal":
        return {"status": "PASS", "reason": "Food looks normal"}

    return {"status": "FAIL", "reason": "Food appears spoiled visually"}

def normalize_storage(storage: Optional[str]) -> str:
    if not storage:
        return "unknown"

    storage = storage.lower().strip()

    if "room" in storage:
        return "room"
    if "fridge" in storage or "refrigerated" in storage:
        return "refrigerated"

    return "unknown"

def cooked_rules(hours_since_cooked: Optional[float], storage: Optional[str]) -> Dict[str, Any]:

    if hours_since_cooked is None:
        return {"decision": "REJECT_DONATION", "reason": "Missing cooking time"}

    storage = normalize_storage(storage)

    if storage == "room" and hours_since_cooked > ROOM_TEMP_LIMIT:
        return {"decision": "REJECT_DONATION", "reason": "Too long at room temperature"}

    if storage == "refrigerated" and hours_since_cooked > REFRIG_LIMIT:
        return {"decision": "REJECT_DONATION", "reason": "Too long refrigerated"}

    if storage == "unknown":
        return {"decision": "REJECT_DONATION", "reason": "Unknown storage condition"}

    return {"decision": "ACCEPT_FOR_DONATION", "reason": "Within safe limits"}
def unified_decision(
    image_path: str,
    food_type: str,
    item_name: Optional[str] = None,
    hours_since_cooked: Optional[float] = None,
    storage: Optional[str] = None,
) -> Dict[str, Any]:

    if not os.path.exists(image_path):
        return {
            "label": "error",
            "confidence": 0.0,
            "final_decision": "REJECT_DONATION",
            "reason": "Image file not found"
        }

    food_type = str(food_type).lower().strip()

    try:
        if food_type == "raw":
            ml_result = safe_ml_output(predict_raw(image_path))
            decision_data = decide_raw(ml_result)

        elif food_type == "cooked":
            ml_result = safe_ml_output(predict_cooked(image_path))

            visual = cooked_visual_gate(ml_result)

            if visual["status"] == "FAIL":
                return {
                    "label": ml_result["label"],
                    "confidence": ml_result["confidence"],
                    "final_decision": "REJECT_DONATION",
                    "reason": visual["reason"]
                }

            decision_data = cooked_rules(hours_since_cooked, storage)

        else:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "final_decision": "REJECT_DONATION",
                "reason": "Invalid food_type"
            }

        return {
            "label": ml_result["label"],
            "confidence": ml_result["confidence"],
            "final_decision": decision_data["decision"],
            "reason": decision_data["reason"]
        }

    except Exception as e:
        return {
            "label": "error",
            "confidence": 0.0,
            "final_decision": "REJECT_DONATION",
            "reason": str(e)
        }

if __name__ == "__main__":

    result = unified_decision(
        image_path=r"C:\Users\ilham\Desktop\Ayan\resqfood-food-quality\dataset_cooked\Train\spoiled\IMG.jpg",
        food_type="cooked",
        item_name="bread",
        hours_since_cooked=3,
        storage="Refrigerated"
    )

    print(result)
