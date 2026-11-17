from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np

app = Flask(__name__)

# Temporary rule-based prediction
# We will replace this with our trained ML model later.

def detect_threat(data):

    risk_score = 0
    reasons = []

    if data["login_after_hours"] == 1:
        risk_score += 20
        reasons.append("Login outside office hours")

    if data["files_downloaded"] > 100:
        risk_score += 30
        reasons.append("Unusual file downloads")

    if data["usb_connected"] == 1:
        risk_score += 15
        reasons.append("USB device connected")

    if data["failed_logins"] >= 5:
        risk_score += 15
        reasons.append("Multiple failed login attempts")

    if data["sensitive_file_access"] >= 10:
        risk_score += 20
        reasons.append("High sensitive file access")

    if risk_score >= 60:
        status = "High Risk"
    elif risk_score >= 30:
        status = "Medium Risk"
    else:
        status = "Low Risk"

    return {
        "risk_score": risk_score,
        "status": status,
        "reasons": reasons
    }


@app.route("/")
def home():
    return """
    <h1>AI Insider Threat Detection System</h1>
    <p>Employee Security Monitoring API</p>
    <p>Use POST /api/predict to analyse employee activity.</p>
    """


@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        required_fields = [
            "login_after_hours",
            "files_downloaded",
            "usb_connected",
            "failed_logins",
            "sensitive_file_access"
        ]

        if not data:
            return jsonify({"error": "JSON input required"}), 400

        missing = [field for field in required_fields if field not in data]

        if missing:
            return jsonify({
                "error": "Missing required fields",
                "fields": missing
            }), 400

        # Validate and convert input
        clean_data = {}

        for field in required_fields:
            value = float(data[field])

            if not np.isfinite(value):
                raise ValueError("Values must be finite")

            if value < 0 or not value.is_integer() if field in [
                "login_after_hours", "files_downloaded",
                "usb_connected", "failed_logins",
                "sensitive_file_access"
            ] else value < 0:
                raise ValueError("Invalid input value")

            clean_data[field] = int(value)

        if clean_data["login_after_hours"] not in [0, 1]:
            raise ValueError("login_after_hours must be 0 or 1")

        if clean_data["usb_connected"] not in [0, 1]:
            raise ValueError("usb_connected must be 0 or 1")

        result = detect_threat(clean_data)

        return jsonify({
            "message": "Employee activity analysed",
            "prediction": result
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Please provide valid numeric values"}), 400

    except Exception:
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
