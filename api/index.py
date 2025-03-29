from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

from api.crop_predictor import get_predicted_crop

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)
client = OpenAI()



@app.route("/", methods=["GET"])
def home():
    return "Home"

# ChatGPT API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    """
    Example JSON Input:
    {
        "message": "what vegetables do i grow?",
        "language": "Kanadda"
    }
    Headers Required:
    Content-Type: application/json
    """

    try:
        data = request.json
        user_message = data.get("message", "")
        lang = data.get("language", "")

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # Call OpenAI API with special instructions
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "developer",
                    "content": f"You are a helpful assisstant, designed to help indian farmers, answer their questions. answer in a short way only. answer in {lang} language only",
                },
                {"role": "user", "content": user_message},
            ],
        )

        bot_reply = completion.choices[0].message.content
        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    """
    Example JSON Input:
    {
        "lat": 21,
        "long": 77
    }
    Headers Required:
    Content-Type: application/json
    """

    data = request.json
    lat = data.get("lat")
    long = data.get("long")

    return jsonify({"predicted_crop": get_predicted_crop(lat, long)})


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
