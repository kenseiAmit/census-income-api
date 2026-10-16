"""Script to send a live POST request to the deployed FastAPI application."""

import argparse
import sys
import requests

DEFAULT_URL = "https://census-income-api-3hq5.onrender.com/predict"

SAMPLE_PAYLOAD = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 14084,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}


def test_live_api(url: str):
    """Send a POST request to the API and print status code and inference."""
    print(f"Sending POST request to: {url}")
    print("Payload:")
    for k, v in SAMPLE_PAYLOAD.items():
        print(f"  {k}: {v}")

    try:
        response = requests.post(url, json=SAMPLE_PAYLOAD, timeout=15)
        print("\nResponse Received:")
        print(f"HTTP Status Code: {response.status_code}")
        print(f"Inference Result: {response.json()}")
        return response.status_code
    except requests.exceptions.RequestException as err:
        print(f"\nRequest failed: {err}", file=sys.stderr)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query live inference API endpoint."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Full URL to the prediction endpoint (default: %(default)s)",
    )
    args = parser.parse_args()
    test_live_api(args.url)
