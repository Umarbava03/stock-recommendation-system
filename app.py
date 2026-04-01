import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"


def show_recommendation(label: str, recommendation: str) -> None:
    if recommendation == "BUY":
        st.success(f"{label}: {recommendation}")
    elif recommendation in {"NOT BUY", "NOT_BUY"}:
        st.error(f"{label}: {recommendation}")
    else:
        st.warning(f"{label}: {recommendation}")


st.set_page_config(page_title="Stock Recommendation Dashboard", layout="wide")

st.title("Stock Recommendation Dashboard")
st.write("Fetches predictions from your FastAPI backend.")

symbol = st.text_input("Enter stock symbol", value="AMZN").upper()

if st.button("Predict"):
    try:
        response = requests.get(API_URL, params={"symbol": symbol}, timeout=120)

        if response.status_code != 200:
            st.error(f"API error: {response.text}")
        else:
            data = response.json()

            st.subheader(f"Results for {data['symbol']}")

            classifier = data["classifier_prediction"]
            ml = data["ml_prediction"]
            lstm = data["lstm_prediction"]
            rule = data["rule_based_prediction"]

            st.markdown("## Main Recommendation")
            show_recommendation("Classifier", classifier["recommendation"])
            st.write(f"Confidence: {classifier['confidence_pct']}%")
            st.write("Probabilities:")
            for label, prob in classifier["probabilities"].items():
                st.write(f"- {label}: {prob * 100:.2f}%")
            st.write("Reasons:")
            for reason in classifier["reasons"]:
                st.write(f"- {reason}")

            st.markdown("## Model Comparison")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### ML Regressor")
                show_recommendation("Recommendation", ml["recommendation"])
                st.write(f"Predicted Return: {ml['predicted_return_pct']}%")
                st.write(f"Confidence: {ml['confidence_pct']}%")
                for reason in ml["reasons"]:
                    st.write(f"- {reason}")

            with col2:
                st.markdown("### LSTM")
                show_recommendation("Recommendation", lstm["recommendation"])
                st.write(f"Confidence: {lstm['confidence_pct']}%")
                st.write("Probabilities:")
                for label, prob in lstm["probabilities"].items():
                    st.write(f"- {label}: {prob * 100:.2f}%")
                for reason in lstm["reasons"]:
                    st.write(f"- {reason}")

            with col3:
                st.markdown("### Rule Based")
                show_recommendation("Recommendation", rule["recommendation"])
                st.write(f"Confidence: {rule['confidence_pct']}%")
                for reason in rule["reasons"]:
                    st.write(f"- {reason}")

            with st.expander("Show Features"):
                st.json(data["features"])

    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
