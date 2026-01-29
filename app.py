import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

# ================= PAGE =================
st.set_page_config(page_title="Traffic AI Pro", page_icon="🚦", layout="wide")

# ================= UI =================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#0f172a,#111827); }
h1,h2,h3 { color:white !important; }
.card {
 background: rgba(255,255,255,0.1);
 border-radius:15px;
 padding:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>🚦 Traffic AI PRO System</h1>", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_all():
    model = load_model("traffic_lstm_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_all()

# ================= PREDICT =================
def predict(vehicle, speed, hour):
    data = np.array([[vehicle, speed, hour, 0]])
    scaled = scaler.transform(data)
    X = scaled[:, :-1].reshape(1,1,3)
    pred = model.predict(X, verbose=0)
    return float(pred[0][0])

# ================= SMART AI =================
def smart_ai(msg):
    msg = msg.lower()

    if "traffic" in msg:
        return "Traffic depends on vehicle density, road capacity and peak hour demand."

    elif "best time" in msg:
        return "Best travel time is early morning or late night."

    elif "congestion" in msg:
        return "Congestion happens when vehicles exceed road capacity."

    elif "model" in msg:
        return "We use LSTM Deep Learning for traffic pattern prediction."

    elif "hello" in msg or "hi" in msg:
        return "Hello! Ask me about traffic prediction, routes or congestion."

    return "Ask me about traffic, congestion, routes or prediction."

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "🚦 Predictor",
    "📊 Analytics",
    "🗺 Map",
    "🤖 AI Assistant"
])

# ================= TAB 1 =================
with tab1:

    col1, col2 = st.columns([1,2])

    with col1:
        vehicle = st.slider("Vehicle Count", 0, 300, 120)
        speed = st.slider("Average Speed", 1, 100, 30)
        hour = st.slider("Hour", 0, 23, 18)

        if st.button("Predict Traffic"):

            result = predict(vehicle, speed, hour)

            if result < 0.3:
                level = "LOW 🟢"
                advice = "Traffic smooth. Travel now."
            elif result < 0.6:
                level = "MEDIUM 🟡"
                advice = "Moderate traffic. Leave early."
            else:
                level = "HIGH 🔴"
                advice = "Heavy traffic. Avoid peak hours."

            st.session_state.history.append({
                "Vehicle": vehicle,
                "Speed": speed,
                "Hour": hour,
                "Score": round(result,3),
                "Level": level
            })

            st.session_state.last_result = (result, level, advice)

    with col2:

        if "last_result" in st.session_state:
            result, level, advice = st.session_state.last_result

            st.markdown(f"""
            <div class="card">
            <h2>Congestion Level: {level}</h2>
            <h3>Score: {result:.3f}</h3>
            <p>{advice}</p>
            </div>
            """, unsafe_allow_html=True)

            # Report
            report = f"""
Traffic Report
Vehicle: {vehicle}
Speed: {speed}
Hour: {hour}
Score: {result}
Level: {level}
"""
            st.download_button("Download Report", report)

# ================= TAB 2 =================
with tab2:

    st.subheader("Prediction History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        st.subheader("Trend Graph")

        fig, ax = plt.subplots()
        ax.plot(df["Hour"], df["Score"], marker='o')
        ax.set_xlabel("Hour")
        ax.set_ylabel("Congestion Score")
        st.pyplot(fig)

# ================= TAB 3 =================
with tab3:

    place = st.text_input("Search Location", "Mangalore, Karnataka")
    map_url = f"https://www.google.com/maps?q={place}&output=embed"
    components.iframe(map_url, height=500)

    st.subheader("Route Distance Estimator")

    dist = st.slider("Distance (KM)", 1, 100, 10)
    avg_speed = st.slider("Avg Speed (KM/H)", 20, 100, 40)

    time = dist / avg_speed * 60
    st.info(f"Estimated Travel Time: {round(time,1)} minutes")

# ================= TAB 4 =================
with tab4:

    st.subheader("AI Traffic Assistant")

    msg = st.text_input("Ask traffic question")

    if st.button("Send"):
        if msg:
            reply = smart_ai(msg)
            st.session_state.chat.append(("You", msg))
            st.session_state.chat.append(("AI", reply))

    for s, m in st.session_state.chat:
        st.write(f"**{s}:** {m}")
