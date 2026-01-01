import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Deep Vision Crowd Monitor",
    layout="wide"
)

st.title("👁️ Deep Vision Crowd Monitor")
st.markdown(
    """
    **AI-powered Crowd Density Estimation and Overcrowding Alert System**

    This dashboard analyzes crowd images and videos using deep learning,
    generates density heatmaps, estimates crowd count, and sends alerts
    when overcrowding is detected.
    """
)

# --------------------------------------------------
# LOAD MODEL (SIMPLE CSRNET-STYLE PLACEHOLDER)
# --------------------------------------------------
class CSRNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

model = CSRNet()
model.load_state_dict(torch.load("model50.pth", map_location="cpu"), strict=False)
model.eval()

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
THRESHOLD = 300
SCALE_FACTOR = 0.001  # Calibrates density → count

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def predict_density(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        density = model(img).squeeze().numpy()

    density[density < 0] = 0
    return density


def create_heatmap(frame, density):
    d = density - density.min()
    d = (d / (d.max() + 1e-6) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay


def send_email_alert(count):
    sender = "yourmail@gmail.com"
    receiver = "receiver@gmail.com"
    password = "YOUR_16_DIGIT_APP_PASSWORD"

    subject = "🚨 Overcrowding Alert – Deep Vision System"
    body = f"""
Dear Authority,

An overcrowding situation has been detected.

Estimated Crowd Count : {count}
Status                : Threshold Exceeded

Immediate action is advised.

Regards,
Deep Vision Crowd Monitoring System
"""

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
    except:
        pass


# --------------------------------------------------
# SIDEBAR INPUT
# --------------------------------------------------
st.sidebar.header("📂 Input Selection")
mode = st.sidebar.radio("Choose Input Type", ["Image", "Video"])

# --------------------------------------------------
# IMAGE MODE
# --------------------------------------------------
if mode == "Image":
    uploaded_image = st.file_uploader("📸 Upload Crowd Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        frame = np.array(img)

        density = predict_density(frame)
        count = int(density.sum() * SCALE_FACTOR)
        overlay = create_heatmap(frame, density)

        st.subheader("📊 Density Heatmap Overlay")
        st.image(overlay, channels="BGR", use_container_width=True)

        st.subheader(f"👥 Estimated Crowd Count: **{count}**")

        if count > THRESHOLD:
            st.error("🚨 OVER-CROWD ALERT! Immediate action required.")
            send_email_alert(count)
        else:
            st.success("✅ Crowd level is within safe limits.")

# --------------------------------------------------
# VIDEO MODE
# --------------------------------------------------
if mode == "Video":
    uploaded_video = st.file_uploader("🎥 Upload Crowd Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)

        frame_placeholder = st.empty()
        count_placeholder = st.empty()
        alert_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            density = predict_density(frame)
            count = int(density.sum() * SCALE_FACTOR)
            overlay = create_heatmap(frame, density)

            frame_placeholder.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            count_placeholder.markdown(
                f"### 👥 Estimated Crowd Count: **{count}**"
            )

            if count > THRESHOLD:
                alert_placeholder.error("🚨 OVER-CROWD ALERT! Immediate action required.")
                send_email_alert(count)
            else:
                alert_placeholder.success("✅ Crowd level is within safe limits.")

        cap.release()
        os.unlink(temp_file.name)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "🔐 **Deep Vision Crowd Monitor** | AI-based Safety & Surveillance System"
)