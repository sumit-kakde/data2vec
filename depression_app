import streamlit as st
import torch
import torchaudio
from transformers import AutoProcessor, Data2VecAudioForSequenceClassification
import numpy as np
import os
from tempfile import NamedTemporaryFile
import soundfile as sf

# ========== CONSTANTS & CONFIG ==========
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10 * SAMPLE_RATE
MODEL_PATHS = {
    "multi": "C:/Users/sumit/Desktop/deep/data2vec_multi_finetuned.pth",
    "binary": "C:/Users/sumit/Desktop/deep/data2vec_binary_finetuned.pth"
}

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .main { background-color: #f8f9fa; }
    .header {
        background: linear-gradient(135deg, #2e7d32, #4CAF50);
        padding: 2.5rem;
        border-radius: 15px;
        animation: fadeIn 1s ease-in-out;
    }
    .header h1 {
        color: white;
        text-align: center;
        font-weight: 700;
        margin: 0;
        animation: pulse 2s infinite;
    }
    .upload-container {
        border: 2px dashed #2e7d32;
        padding: 1.875rem;
        border-radius: 15px;
        transition: all 0.3s;
    }
    .upload-container:hover {
        border-color: #4CAF50;
        background-color: rgba(76, 175, 80, 0.05);
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1.25rem 0;
        animation: slideUp 0.5s ease-out;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        transition: all 0.3s;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .analyze-btn, .report-btn {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.75rem 1.875rem !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
        margin-top: 20px !important;
    }
    .report-btn {
        background-color: #2196F3 !important;
    }
    .analyze-btn:hover, .report-btn:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
    }
    .disclaimer {
        text-align: center;
        padding: 1rem;
        color: #666;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# ========== MODEL LOADING ==========
@st.cache_resource
def load_models():
    try:
        processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base")
        models = {}
        for model_type in ["multi", "binary"]:
            model = Data2VecAudioForSequenceClassification.from_pretrained(
                "facebook/data2vec-audio-base", num_labels=4 if model_type == "multi" else 2
            )
            if os.path.exists(MODEL_PATHS[model_type]):
                model.load_state_dict(torch.load(MODEL_PATHS[model_type], map_location=torch.device('cpu')))
                model.eval()
                models[model_type] = model
            else:
                st.error(f"{model_type.capitalize()} model file not found")
                return None, None, None
        return models["multi"], models["binary"], processor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# ========== AUDIO PROCESSING ==========
def process_audio(audio_path):
    try:
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except:
            data, sample_rate = sf.read(audio_path)
            waveform = torch.from_numpy(data).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        waveform = waveform[:, :MAX_AUDIO_LENGTH]
        if waveform.shape[1] < MAX_AUDIO_LENGTH:
            padding = torch.zeros((1, MAX_AUDIO_LENGTH - waveform.shape[1]))
            waveform = torch.cat((waveform, padding), dim=1)
        return waveform
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return None

# ========== PREDICTION FUNCTION ==========
def predict(audio_path, model_type="multi"):
    waveform = process_audio(audio_path)
    if waveform is None:
        return None, None, None
    try:
        input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
        with torch.no_grad():
            model = model_multi if model_type == "multi" else model_binary
            outputs = model(input_values)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            labels = (["None", "Mild", "Moderate", "Severe"] if model_type == "multi" else ["Not Depressed", "Depressed"])
            return labels[prediction], probabilities.squeeze().numpy(), labels
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# ========== REPORT GENERATION ==========
def generate_report(result, probs, labels, analysis_type):
    html = f"""
    <html><head><meta charset="utf-8"><title>Depression Report</title></head><body>
    <h2 style='color: #2e7d32;'>Depression Analysis Report</h2>
    <hr>
    <h3>Diagnosis: <span style='color: {"#4CAF50" if "Not Depressed" in result else "#ef5350"}'>{result}</span></h3>
    <h4>Analysis Type: {analysis_type}</h4>
    <h3>Confidence Levels:</h3>
    <ul>
    """
    for label, prob in zip(labels, probs):
        html += f"<li>{label}: <strong>{prob:.1%}</strong></li>"
    html += """
    </ul><hr>
    <p style='font-size: 0.8rem; color: #666;'>📅 Disclaimer: This tool provides preliminary analysis and should not replace professional medical advice.</p>
    </body></html>
    """
    return html.encode('utf-8')

# ========== UI COMPONENTS ==========
def render_header():
    st.markdown("""<div class="header"><h1>Depression Severity Detection</h1></div>""", unsafe_allow_html=True)

def render_upload_section():
    st.markdown("<br>", unsafe_allow_html=True)
    return st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"], key="audio_uploader")

def render_analysis_section(temp_path):
    st.markdown("### 🔊 Audio Preview")
    st.audio(temp_path)
    st.markdown("### ⚙️ Analysis Settings")
    analysis_type = st.radio("Select analysis type:", ("Detailed Severity (4 classes)", "Binary Classification (Depressed/Not)"), horizontal=True)
    if st.button("🔍 Analyze Audio", key="analyze_btn", help="Click to analyze the uploaded audio"):
        return analysis_type
    return None

def render_results(result, probs, labels, analysis_type):
    st.markdown("### 📊 Analysis Results")
    result_color = "#4CAF50" if "Not Depressed" in result else "#ef5350"
    st.markdown(f"""<div class="result-card" style="border-left: 5px solid {result_color};">
        <h3 style="margin:0; color: {result_color};">Diagnosis: {result}</h3></div>""", unsafe_allow_html=True)
    st.markdown("#### 📈 Confidence Levels")
    cols = st.columns(len(labels))
    for i, (label, prob) in enumerate(zip(labels, probs)):
        with cols[i]:
            st.markdown(f"""<div class="metric-box"><div style="font-size:14px; color:#666;">{label}</div>
                <div style="font-size:24px; font-weight:bold; color:{result_color};">{prob:.1%}</div></div>""", unsafe_allow_html=True)
    st.markdown("#### 📉 Confidence Distribution")
    st.bar_chart({label: float(prob) for label, prob in zip(labels, probs)}, color=result_color)

# ========== MAIN APP ==========
def main():
    global model_multi, model_binary, processor
    model_multi, model_binary, processor = load_models()
    render_header()
    uploaded_file = render_upload_section()

    if uploaded_file is not None:
        with NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1].lower(), delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        analysis_type = render_analysis_section(temp_path)
        if analysis_type and (model_multi and model_binary):
            with st.spinner("🔬 Analyzing audio patterns..."):
                try:
                    is_binary = "Binary" in analysis_type
                    result, probs, labels = predict(temp_path, "binary" if is_binary else "multi")
                    if result is not None:
                        st.session_state["result"] = result
                        st.session_state["probs"] = probs
                        st.session_state["labels"] = labels
                        st.session_state["analysis_type"] = analysis_type
                        render_results(result, probs, labels, analysis_type)
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

    # Download report instead of showing it
    if "result" in st.session_state:
        report_bytes = generate_report(
            st.session_state["result"],
            st.session_state["probs"],
            st.session_state["labels"],
            st.session_state["analysis_type"]
        )
        st.download_button(
            label="📄 Download Report",
            data=report_bytes,
            file_name="depression_report.html",
            mime="text/html",
            key="download_report_btn",
            help="Click to download the report"
        )

    st.markdown("""<div class="disclaimer"><hr>📅 Disclaimer: This tool provides preliminary analysis and should not replace professional medical advice.</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
