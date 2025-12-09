import gradio as gr
import numpy as np
import tensorflow as tf
import mne
import tempfile
import os
import plotly.graph_objects as go

MODEL_PATHS = {
    "Main Model": "best_model.h5",
    "Backup Model 1": "model_v2.h5",
    "Backup Model 2": "model_v3.h5"
}

# Load models
MODELS = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        MODELS[name] = tf.keras.models.load_model(path)

def preprocess_eeg(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        raw = mne.io.read_raw_edf(temp_path, preload=True)
        raw.filter(0.5, 40)
        data = raw.get_data()

        n_channels = 23
        n_samples = 4096

        segment = data[:n_channels, :n_samples]

        segment = (segment - np.mean(segment)) / np.std(segment)
        segment = np.expand_dims(segment, axis=0)

        os.remove(temp_path)
        return segment, data[:1, :1000]  # raw snippet for plotting

    except Exception as e:
        return None, f"Error: {e}"

def make_plot(eeg_signal):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=eeg_signal.flatten(),
        mode="lines",
        name="EEG Signal"
    ))

    fig.update_layout(
        title="EEG Signal Preview",
        xaxis_title="Time",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=300
    )
    return fig

def predict(file, model_name):
    eeg, snippet = preprocess_eeg(file)

    if eeg is None:
        return snippet, None, None

    model = MODELS[model_name]
    prob = model.predict(eeg)[0][0]

    if prob > 0.5:
        result = "⚠️ **Seizure Detected**"
    else:
        result = "✅ **No Seizure Detected**"

    fig = make_plot(snippet)

    return result, prob, fig


# ------------------------- UI DESIGN -----------------------------

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as iface:

    gr.Markdown(
        """
        <h1 style='text-align:center; color:#4CAF50;'>EEG Seizure Prediction Dashboard</h1>
        <p style='text-align:center;'>Upload an EEG (.edf) file — AI predicts seizure probability & visualizes EEG</p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            eeg_file = gr.File(label="Upload EEG (.edf)", file_types=[".edf"])
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Main Model",
                label="Select Model"
            )
            btn = gr.Button("🔍 Predict", variant="primary")

        with gr.Column(scale=1):
            result_box = gr.Markdown("### Prediction result will appear here")

            gauge = gr.Number(
                label="Seizure Probability",
                precision=4
            )

    plot = gr.Plot(label="EEG Graph")

    btn.click(
        predict,
        inputs=[eeg_file, model_choice],
        outputs=[result_box, gauge, plot]
    )

iface.launch()
