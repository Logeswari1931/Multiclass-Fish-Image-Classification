import streamlit as st
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import load_model
import pandas as pd
import os
import altair as alt
from collections import Counter

st.set_page_config(page_title="Fish Image Classification", layout="wide")
st.title(" Multiclass Fish Image Classification")

#Select Models
models_info = {
    "Custom CNN": "C:/Multiclass Fish Image Classification/Custom CNN/custom_cnn_model.keras",
    "VGG16": "C:/Multiclass Fish Image Classification/VGG16_final.keras",
    "ResNet50": "C:/Multiclass Fish Image Classification/ResNet50_final.keras",
    "MobileNet": "C:/Multiclass Fish Image Classification/MobileNet_final.keras",
    "InceptionV3": "C:/Multiclass Fish Image Classification/InceptionV3_final.keras",
    "EfficientNetV2B0": "C:/Multiclass Fish Image Classification/EfficientNetV2B0_final.keras"
}

selected_models = st.sidebar.multiselect(
    "Select Model(s) to Use for Prediction",
    options=list(models_info.keys()),
    default=list(models_info.keys())
)

# Upload Image(s)
uploaded_files = st.file_uploader(
    "Upload Your Image(s) (JPG only)", 
    type="jpg", 
    accept_multiple_files=True
)

# Load labels from JSON
with open('class_labels.json', 'r') as f:
    labels = json.load(f)
if isinstance(labels, dict):
    labels = [labels[str(i)] for i in range(len(labels))]

# Predict Button
if uploaded_files and st.button("Classify Selected Model(s)"):
    all_results = []

    for file in uploaded_files:
        image_obj = Image.open(file)
        st.image(image_obj, caption=f"Uploaded Image: {file.name}", width=250)

        # Preprocess Image
        img = image_obj.convert("RGB").resize((224, 224))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.markdown(f"#### Predictions for {file.name}")
        image_results = []
        model_top1_labels = []

        cols = st.columns(len(selected_models))
        for i, model_name in enumerate(selected_models):
            model_path = models_info[model_name]

            if os.path.exists(model_path):
                model = load_model(model_path)
                preds = model.predict(img_array)[0]
                top_k = 3
                top_idx = preds.argsort()[-top_k:][::-1]

                # Top prediction
                top1_label = labels[top_idx[0]]
                model_top1_labels.append(top1_label)
                top1_prob = preds[top_idx[0]]

                # Store results for table
                image_results.append({
                    "Model": model_name,
                    "Image": file.name,
                    "Top-1 Class": top1_label,
                    "Confidence": f"{top1_prob:.2%}"
                })

                # Display in column
                with cols[i]:
                    st.markdown(f"**{model_name}**")
                    st.write(f"Top-1: {top1_label} — {top1_prob:.2%}")

                    # Interactive Top-3 Bar Chart using Altair
                    top3_data = pd.DataFrame({
                        "Class": [labels[idx] for idx in top_idx],
                        "Probability": [preds[idx] for idx in top_idx]
                    })

                    chart = alt.Chart(top3_data).mark_bar(color='skyblue').encode(
                        x='Class',
                        y='Probability',
                        tooltip=['Class', alt.Tooltip('Probability', format='.2%')]
                    ).properties(
                        width=150, height=150
                    )

                    st.altair_chart(chart, use_container_width=True)
            else:
                st.warning(f"{model_name} model file not found!")

    
        # Top Prediction Across All Models
        if model_top1_labels:
            counter = Counter(model_top1_labels)
            top_prediction = counter.most_common(1)[0]
            st.subheader("🏆 Top Prediction Across All Models")
            st.write(f"**{top_prediction[0]}** predicted by {top_prediction[1]} / {len(model_top1_labels)} models")

        all_results.extend(image_results)


    # Comparison Table & Download CSV
    
    if all_results:
        st.subheader("📊 Batch Prediction Results")
        results_df = pd.DataFrame(all_results)
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Batch Prediction CSV",
            data=csv,
            file_name="fish_batch_predictions.csv",
            mime="text/csv"
        )