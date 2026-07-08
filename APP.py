import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ExifTags
import io
import base64
import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Cats vs Dogs AI Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'cat_count' not in st.session_state:
    st.session_state.cat_count = 0
if 'dog_count' not in st.session_state:
    st.session_state.dog_count = 0

# ============================================================
# CUSTOM CSS - PROFESSIONAL DARK THEME
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #1e3a5f 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .prediction-box {
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }

    .prediction-box:hover {
        transform: translateY(-5px);
    }

    .cat-pred {
        background: linear-gradient(135deg, #ea580c, #f97316);
        color: white;
    }

    .dog-pred {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 5px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }

    .history-item {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.2s ease;
    }

    .history-item:hover {
        background: rgba(30, 41, 59, 0.9);
        border-color: rgba(255,255,255,0.15);
    }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        overflow: hidden;
        margin-top: 5px;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease;
    }

    .upload-zone {
        border: 2px dashed rgba(96, 165, 250, 0.4);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        background: rgba(30, 41, 59, 0.4);
        transition: all 0.3s ease;
    }

    .upload-zone:hover {
        border-color: rgba(96, 165, 250, 0.8);
        background: rgba(30, 41, 59, 0.6);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
    }

    .exif-data {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 8px;
        padding: 12px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    .comparison-container {
        display: flex;
        gap: 20px;
        align-items: center;
        justify-content: center;
    }

    .vs-badge {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 15px 20px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #3b82f6;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
    <div class="main-header">
        <h1>🐱🐶 AI Image Classifier</h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 10px;">
            Advanced Deep Learning Model for Cat & Dog Classification
        </p>
        <div style="margin-top: 15px;">
            <span style="background: rgba(96, 165, 250, 0.2); color: #60a5fa; padding: 5px 15px; border-radius: 20px; font-size: 0.85rem;">
                🧠 CNN Architecture
            </span>
            <span style="background: rgba(139, 92, 246, 0.2); color: #a78bfa; padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; margin-left: 10px;">
                📊 25K Images Trained
            </span>
            <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; margin-left: 10px;">
                ⚡ Real-time Prediction
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with st.spinner("🔄 Loading AI Model..."):
            model = tf.keras.models.load_model("cats_vs_dogs_cnn.keras")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'cats_vs_dogs_cnn.keras' is in the same folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, img_array):
    """Make prediction and return results"""
    prediction = model.predict(img_array, verbose=0)
    pred_value = float(prediction[0][0])
    cat_prob = 1 - pred_value
    dog_prob = pred_value

    if dog_prob >= 0.5:
        predicted_class = "Dog"
        confidence = dog_prob * 100
        emoji = "🐶"
        color = "#2563eb"
    else:
        predicted_class = "Cat"
        confidence = cat_prob * 100
        emoji = "🐱"
        color = "#ea580c"

    return {
        'class': predicted_class,
        'confidence': confidence,
        'cat_prob': cat_prob * 100,
        'dog_prob': dog_prob * 100,
        'raw_value': pred_value,
        'emoji': emoji,
        'color': color
    }

def get_image_info(image):
    """Extract image metadata"""
    info = {
        'Format': image.format if image.format else 'Unknown',
        'Mode': image.mode,
        'Size': f"{image.size[0]} x {image.size[1]} px",
        'Width': image.size[0],
        'Height': image.size[1],
        'Aspect Ratio': round(image.size[0] / image.size[1], 2) if image.size[1] > 0 else 0,
    }

    # Try to get EXIF data
    try:
        exif = image._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ['DateTime', 'Make', 'Model', 'FNumber', 'ExposureTime', 'ISOSpeedRatings']:
                    info[tag] = str(value)
    except:
        pass

    return info

def add_to_history(result, image):
    """Add prediction to history"""
    history_item = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'class': result['class'],
        'confidence': result['confidence'],
        'cat_prob': result['cat_prob'],
        'dog_prob': result['dog_prob'],
        'image': image
    }
    st.session_state.history.insert(0, history_item)
    st.session_state.total_predictions += 1
    if result['class'] == 'Cat':
        st.session_state.cat_count += 1
    else:
        st.session_state.dog_count += 1

def create_confidence_gauge(confidence, color):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.2)",
            'steps': [
                {'range': [0, 50], 'color': "rgba(255,255,255,0.05)"},
                {'range': [50, 80], 'color': "rgba(255,255,255,0.05)"},
                {'range': [80, 100], 'color': "rgba(255,255,255,0.05)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': confidence
            }
        }
    ))
    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def create_history_chart():
    """Create prediction history chart"""
    if not st.session_state.history:
        return None

    df = pd.DataFrame(st.session_state.history[:20])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    fig = px.scatter(df, x='timestamp', y='confidence', 
                     color='class',
                     color_discrete_map={'Cat': '#ea580c', 'Dog': '#2563eb'},
                     title='Prediction Confidence Over Time',
                     labels={'confidence': 'Confidence (%)', 'timestamp': 'Time'})
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        title_font_color='white',
        legend_title_font_color='white'
    )
    return fig

def create_distribution_chart():
    """Create class distribution pie chart"""
    if st.session_state.total_predictions == 0:
        return None

    labels = ['Cats', 'Dogs']
    values = [st.session_state.cat_count, st.session_state.dog_count]
    colors = ['#ea580c', '#2563eb']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=14
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        annotations=[dict(text=f'Total\n{st.session_state.total_predictions}', 
                         x=0.5, y=0.5, font_size=16, showarrow=False, font_color='white')]
    )
    return fig

def export_results():
    """Export prediction history to JSON"""
    export_data = []
    for item in st.session_state.history:
        export_data.append({
            'timestamp': item['timestamp'],
            'predicted_class': item['class'],
            'confidence': f"{item['confidence']:.2f}%",
            'cat_probability': f"{item['cat_prob']:.2f}%",
            'dog_probability': f"{item['dog_prob']:.2f}%"
        })
    return json.dumps(export_data, indent=2)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #60a5fa; margin-bottom: 5px;">🐾 AI Classifier</h2>
            <p style="color: #64748b; font-size: 0.9rem;">Professional Edition</p>
        </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("<div class='section-title'>📍 Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["🏠 Single Image", "⚔️ Compare Images", "📊 Analytics", "📜 History"], label_visibility="collapsed")

    st.divider()

    # Model Info
    st.markdown("<div class='section-title'>🧠 Model Info</div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background: rgba(30,41,59,0.6); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
            <p>🎯 <b>Classes:</b> Cat | Dog</p>
            <p>📚 <b>Dataset:</b> 25,000 images</p>
            <p>🏗️ <b>Architecture:</b> CNN</p>
            <p>📈 <b>Split:</b> 80/10/10</p>
            <p>📐 <b>Input:</b> 224×224 px</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Settings
    st.markdown("<div class='section-title'>⚙️ Settings</div>", unsafe_allow_html=True)
    show_debug = st.checkbox("🔍 Debug Mode", value=False)
    show_metadata = st.checkbox("📋 Show Metadata", value=True)
    enable_history = st.checkbox("💾 Save to History", value=True)

    st.divider()

    # Stats
    st.markdown("<div class='section-title'>📈 Quick Stats</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_predictions}</div>
                <div class="metric-label">Predictions</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        accuracy = 0
        if st.session_state.total_predictions > 0:
            accuracy = max(st.session_state.cat_count, st.session_state.dog_count) / st.session_state.total_predictions * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_predictions}</div>
                <div class="metric-label">Total</div>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT - SINGLE IMAGE PAGE
# ============================================================
if page == "🏠 Single Image":
    # Upload Section
    st.markdown("<div class='section-title'>📤 Upload Image</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff'],
        help="Supported formats: JPG, PNG, BMP, WEBP, TIFF"
    )

    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file)
        original_image = image.copy()

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Detailed View", "🖼️ Original"])

        with tab1:
            col1, col2 = st.columns([1, 1.2])

            with col1:
                st.subheader("📷 Uploaded Image")
                st.image(image, use_container_width=True, caption="Your Image")

                if show_metadata:
                    st.markdown("<div class='section-title'>📋 Image Metadata</div>", unsafe_allow_html=True)
                    info = get_image_info(image)
                    info_df = pd.DataFrame(list(info.items()), columns=['Property', 'Value'])
                    st.dataframe(info_df, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("🎯 Prediction Result")

                if model is not None:
                    # Preprocess and predict
                    img_array = preprocess_image(image)

                    if show_debug:
                        with st.expander("🔍 Preprocessing Debug"):
                            st.write(f"**Shape:** {img_array.shape}")
                            st.write(f"**Value Range:** [{img_array.min():.3f}, {img_array.max():.3f}]")
                            st.write(f"**Mean:** {img_array.mean():.3f}")
                            st.write(f"**Std:** {img_array.std():.3f}")

                    # Prediction with animation
                    with st.spinner("🧠 Analyzing with Neural Network..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        result = predict_image(model, img_array)
                        progress_bar.empty()

                    # Display result
                    pred_class = result['class']
                    confidence = result['confidence']
                    emoji = result['emoji']
                    color = result['color']

                    # Main prediction box
                    st.markdown(f"""
                        <div class="prediction-box {'cat-pred' if pred_class == 'Cat' else 'dog-pred'}">
                            <div style="font-size: 4rem; margin-bottom: 10px;">{emoji}</div>
                            <h1 style="margin: 0; font-size: 2.5rem;">{pred_class}</h1>
                            <h3 style="margin: 10px 0; opacity: 0.9;">Confidence: {confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    # Confidence Gauge
                    st.plotly_chart(create_confidence_gauge(confidence, color), use_container_width=True)

                    # Confidence Breakdown
                    st.markdown("<div class='section-title'>📊 Confidence Breakdown</div>", unsafe_allow_html=True)

                    col_cat, col_dog = st.columns(2)
                    with col_cat:
                        st.metric("🐱 Cat", f"{result['cat_prob']:.2f}%")
                        st.progress(result['cat_prob'] / 100, text=f"{result['cat_prob']:.1f}%")
                    with col_dog:
                        st.metric("🐶 Dog", f"{result['dog_prob']:.2f}%")
                        st.progress(result['dog_prob'] / 100, text=f"{result['dog_prob']:.1f}%")

                    # Add to history
                    if enable_history:
                        add_to_history(result, original_image)
                        st.success("✅ Saved to history!")

                    # Debug info
                    if show_debug:
                        with st.expander("🔍 Raw Prediction Data"):
                            st.json({
                                'raw_value': result['raw_value'],
                                'cat_probability': result['cat_prob'],
                                'dog_probability': result['dog_prob'],
                                'predicted_class': result['class'],
                                'confidence': result['confidence']
                            })

                            if st.button("📋 Show Model Architecture"):
                                summary = []
                                model.summary(print_fn=lambda x: summary.append(x))
                                st.code("\n".join(summary))
                else:
                    st.error("⚠️ Model not loaded. Please check the model file.")

        with tab2:
            if model is not None:
                img_array = preprocess_image(image)
                result = predict_image(model, img_array)

                # Create comparison bars
                categories = ['Cat', 'Dog']
                probabilities = [result['cat_prob'], result['dog_prob']]
                colors_chart = ['#ea580c', '#2563eb']

                fig = go.Figure(data=[
                    go.Bar(
                        x=categories,
                        y=probabilities,
                        marker_color=colors_chart,
                        text=[f'{p:.2f}%' for p in probabilities],
                        textposition='auto',
                        textfont=dict(size=16, color='white')
                    )
                ])
                fig.update_layout(
                    title='Probability Distribution',
                    yaxis_title='Probability (%)',
                    yaxis_range=[0, 100],
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title_font_color='white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Image statistics
                st.markdown("<div class='section-title'>🎨 Image Statistics</div>", unsafe_allow_html=True)
                img_array_stats = np.array(image)

                col_r, col_g, col_b = st.columns(3)
                with col_r:
                    st.metric("🔴 Red Mean", f"{img_array_stats[:,:,0].mean():.1f}")
                with col_g:
                    st.metric("🟢 Green Mean", f"{img_array_stats[:,:,1].mean():.1f}")
                with col_b:
                    st.metric("🔵 Blue Mean", f"{img_array_stats[:,:,2].mean():.1f}")

                # Histogram
                fig_hist = go.Figure()
                for i, (color_name, color_val) in enumerate(zip(['Red', 'Green', 'Blue'], ['#ef4444', '#22c55e', '#3b82f6'])):
                    fig_hist.add_trace(go.Histogram(
                        x=img_array_stats[:,:,i].flatten(),
                        name=color_name,
                        marker_color=color_val,
                        opacity=0.7,
                        nbinsx=50
                    ))
                fig_hist.update_layout(
                    title='Color Distribution Histogram',
                    xaxis_title='Pixel Value',
                    yaxis_title='Frequency',
                    barmode='overlay',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title_font_color='white',
                    height=350
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        with tab3:
            st.image(image, use_container_width=True)

            # Download button
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            byte_im = buf.getvalue()

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="📥 Download Image",
                    data=byte_im,
                    file_name=f"uploaded_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            with col_d2:
                if model is not None:
                    img_array = preprocess_image(image)
                    result = predict_image(model, img_array)
                    report = f"""
Classification Report
=====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Predicted Class: {result['class']} {result['emoji']}
Confidence: {result['confidence']:.2f}%
Cat Probability: {result['cat_prob']:.2f}%
Dog Probability: {result['dog_prob']:.2f}%
Image Size: {image.size[0]} x {image.size[1]} px
                    """
                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name=f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    else:
        # Empty state
        st.markdown("""
            <div style="text-align: center; padding: 80px 20px; background: rgba(30,41,59,0.4); border-radius: 20px; border: 2px dashed rgba(96,165,250,0.3); margin-top: 20px;">
                <div style="font-size: 80px; margin-bottom: 20px;">📤</div>
                <h2 style="color: #e2e8f0; margin-bottom: 10px;">Upload an Image to Begin</h2>
                <p style="color: #64748b; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                    Drag and drop or click the uploader above to classify your cat or dog image using our advanced AI model.
                </p>
                <div style="margin-top: 30px; display: flex; justify-content: center; gap: 15px;">
                    <span style="background: rgba(234, 88, 12, 0.2); color: #fb923c; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">🐱 Cats</span>
                    <span style="background: rgba(37, 99, 235, 0.2); color: #60a5fa; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">🐶 Dogs</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# COMPARE IMAGES PAGE
# ============================================================
elif page == "⚔️ Compare Images":
    st.markdown("<div class='section-title'>⚔️ Compare Two Images</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 20px;'>Upload two images and compare their classification results side by side.</p>", unsafe_allow_html=True)

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        file1 = st.file_uploader("Upload Image 1", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], key="img1")
    with col_up2:
        file2 = st.file_uploader("Upload Image 2", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], key="img2")

    if file1 and file2 and model is not None:
        img1 = Image.open(file1)
        img2 = Image.open(file2)

        if img1.mode == 'RGBA':
            img1 = img1.convert('RGB')
        if img2.mode == 'RGBA':
            img2 = img2.convert('RGB')

        # Predict both
        arr1 = preprocess_image(img1)
        arr2 = preprocess_image(img2)

        with st.spinner("🧠 Analyzing both images..."):
            res1 = predict_image(model, arr1)
            res2 = predict_image(model, arr2)

        # Display comparison
        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        col_res1, col_vs, col_res2 = st.columns([2, 0.5, 2])

        with col_res1:
            st.image(img1, use_container_width=True, caption="Image 1")
            st.markdown(f"""
                <div class="prediction-box {'cat-pred' if res1['class'] == 'Cat' else 'dog-pred'}" style="margin-top: 15px;">
                    <div style="font-size: 3rem;">{res1['emoji']}</div>
                    <h2 style="margin: 5px 0;">{res1['class']}</h2>
                    <p style="font-size: 1.2rem; margin: 0;">{res1['confidence']:.2f}% confidence</p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(res1['confidence'] / 100, text=f"Confidence: {res1['confidence']:.1f}%")

        with col_vs:
            st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
            st.markdown("""
                <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
                    <div class="vs-badge">VS</div>
                </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.image(img2, use_container_width=True, caption="Image 2")
            st.markdown(f"""
                <div class="prediction-box {'cat-pred' if res2['class'] == 'Cat' else 'dog-pred'}" style="margin-top: 15px;">
                    <div style="font-size: 3rem;">{res2['emoji']}</div>
                    <h2 style="margin: 5px 0;">{res2['class']}</h2>
                    <p style="font-size: 1.2rem; margin: 0;">{res2['confidence']:.2f}% confidence</p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(res2['confidence'] / 100, text=f"Confidence: {res2['confidence']:.1f}%")

        # Comparison chart
        st.markdown("<div class='section-title' style='margin-top: 30px;'>📊 Comparison Chart</div>", unsafe_allow_html=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Image 1',
            x=['Cat', 'Dog'],
            y=[res1['cat_prob'], res1['dog_prob']],
            marker_color=['rgba(234, 88, 12, 0.8)', 'rgba(37, 99, 235, 0.8)'],
            text=[f'{res1["cat_prob"]:.1f}%', f'{res1["dog_prob"]:.1f}%'],
            textposition='auto'
        ))
        fig_comp.add_trace(go.Bar(
            name='Image 2',
            x=['Cat', 'Dog'],
            y=[res2['cat_prob'], res2['dog_prob']],
            marker_color=['rgba(234, 88, 12, 0.4)', 'rgba(37, 99, 235, 0.4)'],
            text=[f'{res2["cat_prob"]:.1f}%', f'{res2["dog_prob"]:.1f}%'],
            textposition='auto'
        ))
        fig_comp.update_layout(
            barmode='group',
            title='Probability Comparison',
            yaxis_title='Probability (%)',
            yaxis_range=[0, 100],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font_color='white',
            legend=dict(font_color='white'),
            height=400
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Winner announcement
        if res1['confidence'] > res2['confidence']:
            winner = "Image 1"
            winner_emoji = res1['emoji']
            winner_conf = res1['confidence']
        else:
            winner = "Image 2"
            winner_emoji = res2['emoji']
            winner_conf = res2['confidence']

        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f59e0b, #d97706); padding: 20px; border-radius: 16px; text-align: center; margin-top: 20px;">
                <h2 style="margin: 0; color: white;">🏆 Winner: {winner}</h2>
                <p style="margin: 10px 0 0 0; color: white; font-size: 1.2rem;">{winner_emoji} with {winner_conf:.2f}% confidence</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Please upload both images to start the comparison.")

# ============================================================
# ANALYTICS PAGE
# ============================================================
elif page == "📊 Analytics":
    st.markdown("<div class='section-title'>📊 Analytics Dashboard</div>", unsafe_allow_html=True)

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_predictions}</div>
                <div class="metric-label">Total Predictions</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ea580c;">{st.session_state.cat_count}</div>
                <div class="metric-label">Cats Detected</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #2563eb;">{st.session_state.dog_count}</div>
                <div class="metric-label">Dogs Detected</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_conf = 0
        if st.session_state.history:
            avg_conf = np.mean([h['confidence'] for h in st.session_state.history])
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #a78bfa;">{avg_conf:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div class='section-title'>🥧 Class Distribution</div>", unsafe_allow_html=True)
        dist_chart = create_distribution_chart()
        if dist_chart:
            st.plotly_chart(dist_chart, use_container_width=True)
        else:
            st.info("No predictions yet. Start classifying images!")

    with col_chart2:
        st.markdown("<div class='section-title'>📈 Confidence Trend</div>", unsafe_allow_html=True)
        hist_chart = create_history_chart()
        if hist_chart:
            st.plotly_chart(hist_chart, use_container_width=True)
        else:
            st.info("No history data available yet.")

    st.divider()

    # Export section
    st.markdown("<div class='section-title'>💾 Export Data</div>", unsafe_allow_html=True)
    if st.session_state.history:
        json_data = export_results()
        st.download_button(
            label="📥 Download History (JSON)",
            data=json_data,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # CSV export
        csv_data = []
        for item in st.session_state.history:
            csv_data.append({
                'Timestamp': item['timestamp'],
                'Class': item['class'],
                'Confidence': f"{item['confidence']:.2f}",
                'Cat Prob': f"{item['cat_prob']:.2f}",
                'Dog Prob': f"{item['dog_prob']:.2f}"
            })
        df_csv = pd.DataFrame(csv_data)
        csv_buffer = io.StringIO()
        df_csv.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download History (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No data to export yet.")

# ============================================================
# HISTORY PAGE
# ============================================================
elif page == "📜 History":
    st.markdown("<div class='section-title'>📜 Prediction History</div>", unsafe_allow_html=True)

    if st.session_state.history:
        # Filter options
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_class = st.selectbox("Filter by Class", ["All", "Cat", "Dog"])
        with col_f2:
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.history = []
                st.session_state.total_predictions = 0
                st.session_state.cat_count = 0
                st.session_state.dog_count = 0
                st.rerun()

        # Display history
        filtered_history = st.session_state.history
        if filter_class != "All":
            filtered_history = [h for h in filtered_history if h['class'] == filter_class]

        for i, item in enumerate(filtered_history):
            with st.container():
                st.markdown(f"""
                    <div class="history-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.5rem; margin-right: 10px;">{'🐱' if item['class'] == 'Cat' else '🐶'}</span>
                                <span style="font-weight: 600; color: #e2e8f0;">{item['class']}</span>
                                <span style="color: #64748b; margin-left: 10px; font-size: 0.9rem;">{item['timestamp']}</span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-weight: 700; color: {'#ea580c' if item['class'] == 'Cat' else '#2563eb'}; font-size: 1.2rem;">{item['confidence']:.2f}%</span>
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-bottom: 3px;">
                                <span>Confidence</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {item['confidence']}%; background: linear-gradient(90deg, {'#ea580c' if item['class'] == 'Cat' else '#2563eb'}, {'#fb923c' if item['class'] == 'Cat' else '#60a5fa'});"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Show image thumbnail
                with st.expander("🖼️ View Image"):
                    st.image(item['image'], width=300)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 60px; margin-bottom: 20px;">📭</div>
                <h2 style="color: #e2e8f0;">No History Yet</h2>
                <p style="color: #64748b;">Start classifying images to build your prediction history.</p>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 20px; font-size: 0.9rem;">
        <p>Made with ❤️ using TensorFlow, Keras & Streamlit</p>
        <p style="margin-top: 5px; font-size: 0.8rem;">Professional Edition v2.0 | Deep Learning Image Classification</p>
    </div>
""", unsafe_allow_html=True)