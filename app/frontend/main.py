import streamlit as st
import pandas as pd
import plotly.express as px
from app.inference_service.model_engine import ViExToxicModel
from app.inference_service.streaming_engine import MockSparkStreaming
from datetime import datetime

# 1. Cấu hình trang
st.set_page_config(page_title='Vi-ExToxic Streaming Analyzer', layout='wide')

# Custom CSS cho màu đỏ đậm và UI
st.markdown("""
    <style>
    .main-title { color: #8B0000; font-size: 40px; font-weight: bold; text-align: center; }
    .stExpander { border: 1px solid #8B0000; }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = ViExToxicModel()

# 2. Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150?text=Vi-ExToxic+Logo", width=150)
    st.title("Settings")
    st.info("**Model:** Qwen 2.5-3B (LoRA Fine-tuned)\n\n**Engine:** Spark Streaming Backend")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# 3. Main Page
st.markdown("<p class='main-title'>Vi-ExToxic Analyzer System</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Manual Analysis", "📡 Live Stream Monitor", "📊 Dashboard Stats"])

# --- TAB 1: Manual Analysis ---
with tab1:
    user_input = st.text_area("Nhập nội dung cần phân tích:", placeholder="Ví dụ: Giỏi quá vcl cả họ tự hào smirk")
    
    if st.button("Analyze Now"):
        if not user_input.strip():
            st.error("⚠️ Vui lòng nhập nội dung trước khi phân tích!")
        else:
            with st.status("🔮 Qwen 2.5-3B is thinking...", expanded=True) as status:
                st.write("Decoding semantics...")
                result = st.session_state.model.predict(user_input)
                st.write("Checking contextual conflicts...")
                st.write("Finalizing logic...")
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # Hiển thị kết quả 3 cột
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.subheader("📋 Reasoning Scaffolding")
                st.json(result["reasoning_scaffolding"])
            
            with col2:
                st.subheader("🧠 Thought Trace")
                with st.expander("View AI Logic", expanded=True):
                    st.write(result["thought_trace"])
            
            with col3:
                st.subheader("🎯 Final Label")
                label = result["final_label"]
                color = "red" if "Hostility" in label or "Toxicity" in label or "Hate" in label else "green"
                st.markdown(f"""
                    <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
                        <h2 style="color:white; margin:0;">{label}</h2>
                        <p style="color:white; margin:0;">Confidence: {result['confidence_score']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.session_state.history.append({**result, "text": user_input, "time": datetime.now()})

# --- TAB 2: Live Stream Monitor ---
with tab2:
    stream_url = st.text_input("Enter YouTube Link / Stream URL:", placeholder="https://www.youtube.com/watch?v=...")
    start_stream = st.button("Start Live Monitor")
    
    if start_stream:
        st.success(f"Connected to stream: {stream_url}")
        streamer = MockSparkStreaming()
        placeholder = st.empty()
        
        for comment in streamer.stream_generator():
            res = st.session_state.model.predict(comment)
            st.session_state.history.append({**res, "text": comment, "time": datetime.now()})
            
            with placeholder.container():
                st.markdown(f"**Latest Comment:** _{comment}_")
                # Visualize mini-card cho streaming
                c1, c2 = st.columns([3, 1])
                c1.info(f"AI Thought: {res['thought_trace']}")
                label_color = "🔴" if "Hostility" in res['final_label'] else "🟢"
                c2.metric("Label", f"{label_color} {res['final_label']}")
                st.divider()

# --- TAB 3: Dashboard Stats ---
with tab3:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.pie(df, names='final_label', title='Phân bổ nhãn độc hại', color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            df['hour'] = pd.to_datetime(df['time']).dt.second # Demo theo giây cho nhanh
            fig2 = px.line(df.groupby('hour').size().reset_index(name='counts'), x='hour', y='counts', title='Tần suất comment theo thời gian')
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("Dữ liệu chi tiết")
        st.dataframe(df[['time', 'text', 'final_label', 'confidence_score']])
    else:
        st.info("Chưa có dữ liệu để hiển thị thống kê.")


# python -m streamlit run app/frontend/main.py