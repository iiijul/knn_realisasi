import streamlit as st
import pickle
import pandas as pd
import numpy as np
import altair as alt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Prediksi Realisasi Anggaran",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHE & LOAD DATA/MODEL
# ============================================================================
@st.cache_resource
def load_model_and_data():
    """Load model dan data training"""
    with open('model/knn.pkcls', 'rb') as f:
        model = pickle.load(f)
    
    data = pd.read_csv('data/02_realisasi_anggaran_klasifikasi.csv')
    return model, data

@st.cache_data
def prepare_features_data(data):
    """Persiapkan data fitur untuk analisis"""
    # Features yang digunakan model
    feature_cols = ['jumlah_spm', 'revisi_dipa', 'deviasi_rpd_persen', 'skor_ikpa', 'tipe_satker']
    
    # Encode tipe_satker - gunakan underscore sesuai dengan output pd.get_dummies
    data_encoded = data.copy()
    tipe_satker_dummies = pd.get_dummies(data_encoded['tipe_satker'], prefix='tipe_satker')
    data_encoded = pd.concat([data_encoded[['jumlah_spm', 'revisi_dipa', 'deviasi_rpd_persen', 'skor_ikpa']], 
                             tipe_satker_dummies], axis=1)
    
    return feature_cols, data_encoded

# Load resources
model, data = load_model_and_data()
feature_cols, data_encoded = prepare_features_data(data)

# ============================================================================
# HEADER & PENJELASAN
# ============================================================================
st.title("📊 Sistem Prediksi Realisasi Anggaran (KNN Classifier)")
st.markdown("---")

with st.expander("ℹ️ **Informasi Model & Fitur**", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model")
        st.write("- **Algoritma:** k-Nearest Neighbors (KNN)")
        st.write("- **Dataset:** 500 satuan kerja pemerintah Indonesia")
        st.write("- **Target:** Prediksi realisasi anggaran mencapai 95%")
        st.write("- **Classes:** Ya (207 records) / Tidak (293 records)")
    
    with col2:
        st.subheader("Fitur Input Model (5 Features)")
        st.write("1. **Jumlah SPM** - Banyak Surat Perintah Membayar")
        st.write("2. **Revisi DIPA** - Jumlah perubahan DIPA")
        st.write("3. **Deviasi RPD** - Penyimpangan Rencana Penarikan Dana (%)")
        st.write("4. **Skor IKPA** - Indikator Kinerja Pembiayaan Anggaran")
        st.write("5. **Tipe Satker** - Jenis satuan kerja (one-hot encoded)")

# ============================================================================
# SIDEBAR - INPUT FITUR
# ============================================================================
st.sidebar.header("🎯 Input Fitur untuk Prediksi")
st.sidebar.markdown("---")

# Input untuk numeric features
jumlah_spm = st.sidebar.slider(
    "Jumlah SPM",
    min_value=int(data['jumlah_spm'].min()),
    max_value=int(data['jumlah_spm'].max()),
    value=int(data['jumlah_spm'].median()),
    help=f"Range: {int(data['jumlah_spm'].min())} - {int(data['jumlah_spm'].max())}"
)

revisi_dipa = st.sidebar.slider(
    "Revisi DIPA",
    min_value=int(data['revisi_dipa'].min()),
    max_value=int(data['revisi_dipa'].max()),
    value=int(data['revisi_dipa'].median()),
    help=f"Range: {int(data['revisi_dipa'].min())} - {int(data['revisi_dipa'].max())}"
)

deviasi_rpd = st.sidebar.slider(
    "Deviasi RPD (%)",
    min_value=float(data['deviasi_rpd_persen'].min()),
    max_value=float(data['deviasi_rpd_persen'].max()),
    value=float(data['deviasi_rpd_persen'].median()),
    step=0.1,
    help=f"Range: {data['deviasi_rpd_persen'].min():.2f}% - {data['deviasi_rpd_persen'].max():.2f}%"
)

skor_ikpa = st.sidebar.slider(
    "Skor IKPA",
    min_value=float(data['skor_ikpa'].min()),
    max_value=float(data['skor_ikpa'].max()),
    value=float(data['skor_ikpa'].median()),
    step=0.1,
    help=f"Range: {data['skor_ikpa'].min():.2f} - {data['skor_ikpa'].max():.2f}"
)

tipe_satker = st.sidebar.selectbox(
    "Tipe Satuan Kerja",
    options=['Kantor Pusat', 'Kantor Daerah', 'Dekonsentrasi', 'Tugas Pembantuan'],
    help="Pilih jenis satuan kerja"
)

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================
tab_prediksi, tab_visual = st.tabs(["Prediksi", "Visualisasi Data & Statistik"])

with tab_prediksi:
    col_pred, col_stats = st.columns([3, 2])

    with col_pred:
        st.subheader("🔮 Hasil Prediksi")
        
        # Persiapan data untuk prediksi
        input_data = pd.DataFrame({
            'jumlah_spm': [jumlah_spm],
            'revisi_dipa': [revisi_dipa],
            'deviasi_rpd_persen': [deviasi_rpd],
            'skor_ikpa': [skor_ikpa],
            'tipe_satker': [tipe_satker]
        })
        
        # Encode tipe_satker untuk input - gunakan underscore sesuai dengan pd.get_dummies
        input_encoded = pd.DataFrame({
            'jumlah_spm': [jumlah_spm],
            'revisi_dipa': [revisi_dipa],
            'deviasi_rpd_persen': [deviasi_rpd],
            'skor_ikpa': [skor_ikpa],
            'tipe_satker_Dekonsentrasi': [1 if tipe_satker == 'Dekonsentrasi' else 0],
            'tipe_satker_Kantor Daerah': [1 if tipe_satker == 'Kantor Daerah' else 0],
            'tipe_satker_Kantor Pusat': [1 if tipe_satker == 'Kantor Pusat' else 0],
            'tipe_satker_Tugas Pembantuan': [1 if tipe_satker == 'Tugas Pembantuan' else 0]
        })
        
        try:
            # Persiapan data dan prediksi
            X_input = input_encoded.values
            
            # Ekstrak sklearn model untuk prediksi
            skl_model = model.skl_model
            
            # Get class prediction
            class_pred = skl_model.predict(X_input)[0]
            
            # Get probabilities
            proba = skl_model.predict_proba(X_input)[0]
            
            # Map class to name
            if class_pred == 0:
                predicted_class = 'Tidak'
                proba_tidak = proba[0]
                proba_ya = proba[1]
            else:
                predicted_class = 'Ya'
                proba_tidak = proba[0]
                proba_ya = proba[1]
            
            # Display prediksi
            if predicted_class == 'Ya':
                st.success(f"### ✅ Prediksi: **TERCAPAI (Ya)**", icon="✅")
            else:
                st.error(f"### ❌ Prediksi: **TIDAK TERCAPAI (Tidak)**", icon="❌")
            
            # Probabilitas
            st.markdown("#### Confidence Score")
            
            col_ya, col_tidak = st.columns(2)
            with col_ya:
                st.metric("Probabilitas TERCAPAI (Ya)", f"{proba_ya*100:.1f}%")
            with col_tidak:
                st.metric("Probabilitas TIDAK (Tidak)", f"{proba_tidak*100:.1f}%")
            
            # Confidence visualization
            st.markdown("#### Confidence Level")
            max_proba = max(proba_ya, proba_tidak)
            st.progress(max_proba, f"Confidence: {max_proba*100:.1f}%")
            
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")

    with col_stats:
        st.subheader("📈 Statistik Input")
        st.markdown("**Dibandingkan dengan Dataset:**")
        
        stats_data = {
            'Fitur': ['Jumlah SPM', 'Revisi DIPA', 'Deviasi RPD (%)', 'Skor IKPA'],
            'Input': [jumlah_spm, revisi_dipa, round(deviasi_rpd, 2), round(skor_ikpa, 2)],
            'Mean': [
                round(data['jumlah_spm'].mean(), 1),
                round(data['revisi_dipa'].mean(), 1),
                round(data['deviasi_rpd_persen'].mean(), 2),
                round(data['skor_ikpa'].mean(), 2)
            ],
            'Min': [
                int(data['jumlah_spm'].min()),
                int(data['revisi_dipa'].min()),
                round(data['deviasi_rpd_persen'].min(), 2),
                round(data['skor_ikpa'].min(), 2)
            ],
            'Max': [
                int(data['jumlah_spm'].max()),
                int(data['revisi_dipa'].max()),
                round(data['deviasi_rpd_persen'].max(), 2),
                round(data['skor_ikpa'].max(), 2)
            ]
        }
        
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Tetangga Terdekat (Kasus Serupa dari Training Data)")
    st.write("Data yang paling mirip dengan input Anda. Fitur-fitur ini mempengaruhi prediksi karena KNN menggunakan jarak euclidean dari tetangga terdekat.")

    try:
        # Prepare features for nearest neighbor search
        features_for_neighbors = data_encoded.copy()
        
        # Normalisasi
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_for_neighbors)
        
        # Input normalized
        input_normalized = scaler.transform(input_encoded)
        
        # Cari k nearest neighbors
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(features_normalized)
        distances, indices = nbrs.kneighbors(input_normalized)
        
        # Ambil data tetangga
        neighbors_data = data.iloc[indices[0]].copy()
        neighbors_data['distance'] = distances[0]
        
        # Display tetangga
        display_cols = ['kode_satker', 'nama_kementerian', 'tipe_satker', 'jumlah_spm', 
                        'revisi_dipa', 'deviasi_rpd_persen', 'skor_ikpa', 'realisasi_tercapai_95persen', 'distance']
        
        neighbors_display = neighbors_data[display_cols].copy()
        neighbors_display['distance'] = neighbors_display['distance'].round(3)
        neighbors_display.columns = ['Kode', 'Kementerian', 'Tipe Satker', 'SPM', 'Revisi DIPA', 
                                     'Deviasi RPD (%)', 'Skor IKPA', 'Target', 'Jarak']
        
        st.dataframe(neighbors_display, hide_index=True, use_container_width=True)
        
        # Insight dari neighbors
        st.markdown("#### 📌 Insight dari Tetangga Terdekat:")
        target_counts = neighbors_data['realisasi_tercapai_95persen'].value_counts()
        
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        with col_insight1:
            st.metric("Tetangga 'Ya'", target_counts.get('Ya', 0))
        with col_insight2:
            st.metric("Tetangga 'Tidak'", target_counts.get('Tidak', 0))
        with col_insight3:
            avg_distance = neighbors_data['distance'].mean()
            st.metric("Rata-rata Jarak", f"{avg_distance:.3f}")
        
    except Exception as e:
        st.warning(f"Error dalam mencari tetangga terdekat: {str(e)}")

    st.markdown("---")
    st.subheader("🎯 Konteks Fitur yang Mempengaruhi Prediksi")
    st.write("Fitur-fitur berikut digunakan oleh model KNN untuk membuat prediksi. Perubahan nilai pada fitur ini akan mempengaruhi hasil prediksi.")

    feature_context = {
        'Fitur': ['Jumlah SPM', 'Revisi DIPA', 'Deviasi RPD (%)', 'Skor IKPA', 'Tipe Satker'],
        'Penjelasan': [
            'Banyaknya dokumen pembayaran. Lebih banyak SPM menunjukkan aktivitas transaksi yang tinggi.',
            'Jumlah perubahan anggaran DIPA. Revisi banyak dapat menunjukkan ketidakstabilan perencanaan.',
            'Penyimpangan dari Rencana Penarikan Dana. Deviasi tinggi menunjukkan ketidaktepatan perencanaan likuiditas.',
            'Indikator Kinerja Pembiayaan Anggaran. Skor lebih tinggi menunjukkan performa yang lebih baik.',
            'Jenis organisasi: Kantor Pusat, Kantor Daerah, Dekonsentrasi, atau Tugas Pembantuan.'
        ],
        'Pengaruh': [
            'Positif - SPM tinggi biasanya mengindikasikan realisasi lebih baik',
            'Negatif - Revisi terlalu banyak dapat mengganggu pencapaian target',
            'Negatif - Deviasi RPD tinggi menunjukkan manajemen arus kas kurang baik',
            'Positif - Skor IKPA tinggi menunjukkan kinerja pembiayaan lebih baik',
            'Bervariasi - Setiap tipe satker memiliki karakteristik pencapaian berbeda'
        ]
    }

    st.dataframe(pd.DataFrame(feature_context), hide_index=True, use_container_width=True)

with tab_visual:
    st.subheader("📊 Visualisasi Data & Statistik Deskriptif")
    st.write("Analisis mendalam dari dataset dan alasan mengapa kNN adalah pilihan yang cocok untuk model ini.")

    overview_cols = ['jumlah_spm', 'revisi_dipa', 'deviasi_rpd_persen', 'skor_ikpa']
    summary = data[overview_cols].describe().T.reset_index().rename(columns={'index': 'Fitur'})

    st.markdown("#### Ringkasan Statistik Deskriptif")
    st.dataframe(summary, use_container_width=True)

    st.markdown("#### Komposisi Target")
    class_counts = data['realisasi_tercapai_95persen'].value_counts().reset_index()
    class_counts.columns = ['Target', 'Jumlah']
    class_chart = alt.Chart(class_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('Target:N', sort='-y', title=None),
        y=alt.Y('Jumlah:Q', title='Jumlah Satker'),
        color=alt.Color('Target:N', legend=None),
        tooltip=['Target', 'Jumlah']
    ).properties(height=320)
    st.altair_chart(class_chart, use_container_width=True)

    st.markdown("#### Distribusi Fitur Numerik")
    long_features = data[overview_cols].transform(lambda x: x.astype(float)).reset_index().melt(id_vars=['index'], var_name='Fitur', value_name='Nilai')
    hist_chart = alt.Chart(long_features).mark_bar(opacity=0.8).encode(
        x=alt.X('Nilai:Q', bin=alt.Bin(maxbins=40), title='Nilai'),
        y=alt.Y('count():Q', title='Frekuensi'),
        column=alt.Column('Fitur:N', header=alt.Header(labelAngle=0, titleOrient='top')),
        color=alt.Color('Fitur:N', legend=None)
    ).properties(height=250)
    st.altair_chart(hist_chart, use_container_width=True)

    st.markdown("#### Korelasi Antar Fitur dan Target")
    corr_df = data[overview_cols + ['realisasi_tercapai_95persen']].copy()
    corr_df['target_code'] = corr_df['realisasi_tercapai_95persen'].map({'Tidak': 0, 'Ya': 1})
    corr_matrix = corr_df[overview_cols + ['target_code']].corr().reset_index().melt(id_vars='index', var_name='Feature', value_name='Correlation')
    corr_matrix.columns = ['Fitur', 'Feature', 'Correlation']
    heatmap = alt.Chart(corr_matrix).mark_rect().encode(
        x=alt.X('Feature:N', title=None),
        y=alt.Y('Fitur:N', title=None),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
        tooltip=['Fitur', 'Feature', alt.Tooltip('Correlation:Q', format='.2f')]
    ).properties(height=320)
    st.altair_chart(heatmap, use_container_width=True)

    st.markdown("#### Visualisasi Kinerja Fitur")
    scatter = alt.Chart(data).mark_circle(size=70, opacity=0.7).encode(
        x='skor_ikpa:Q',
        y='revisi_dipa:Q',
        color='realisasi_tercapai_95persen:N',
        tooltip=['kode_satker', 'nama_kementerian', 'skor_ikpa', 'revisi_dipa', 'realisasi_tercapai_95persen']
    ).properties(height=380)
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("#### Mengapa kNN?")
    st.write("kNN dipilih karena karakteristik dataset dan tujuan analisis ini:")
    st.write("- kNN adalah model berbasis kemiripan yang mudah dijelaskan dengan tetangga terdekat.")
    st.write("- Sangat cocok untuk dataset ukuran menengah dengan fitur numerik dan kategori yang bermakna.")
    st.write("- Tanpa asumsi bentuk distribusi, sehingga cocok untuk data realisasi anggaran yang tidak selalu linear.")
    st.write("- Hasilnya bisa divalidasi langsung dengan melihat data tetangga terdekat dari input pengguna.")
    st.write("- Karena kita ingin memahami kasus serupa, kNN membantu menjelaskan prediksi secara intuitif.")

    st.markdown("#### Catatan Akurasi & Interpretasi")
    st.write("Model kNN bekerja dengan menghitung jarak Euclidean antara input dan data training, sehingga fitur yang distandarisasi dan one-hot encoding tipe satker penting untuk keakuratan.")
    st.write("Semakin banyak tetangga serupa (dengan jarak kecil) yang memiliki target 'Ya', semakin tinggi kepercayaan model pada prediksi tercapai.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>📊 KNN Classifier untuk Prediksi Realisasi Anggaran</p>
    <p>Model: Orange3 KNN | Dataset: 500 satuan kerja | Features: 5 dimensi</p>
</div>
""", unsafe_allow_html=True)
