import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# python -m streamlit run e_StreamlitPipeline.py

# CONFIG
st.set_page_config(
    page_title="Placement Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD MODELS
rf_model_class = joblib	.load("artifacts/classifier.pkl")
lr_model_reg = joblib.load("artifacts/regressor.pkl")

# SIDEBAR
st.sidebar.title("🎓 Navigation")
menu = st.sidebar.radio("Menu", ["Prediction Dashboard", "About Model"])

st.sidebar.markdown("---")
st.sidebar.info("ML Placement System v1.0")

# MAIN APP
if menu == "Prediction Dashboard":

    st.title("🎯 Placement & Salary Prediction System")
    st.caption("Predict student placement status and expected salary (if placed)")


    # FORM INPUT
    with st.form("input_form"):

        st.subheader("📌 Student Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.radio("Gender", ["Male", "Female"])
            branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "ME", "CE"])
            part_time_job = st.radio("Part Time Job", ["No", "Yes"])
            internet_access = st.radio("Internet Access", ["Yes", "No"])

        with col2:
            family_income_level = st.radio("Family Income", ["Low", "Medium", "High"])
            city_tier = st.radio("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
            extracurricular_involvement = st.radio("Extracurricular", ["Low", "Medium", "High"])

        with col3:
            cgpa = st.number_input("CGPA", 5.0, 10.0)
            tenth_percentage = st.number_input("10th %", 50.0, 100.0)
            twelfth_percentage = st.number_input("12th %", 50.0, 100.0)

        st.subheader("📊 Performance Metrics")

        col4, col5 = st.columns(2)

        with col4:
            backlogs = st.number_input("Backlogs", 0, 10)
            study_hours_per_day = st.number_input("Study Hours", 0.0, 10.0)
            attendance_percentage = st.number_input("Attendance %", 0.0, 100.0)

        with col5:
            projects_completed = st.number_input("Projects", 0, 8)
            internships_completed = st.number_input("Internships", 0, 4)
            hackathons_participated = st.number_input("Hackathons", 0, 6)
            certifications_count = st.number_input("Certifications", 0, 9)

        st.subheader("🧠 Skills")

        col6, col7, col8 = st.columns(3)

        with col6:
            coding_skill_rating = st.slider("Coding Skill", 1, 5)
        with col7:
            communication_skill_rating = st.slider("Communication", 1, 5)
        with col8:
            aptitude_skill_rating = st.slider("Aptitude", 1, 5)

        st.subheader("🧘 Lifestyle")

        col9, col10 = st.columns(2)

        with col9:
            sleep_hours = st.number_input("Sleep Hours", 4.0, 9.0)
        with col10:
            stress_level = st.slider("Stress Level", 1, 10)

        submit = st.form_submit_button("🚀 Predict Now")


    # PROCESS
    if submit:

        data = {
            "gender": gender,
            "branch": branch,
            "part_time_job": part_time_job,
            "internet_access": internet_access,
            "family_income_level": family_income_level,
            "city_tier": city_tier,
            "extracurricular_involvement": extracurricular_involvement,

            "cgpa": float(cgpa),
            "tenth_percentage": float(tenth_percentage),
            "twelfth_percentage": float(twelfth_percentage),

            "backlogs": int(backlogs),
            "study_hours_per_day": float(study_hours_per_day),
            "attendance_percentage": float(attendance_percentage),

            "projects_completed": int(projects_completed),
            "internships_completed": int(internships_completed),
            "coding_skill_rating": int(coding_skill_rating),
            "communication_skill_rating": int(communication_skill_rating),
            "aptitude_skill_rating": int(aptitude_skill_rating),

            "hackathons_participated": int(hackathons_participated),
            "certifications_count": int(certifications_count),

            "sleep_hours": float(sleep_hours),
            "stress_level": int(stress_level)
        }

        df = pd.DataFrame([data])

        # engineered features
        df["academic_score"] = (df["cgpa"] + df["tenth_percentage"] / 10 + df["twelfth_percentage"] / 10) / 3
        df["experience_score"] = (df["projects_completed"] + df["internships_completed"] * 2 + df["hackathons_participated"] + df["certifications_count"])
        df["Student_ID"] = 0 
        
        # PREDICTION
        placement = rf_model_class.predict(df)[0]

        st.divider()
        st.subheader("📊 Result Dashboard")

        colA, colB = st.columns(2)

        with colA:
            if placement == 0:
                st.error("❌ Not Placed")
                salary = None
            else:
                st.success("✅ Placed")
                salary = lr_model_reg.predict(df)[0]
                st.metric("Predicted Salary (LPA)", f"{salary:.2f}")

        with colB:
            st.metric("Academic Score", f"{df['academic_score'].values[0]:.2f}")
            st.metric("Experience Score", f"{df['experience_score'].values[0]:.2f}")

    
        # VISUALIZATION
        st.subheader("📈 Skill Overview")

        skills = {
            "Coding": coding_skill_rating,
            "Communication": communication_skill_rating,
            "Aptitude": aptitude_skill_rating,
            "Academic": df["academic_score"].values[0],
            "Experience": df["experience_score"].values[0]
        }

        fig, ax = plt.subplots()
        ax.bar(skills.keys(), skills.values())
        ax.set_ylim(0, max(skills.values()) + 2)
        plt.xticks(rotation=30)

        st.pyplot(fig)

        # DATA PREVIEW
        st.subheader("🧾 Input Summary")
        st.dataframe(df, use_container_width=True)


# ABOUT PAGE
elif menu == "About Model":
    st.title("ℹ️ About This Application")
    st.markdown("""
## About This Application

Aplikasi ini merupakan sistem berbasis Machine Learning yang dirancang untuk memprediksi dua aspek utama pada mahasiswa:

- **Placement Prediction (Classification)**: menentukan status penempatan kerja (*Placed / Not Placed*)
- **Salary Prediction (Regression)**: memperkirakan gaji awal dalam satuan LPA (Lakh Per Annum) bagi mahasiswa yang berhasil ditempatkan

Model yang digunakan dalam sistem ini:
- Random Forest Classifier untuk prediksi status penempatan kerja  
- Linear Regression untuk estimasi gaji mahasiswa  

Sistem ini dibangun menggunakan pipeline Machine Learning berbasis Scikit-Learn dan di-deploy menggunakan Streamlit sebagai antarmuka interaktif.

---

## Dataset Context (Academic & Student Profiling Data)

Dataset yang digunakan merepresentasikan profil mahasiswa secara menyeluruh yang mencakup aspek akademik, sosial, pengalaman, keterampilan, dan gaya hidup.

Dataset ini umum digunakan dalam studi *campus placement prediction* untuk menganalisis faktor-faktor yang memengaruhi kesiapan mahasiswa memasuki dunia kerja.

---

## 1. Demografi dan Sosial Ekonomi

- **Gender**: jenis kelamin mahasiswa (Male / Female)  
- **Branch**: program studi (CSE - Computer Science & Engineering, ECE - Electronics and Communication Engineering, IT - Information Technology, ME - Mechanical Engineering, CE - Civil Engineering)  
- **Family Income**: tingkat pendapatan keluarga (Low / Medium / High)  
- **City Tier**: kategori kota asal (Tier 1 / Tier 2 / Tier 3)  

---

## 2. Aktivitas dan Akses Pendukung

- **Part Time Job**: status pekerjaan paruh waktu (Yes / No)  
- **Internet Access**: ketersediaan akses internet untuk pembelajaran digital (Yes / No)  
- **Extracurricular Involvement**: tingkat keterlibatan kegiatan non-akademik (Low / Medium / High)  

---

## 3. Performa Akademik

- **CGPA**: rata-rata nilai akademik selama perkuliahan (skala 5.00 – 10.00)  
- **10th Percentage**: nilai pendidikan SMP (50 – 100)  
- **12th Percentage**: nilai pendidikan SMA (50 – 100)  
- **Backlogs**: jumlah mata kuliah yang belum lulus  

---

## 4. Kebiasaan Belajar dan Disiplin

- **Study Hours per Day**: rata-rata waktu belajar harian  
- **Attendance Percentage**: tingkat kehadiran dalam perkuliahan  

---

## 5. Pengalaman dan Keterampilan

- **Projects Completed**: jumlah proyek yang telah dikerjakan  
- **Internships Completed**: pengalaman magang di industri  
- **Hackathons Participated**: partisipasi dalam kompetisi teknologi  
- **Certifications Count**: jumlah sertifikasi tambahan dari kursus atau pelatihan  

---

## 6. Soft Skills dan Kemampuan Teknis

- **Coding Skill (1–5)**: tingkat kemampuan pemrograman  
- **Communication Skill (1–5)**: kemampuan komunikasi interpersonal  
- **Aptitude Skill (1–5)**: kemampuan logika, analisis, dan problem solving  

---

## 7. Faktor Gaya Hidup

- **Sleep Hours**: rata-rata durasi tidur harian  
- **Stress Level**: tingkat stres (skala 1–10)  

---

## Kesimpulan

Model ini menggabungkan berbagai dimensi data, tidak hanya aspek akademik, tetapi juga pengalaman, keterampilan, dan faktor gaya hidup. Pendekatan ini menghasilkan prediksi yang lebih komprehensif terhadap kesiapan mahasiswa dalam memasuki dunia kerja (*employability prediction*), dibandingkan pendekatan yang hanya berfokus pada nilai akademik.
""")