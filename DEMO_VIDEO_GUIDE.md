# Demo Video Presentation & Recording Guide

This guide provides a structured, word-for-word presentation script and recording checklist for the **3–5 minute video demonstration** required for the **MNR University Machine Learning Mini Project Submission**.

---

## 📹 How to Record Your Screen on Windows

1. **Quickest Built-in Tool: Windows Game Bar**
   - Press `Win + Alt + R` to immediately begin recording your desktop screen and microphone.
   - Speak clearly at an even pace.
   - Press `Win + Alt + R` again to stop recording.
   - Captured MP4 files are located in: `This PC > Videos > Captures`.

2. **Alternative Recording Tools:**
   - Microsoft Clipchamp (built-in Windows 11 application).
   - OBS Studio (free, open-source screen recorder).

---

## 🎙️ 3–5 Minute Presentation Script

### ⏱️ Minute 0:00 – 0:50 | Introduction & Problem Overview
> *"Respected faculty and evaluators, greetings. My name is K. Tanmaisri (Roll No: 25RBAIM0049), from MNR University, School of Engineering, Technology & Artificial Intelligence, Department of Computer Science & Engineering, 2nd Year 1st Semester. Today I present our Machine Learning Mini Project entitled **'AI-Based Student Skill Gap Analysis and Career Recommendation System'**, developed under the guidance of Dr. Praveen M.*
>
> *Today, engineering and AI/ML students frequently experience ambiguity regarding how their academic skills align with specialized industrial job roles. Traditional career guidance is manual, generic, and lacks quantitative depth. Our project addresses this challenge by combining multi-class supervised machine learning with a deterministic skill gap analysis engine."*

### ⏱️ Minute 0:50 – 1:40 | System Architecture & Dataset Design
*(Show Slide 4 & 5 or open dataset in CSV viewer)*
> *"Our architecture ingests 16 comprehensive student attributes—including core academic subject scores like Data Structures, Operating Systems, Computer Networks, and Machine Learning, alongside practical metrics such as LeetCode contest ratings, hackathons, projects, and certifications.*
>
> *We evaluated 2,400 student records across six balanced career tracks: AI & Machine Learning Engineer, Software Development Engineer (SDE), Cybersecurity Analyst, Cloud & DevOps Engineer, Data Engineer & DBA, and Full Stack Web Developer. Our preprocessing incorporates `StandardScaler` inside Scikit-learn `Pipeline` architectures to strictly eliminate data leakage."*

### ⏱️ Minute 1:40 – 3:15 | Live Execution & Model Benchmark Results
*(Switch screen to VS Code / Terminal or Jupyter Notebook and run `python student_skill_gap_career_recommendation.py`)*
> *"Let us execute the live Python pipeline. As you see in the terminal, the system trains six supervised classification models using 5-fold stratified cross-validation:*
>
> *1. Logistic Regression achieved our top performance with **92.29% test accuracy**, a cross-validation score of **92.14%**, and a multi-class One-vs-Rest ROC-AUC of **0.9931**.*
> *2. Gaussian Naive Bayes and Support Vector Machine also performed exceptionally well, achieving **92.29%** and **91.67%** test accuracies respectively.*
> *3. Random Forest achieved **91.25%** accuracy and identified Machine Learning, Networks, Cloud, and DSA as dominant discriminative features.*
>
> *Looking at our multi-class ROC curves, every single career track achieves an AUC greater than 0.988, proving near-perfect class separability."*

### ⏱️ Minute 3:15 – 4:15 | Skill Gap Analysis Engine Demonstration
*(Point to the generated Figure 9 chart and terminal gap output)*
> *"Here is our core innovation—the Skill Gap Analysis Engine. When evaluated on a candidate profile with strong Machine Learning and Math skills, the model predicted **AI & Machine Learning Engineer** with 99.99% confidence.*
>
> *The system then compared the student's score vector against the 75th-percentile benchmark for that role. It instantly identified actionable deficiencies: Cloud Computing (-7.6 pts), Computer Networks (-9.4 pts), and Database Systems (-5.7 pts). It generated a clear learning roadmap: recommending AWS Cloud Practitioner certification, Docker deployment, and advanced SQL indexing."*

### ⏱️ Minute 4:15 – 4:45 | Conclusion & Ethical AI Governance
> *"In conclusion, this project provides academic advisors and students with an objective, explainable, and proactive guidance system. Ethically, recommendations serve strictly as advisory decision support to empower students, never as automated gatekeepers.*
>
> *Thank you very much."*

---

## 🌐 How to Upload and Share Your Video Link

1. **YouTube (Recommended):**
   - Visit [YouTube Studio](https://studio.youtube.com).
   - Click **Create > Upload videos**.
   - Select visibility: **Unlisted** (anyone with the URL can view; invisible in public search).
   - Copy the link (e.g., `https://youtu.be/your-video-id`).

2. **Google Drive:**
   - Upload the recording to your Google Drive.
   - Right-click the video > **Share** > Change General Access to **"Anyone with the link can view"**.
   - Copy the shareable link.

3. **Update `README.md`:**
   - Open `README.md` and insert your actual URL:
   ```markdown
   [![Watch Project Demo Video](https://img.shields.io/badge/YouTube-Project%20Demo%20Video-red?logo=youtube)](https://youtu.be/YOUR_ACTUAL_VIDEO_LINK)
   ```
