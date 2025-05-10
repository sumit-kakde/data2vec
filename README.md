data2vec-Depression Severity Detection (audio-based)

This Streamlit app uses a fine-tuned [`facebook/data2vec-audio-base`](https://huggingface.co/facebook/data2vec-audio-base) model to classify depression severity based on a user's voice recording. It supports both multi-class (None, Mild, Moderate, Severe) and binary (Depressed/Not Depressed) classification.

---

 Features

 Upload `.wav`, `.mp3`, or `.ogg` audio files
 Choose between detailed severity detection or binary classification
 View predicted label with confidence scores
 Confidence bar chart
 Downloadable HTML report
Responsive and modern UI with custom CSS

---

License
This project is licensed under the MIT License.
---
Disclaimer
This tool provides a preliminary voice-based mental health analysis and does not replace professional medical evaluation.
Installation
---
Clone the repo
   ```bash
   git clone https://github.com/sumit-kakde/data2vec.git
   cd data2vec

Install dependencies
   pip install -r requirements.txt

 Run the app
   streamlit run depression_app.py







