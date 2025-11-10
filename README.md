AI Image Caption Generator

A Streamlit web app that generates captions for any image you upload using AI. Powered by the Hugging Face nlpconnect/vit-gpt2-image-captioning model, it can describe your images in natural language.

🚀 Demo:

(Replace with your deployed Streamlit link once live)
👉 Live App on Streamlit Cloud

💡 Features

🖼️ Upload JPG, JPEG, PNG, or WEBP images

🤖 AI automatically generates descriptive captions for your images

✅ Displays uploaded image alongside generated caption

📝 Copy or view caption easily

🧰 Tech Stack

Python 3.9+

Streamlit — Web UI

Transformers (pipeline) — Hugging Face image-to-text model

Pillow (PIL) — Image handling

⚙️ Installation

Clone the repository

git clone https://github.com/preyeandaye3008-hash/AI-Image-Caption-Generator.git


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run "AI Image Caption Generator.py"


Open your browser and visit 👉 http://localhost:8501

🧾 Requirements File (requirements.txt)
streamlit
transformers
torch
Pillow


Make sure you have torch installed, as it’s required by the Hugging Face model.

🔑 How It Works

Upload an image in the app.

The model nlpconnect/vit-gpt2-image-captioning generates a descriptive caption.

The app displays the image and the AI-generated caption instantly.

🧑‍💻 Author

Andaye Preye
🔗 [GitHub Profile] (https://github.com/preyeandaye3008-hash)

🪄 License

MIT License — free to use, modify, and share.
