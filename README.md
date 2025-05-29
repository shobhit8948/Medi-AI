🩺 Medicine Recommendation System with Symptom-Based Diagnosis & Blog Platform
This is a web-based application that allows users to input their symptoms and receive disease predictions, along with detailed descriptions, recommended medications, precautions, diets, and workout routines. Additionally, it includes a blog platform where users can share health-related content, comment, and interact with other users.

🔍 Features
🧠 Medical Prediction
Predicts possible diseases based on selected symptoms using an ML model.

Provides disease description, medication suggestions, precautionary steps, dietary recommendations, and workouts.

✍️ Blog System
User registration and login system.

Users can create, edit, and delete health blogs.

Like and dislike functionality with restriction to one action per user per blog.

Commenting system on each blog post.

Search functionality to find blogs by title or content.

🛠 Tech Stack
Backend: Python, Flask, SQLite, SQLAlchemy

Frontend: HTML, CSS, Jinja2

Machine Learning: Scikit-learn (SVC model)

Data Handling: Pandas, NumPy

Authentication: Secure password hashing via werkzeug.security


📦 Requirements
Python 3.7+

Flask

Flask-SQLAlchemy

Flask-Migrate

Pandas

NumPy

Scikit-learn

Werkzeug

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Note: Create a requirements.txt with necessary libraries.

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/medicine-recommendation-system.git
cd medicine-recommendation-system
Create the folders (if not present):

bash
Copy
Edit
mkdir static/uploads
Ensure the SQLite DB is initialized:

bash
Copy
Edit
flask db init
flask db migrate
flask db upgrade
Start the server:

bash
Copy
Edit
python app.py
Open http://127.0.0.1:5000/ in your browser.

📌 Usage
Navigate to the homepage to explore blogs or register/login.

Go to the "Health" section to input symptoms.

View diagnosis results with all recommended medical information.

Write and manage blogs to share health-related insights.

🧠 Model Info
The app uses a Support Vector Classifier (SVC) model trained on a multi-disease symptom dataset. Symptoms are one-hot encoded to predict disease labels.

✅ Future Improvements
Integrate real-time APIs for drug updates.

Add doctor consultation booking.

Deploy the app on a cloud platform (e.g., Heroku, Render, AWS).

📝 License
This project is licensed under the MIT License.