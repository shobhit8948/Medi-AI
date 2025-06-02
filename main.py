from flask import Flask, render_template, request, redirect, url_for, session, jsonify,flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import numpy as np
import pandas as pd
import pickle

# In app.py, modify the Flask app initialization:
app = Flask(__name__, static_url_path='/static')  # Add this
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    mobile = db.Column(db.String(15), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Blog model
class Blog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image = db.Column(db.String(300))
    likes = db.Column(db.Integer, default=0)
    dislikes = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('blogs', lazy=True))

# Comment model
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    blog_id = db.Column(db.Integer, db.ForeignKey('blog.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Add this line
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('comments', lazy=True))  # Add this relationship

# LikeDislike model to restrict user to like/dislike once
class LikeDislike(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    blog_id = db.Column(db.Integer, db.ForeignKey('blog.id'), nullable=False)
    is_like = db.Column(db.Boolean, nullable=False)  # True for like, False for dislike
    __table_args__ = (db.UniqueConstraint('user_id', 'blog_id', name='unique_user_blog'),)

with app.app_context():
    db.create_all()

# load databasedataset===================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# load model===========================================
svc = pickle.load(open('models/svc.pkl','rb'))

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    
    # Convert to DataFrame with proper feature names to avoid warning
    feature_names = list(symptoms_dict.keys())
    input_df = pd.DataFrame([input_vector], columns=feature_names)
    return diseases_list[svc.predict(input_df)[0]]

def format_symptom_display(symptom):
    """Format symptom for display (capitalize first letters, replace underscores)"""
    # Handle special cases first
    special_cases = {
        'gerd': 'GERD',
        'aids': 'AIDS',
        'hiv': 'HIV'
    }
    
    lower_symptom = symptom.lower()
    if lower_symptom in special_cases:
        return special_cases[lower_symptom]
    
    # Replace underscores with spaces and capitalize each word
    return ' '.join(word.capitalize() for word in symptom.split('_'))

@app.route('/')
def index():
    blogs = Blog.query.order_by(Blog.created_at.desc()).all()
    symptoms_list = [(symptom, format_symptom_display(symptom)) 
                    for symptom in sorted(symptoms_dict.keys())]
    return render_template('index.html', blogs=blogs, symptoms_list=symptoms_list)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        mobile = request.form['mobile']
        password = generate_password_hash(request.form['password'])

        if User.query.filter((User.username == username) | (User.email == email) | (User.mobile == mobile)).first():
            return 'User already exists'

        user = User(name=name, username=username, email=email, mobile=mobile, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['name'] = user.name
            return redirect('/')
        return 'Invalid credentials'

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    # Basic search implementation - you might want to use a proper search engine
    if query:
        blogs = Blog.query.filter(
            (Blog.title.ilike(f'%{query}%')) | 
            (Blog.content.ilike(f'%{query}%'))
        ).order_by(Blog.created_at.desc()).all()
    else:
        blogs = []
    return render_template('search_results.html', blogs=blogs, query=query)

# Add these near the top of app.py
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/new', methods=['GET', 'POST'])
def new_blog():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        image_file = request.files['image']
        image_filename = None
        
        if image_file and image_file.filename != '':
            if not allowed_file(image_file.filename):
                flash('Invalid file type', 'error')
                return redirect(request.url)
                
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            
            try:
                image_file.save(image_path)
                # Store with static prefix
                image_url = url_for('static', filename=f'uploads/{image_filename}')
            except Exception as e:
                flash('Error saving image file', 'error')
                return redirect(request.url)

        new_blog = Blog(
            user_id=session['user_id'],
            title=title,
            content=content,
            image=image_url if image_filename else None
        )
        db.session.add(new_blog)
        db.session.commit()
        flash('Blog created successfully!', 'success')
        return redirect('/')

    return render_template('new_blog.html')

@app.route('/edit/<int:blog_id>', methods=['GET', 'POST'])
def edit_blog(blog_id):
    blog = Blog.query.get_or_404(blog_id)
    if 'user_id' not in session or blog.user_id != session['user_id']:
        return 'Unauthorized', 403

    if request.method == 'POST':
        blog.title = request.form['title']
        blog.content = request.form['content']
        
        # Handle image removal
        if 'remove_image' in request.form and request.form['remove_image'] == 'on':
            if blog.image:
                try:
                    os.remove(os.path.join('static', blog.image))
                except:
                    pass
            blog.image = None
        
        # Handle new image upload
        image_file = request.files['image']
        if image_file and image_file.filename != '':
            if not allowed_file(image_file.filename):
                flash('Invalid file type', 'error')
                return redirect(request.url)
                
            # Delete old image if exists
            if blog.image:
                try:
                    os.remove(os.path.join('static', blog.image))
                except:
                    pass
                    
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            
            try:
                image_file.save(image_path)
                blog.image = url_for('static', filename=f'uploads/{image_filename}')
            except Exception as e:
                flash('Error saving image file', 'error')
                return redirect(request.url)
        
        db.session.commit()
        flash('Blog updated successfully!', 'success')
        return redirect('/myblogs')

    return render_template('edit_blog.html', blog=blog)

@app.route('/blog/<int:blog_id>', methods=['GET', 'POST'])
def blog_detail(blog_id):
    blog = Blog.query.get_or_404(blog_id)
    if request.method == 'POST':
        comment_content = request.form['comment']
        if comment_content and 'user_id' in session:
            comment = Comment(
                blog_id=blog_id,
                user_id=session['user_id'],  # Add user_id
                content=comment_content
            )
            db.session.add(comment)
            db.session.commit()
        return redirect(url_for('blog_detail', blog_id=blog_id))

    comments = Comment.query.filter_by(blog_id=blog_id).order_by(Comment.created_at.desc()).all()
    return render_template('blog_detail.html', blog=blog, comments=comments)

@app.route('/like/<int:blog_id>')
def like_blog(blog_id):
    if 'user_id' not in session:
        return redirect('/login')

    existing = LikeDislike.query.filter_by(user_id=session['user_id'], blog_id=blog_id).first()
    blog = Blog.query.get_or_404(blog_id)

    if existing:
        if existing.is_like:
            return redirect(url_for('blog_detail', blog_id=blog_id))  # Already liked
        else:
            existing.is_like = True
            blog.likes += 1
            blog.dislikes -= 1
            db.session.commit()
    else:
        blog.likes += 1
        db.session.add(LikeDislike(user_id=session['user_id'], blog_id=blog_id, is_like=True))
        db.session.commit()

    return redirect(url_for('blog_detail', blog_id=blog_id))

@app.route('/unlike/<int:blog_id>')
def unlike_blog(blog_id):
    if 'user_id' not in session:
        return redirect('/login')

    existing = LikeDislike.query.filter_by(user_id=session['user_id'], blog_id=blog_id).first()
    blog = Blog.query.get_or_404(blog_id)

    if existing:
        if not existing.is_like:
            return redirect(url_for('blog_detail', blog_id=blog_id))  # Already disliked
        else:
            existing.is_like = False
            blog.likes -= 1
            blog.dislikes += 1
            db.session.commit()
    else:
        blog.dislikes += 1
        db.session.add(LikeDislike(user_id=session['user_id'], blog_id=blog_id, is_like=False))
        db.session.commit()

    return redirect(url_for('blog_detail', blog_id=blog_id))

@app.route('/health')
def health():
    symptoms_list = [(symptom, format_symptom_display(symptom)) 
                    for symptom in sorted(symptoms_dict.keys())]
    return render_template("health.html", symptoms_list=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    
    if not selected_symptoms:
        # Create the proper tuple list for symptoms when showing error
        symptoms_list = [(symptom, format_symptom_display(symptom)) 
                        for symptom in sorted(symptoms_dict.keys())]
        return render_template('health.html', 
                            symptoms_list=symptoms_list,
                            message="Please select at least one symptom")
    
    predicted_disease = get_predicted_value(selected_symptoms)
    dis_des, precautions_list, medications_list, rec_diet, workout_list = helper(predicted_disease)

    my_precautions = []
    for i in precautions_list[0]:
        my_precautions.append(i)

    # Create the proper tuple list for symptoms when showing results
    symptoms_list = [(symptom, format_symptom_display(symptom)) 
                    for symptom in sorted(symptoms_dict.keys())]
    
    return render_template('health.html', 
                         predicted_disease=predicted_disease, 
                         dis_des=dis_des,
                         my_precautions=my_precautions, 
                         medications=medications_list, 
                         my_diet=rec_diet,
                         workout=workout_list,
                         symptoms_list=symptoms_list)

@app.route('/myblogs')
def my_blogs():
    if 'user_id' not in session:
        return redirect('/login')

    blogs = Blog.query.filter_by(user_id=session['user_id']).order_by(Blog.created_at.desc()).all()
    return render_template('myblogs.html', blogs=blogs)

@app.route('/delete/<int:blog_id>')
def delete_blog(blog_id):
    blog = Blog.query.get_or_404(blog_id)
    if 'user_id' not in session or blog.user_id != session['user_id']:
        return 'Unauthorized', 403
    db.session.delete(blog)
    db.session.commit()
    return redirect('/myblogs')

if __name__ == '__main__':
    app.run(debug=True)
