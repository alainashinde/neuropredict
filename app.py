from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
import numpy as np
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# load trained model
model = joblib.load('best_svm_model.pkl')

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict-form', methods=['GET'])
@login_required
def predict_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # input values from form
        srs_raw_total = float(request.form['SRS_RAW_TOTAL'])
        srs_awareness = float(request.form['SRS_AWARENESS'])
        srs_cognition = float(request.form['SRS_COGNITION'])
        srs_communication = float(request.form['SRS_COMMUNICATION'])
        srs_motivation = float(request.form['SRS_MOTIVATION'])
        srs_mannerisms = float(request.form['SRS_MANNERISMS'])

        # prep input array for prediction
        input_data = np.array([[srs_raw_total, srs_awareness, srs_cognition, srs_communication, srs_motivation, srs_mannerisms]])

        # make prediction
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0]

        # output in readable format
        adhd_prediction = 'True' if prediction == 1 else 'False'
        confidence_score = confidence[prediction] * 100

        return render_template('result.html', prediction=adhd_prediction, confidence=confidence_score)

    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('predict_form'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
