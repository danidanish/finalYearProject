from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
from flask import Flask,render_template,url_for,request
from flask_material import Material
import MySQLdb.cursors
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import plotly
import plotly.graph_objs as go
import json
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
UPLOAD_FOLDER = 'static/uploads/'



app = Flask(__name__)
Material(app)


# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'

# Enter your database connection details below


# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)



# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        # cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = 1
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = 1
            session['username'] = "Rohit"
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('index.html', msg='')
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/pythonlogin/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        
        # User is loggedin show them the home page
        return render_template('home.html')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
@app.route('/pythonlogin/analyze', methods=['POST'])
def analyze():
	if request.method == 'POST':
		age = request.form['age']
		bp = request.form['bp']
		sg = request.form['sg']
		suger = request.form['suger']
		al = request.form['al']
		ba = request.form['ba']
		bu =request.form['bu']
		sc = request.form['sc']
		hemo = request.form['hemo']
		hype = request.form['hype']
		dm = request.form['dm']
		cad = request.form['cad']
		app = request.form['app']
		pe = request.form['pe']
		ane = request.form['ane']



		age2=0
		sc2=0
		bp2=0
		suger2=0
		hemo2=0
		# pot1=0
		
		
		sample_data = [age,bp,sg,suger,al,ba,bu,sc,hemo,hype,dm,cad,app,pe,ane]
		clean_data = [float(i) for i in sample_data]
		ex1 = np.array(clean_data).reshape(1,-1)

		age1=int(age)
		sc1=int(float(sc))
		bp1=int(bp)
		suger1=int(suger)
		hemo1=float(hemo)


		if age1<18 or age1>50:
			age2=age1
		elif sc1>2:
			sc2=sc1
		elif bp1>120:
			bp2=bp1
		elif suger1>120:
			suger2=suger1
		elif hemo1<13.5:
			hemo2=hemo1


		data = pd.read_csv('chronic.csv')
		GFR=175*sc1**(-1.154)*age1**(-0.203)*(0.742)

		data=data.drop(columns=['rbc','pc','pcc','bgr','sod','pot','pcv','wbcc','rbcc'])
		data=data.replace(['ckd','notckd','yes','no','good','poor','present','notpresent','abnormal','normal'],[1,0,1,0,1,0,1,0,0,1])

		
		data.dropna(inplace=True)
		X1=data.drop(columns='class')
		y1 = data['class'].values
		X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25)
		
		dt=tree.DecisionTreeClassifier()
		dt.fit(X_train1,y_train1)

		gnb = GaussianNB()
		gnb.fit(X_train1, y_train1)

		knn = KNeighborsClassifier(n_neighbors = 3)
		knn.fit(X_train1,y_train1)

		result=dt.predict(ex1)
		if result==1:
			class1=" Chronic Kidney Disease"
		else:
			class1="No Chronic Kidney Disease "

		Y_pred_dt =dt.predict(X_test1)
		Y_pred_dt.shape
		score_dt = round(accuracy_score(Y_pred_dt,y_test1)*100,2)	

		result=gnb.predict(ex1)
		result1=result
		if result==1:
			class2=" Chronic Kidney Disease"
		else:
			class2="No Chronic Kidney Disease "
		Y_pred_nb =gnb.predict(X_test1)
		Y_pred_nb.shape
		score_nb = round(accuracy_score(Y_pred_nb,y_test1)*100,2)	

		result=knn.predict(ex1)
		if result==1:
			class3="Chronic Kidney Disease"
		else:
			class3="No Chronic Kidney Disease "
		Y_pred_knn =knn.predict(X_test1)
		Y_pred_knn.shape
		score_knn = round(accuracy_score(Y_pred_knn,y_test1)*100,2)	

		if GFR > 90:
			stage="Person in Normal Stage"
		if GFR >=60 and GFR <=89:
			stage="Person in Mild Stage"
		if GFR >=30 and GFR <=59:
			stage="Person in Moderate Stage (stage 1)"
		if GFR >=15 and GFR <=29:
			stage="Person in Sever Stage(Stage 2)"
		if GFR <15:
			stage="Kidney Failure(Stage 3)"

	return render_template('home.html',age=age,bp=bp,sg=sg,al=al,ba=ba,bu=bu,sc=sc,suger=suger,hemo=hemo,hype=hype,dm=dm,cad=cad,app=app,pe=pe,ane=ane,age2=age2,sc2=sc2,bp2=bp2,suger2=suger2,hemo2=hemo2,class1=class1,score_dt=score_dt,class2=class2,score_nb=score_nb,score_knn=score_knn,class3=class3,GFR=GFR,result1=result1,stage=stage)



@app.route('/pythonlogin/preview', methods=['GET', 'POST'])
def preview():
		feature = 'Bar'
		bar = create_plot(feature)
		return render_template('preview.html', plot=bar)
def create_plot(feature):
		data = pd.read_csv('chronic.csv')
		data=data.drop(columns=['rbc','pc','pcc','bgr','sod','pot','pcv','wbcc','rbcc'])
		data=data.replace(['ckd','notckd','yes','no','good','poor','present','notpresent','abnormal','normal'],[1,0,1,0,1,0,1,0,0,1])

		
		data.dropna(inplace=True)
		X1=data.drop(columns='class')
		y1 = data['class'].values
		X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25)


		dt=tree.DecisionTreeClassifier()
		dt.fit(X_train1,y_train1)
		Y_pred_dt =dt.predict(X_test1)
		Y_pred_dt.shape
		score_dt = round(accuracy_score(Y_pred_dt,y_test1)*100,2)	


		nb = GaussianNB()
		nb.fit(X_train1,y_train1)
		Y_pred_nb = nb.predict(X_test1)
		Y_pred_nb.shape
		score_nb = round(accuracy_score(Y_pred_nb,y_test1)*100,2)

		

		knn = KNeighborsClassifier(n_neighbors=7)
		knn.fit(X_train1,y_train1)
		Y_pred_knn=knn.predict(X_test1)
		Y_pred_knn.shape
		score_knn = round(accuracy_score(Y_pred_knn,y_test1)*100,2)

		

		data = [
			go.Bar(
				# assign x as the dataframe column 'x'
				x=["Decision Tree","Naive Bayes","K-Nearest Neighbors"],
				y=[score_dt,score_nb,score_knn]
				)
			]
		graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
		return graphJSON
@app.route('/bar', methods=['GET', 'POST'])
def change_features():

	feature = request.args['selected']
	graphJSON= create_plot(feature)

	return graphJSON

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
@app.route('/pythonlogin/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
       
        
        
                
            
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        
        account = cursor.fetchone()
        cursor1 = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor1.execute('SELECT * FROM predictiondetails WHERE username = %s', (session['username'],))
        prediction_details = cursor1.fetchall()
        # Show the profile page with account info
        return render_template('profile.html', account=account,prediction_details = prediction_details)
    # User is not loggedin redirect to login page
    return redirect(url_for('login')) 
    



    





if __name__ =='__main__':
	app.run()
