import sys
#sys.path.append('../')
from application import app, db
from flask import render_template, request, json, Response, redirect, flash, url_for, session
from application.models import User, Location
from application.forms import LoginForm, RegisterForm, AddLocationForm
from application.detection_model import *
from application import *
import cv2

@app.route("/")
@app.route("/index")
@app.route("/home")
def index():
    return render_template("index.html", index=True )


##### USERCRUD ROUTES ######


#REGISTER
@app.route("/register", methods=['POST','GET'])
def register():
    if session.get('username'):
        return redirect(url_for('index'))
    form = RegisterForm()
    if form.validate_on_submit():
        user_id =User.objects.count()
        user_id +=1

        email =form.email.data
        password =form.password.data
        first_name =form.first_name.data
        last_name =form.last_name.data
        designation =form.designation.data
        user = User(user_id=user_id, email=email, first_name=first_name, last_name=last_name,designation=designation)
        user.set_password(password)
        user.save()
        flash("You are successfully registered!","success")
        session['user_id'] = user.user_id
        session['username'] = user.first_name
        session['designation'] = user.designation
        return redirect(url_for('index'))
    return render_template("register.html", title="Register", form=form, register=True)


#LOGIN
@app.route("/login", methods=['GET','POST'])
def login():
    if session.get('username'):
        return redirect(url_for('index'))

    form = LoginForm()
    if form.validate_on_submit():
        email =form.email.data
        password =form.password.data

        user = User.objects(email=email).first()
        if user and user.get_password(password):
            flash(f"{user.first_name}, you are successfully logged in!", "success")
            session['user_id'] = user.user_id
            session['username'] = user.first_name
            session['designation'] = user.designation
            return redirect("/index")
        else:
            flash("Sorry, something went wrong.","danger")
    return render_template("login.html", title="Login", form=form, login=True )


#LOGOUT
@app.route("/logout")
def logout():
    session['user_id']=False
    session.pop('username',None)
    session.pop('designation',None)
    return redirect(url_for('index'))


#########  LOCATIONCRUD  ###########

#SHOW LOCATIONS
@app.route("/locations/")
def location():
    classes = Location.objects.order_by("+locationID")
    return render_template("locations.html", locationData=classes, locations = True)

#ADD LOCATIONS
@app.route("/addlocation", methods=['POST','GET'])
def addlocation():

    if not session.get('username'):
        return redirect(url_for('login'))

    if session.get('designation') != "Administrator":
        return resirect(url_for('page_not_found'))
    
    form = AddLocationForm()
    if form.validate_on_submit():
        locationID =Location.objects.count()
        locationID +=1

        title =form.title.data
        IPaddress =form.IPaddress.data
        category =form.category.data
        description =form.description.data
        location = Location(locationID=locationID,title=title,IPaddress=IPaddress,category=category,description=description)
        location.save()
        flash("Location added successfully!","success")
        return redirect(url_for('index'))
    return render_template("addLocation.html", title="ADD LOCATION", form=form, addLocation=True)

#SPECIFIC LOCATION

def gen_frames(ip,category):  # generate frame by frame from camera
    camera=cv2.VideoCapture(ip)
    i=0
    while True:
        # Capture frame-by-frame
        if i%1000 == 0:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                ####
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                outputs = net.forward(layers)
                classIds,confidences,boxes=find_box_dimension(frame,outputs)
                nonMaxSup(frame,classIds,confidences,boxes,confThreshold,nmsThreshold)
                cv2.putText(frame,str(count),(5,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2,cv2.LINE_4) 
                ###
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        i=i+1
        i=i%1000
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    args = request.args
    return Response(gen_frames(args["ip"],args["category"]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/location/")
@app.route("/location/<idx>", methods=["GET","POST"])
def specific_location(idx=None):
    if not session.get('username'):
        return redirect(url_for('login'))

    if(idx == None):
        return redirect(url_for('page_not_found'))

    location=Location.objects(locationID=int(idx)).first()
    #print(location)
    return render_template("location.html",title=location.title,ip=location.IPaddress,category=location.category)    

@app.route("/user")
def user():
     #User(user_id=1, first_name="Christian", last_name="Hur", email="christian@uta.com", password="abc1234").save()
     #User(user_id=2, first_name="Mary", last_name="Jane", email="mary.jane@uta.com", password="password123").save()
     users = User.objects.all()
     return render_template("user.html", users=users)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404