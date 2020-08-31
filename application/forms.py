from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField,SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from application.models import User
import re
class LoginForm(FlaskForm):
    email   = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6,max=15)])
    remember_me = BooleanField("Remember Me")
    submit = SubmitField("Login")

class RegisterForm(FlaskForm):
    email   = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(),Length(min=6,max=15)])
    password_confirm = PasswordField("Confirm Password", validators=[DataRequired(),Length(min=6,max=15), EqualTo('password')])
    first_name = StringField("First Name", validators=[DataRequired(),Length(min=2,max=55)])
    last_name = StringField("Last Name", validators=[DataRequired(),Length(min=2,max=55)])
    designation = SelectField("Designation", validators=[DataRequired()],choices=["Student","Faculty","Administrator"])
    submit = SubmitField("Register Now")

    def validate_email(self,email):
        user = User.objects(email=email.data).first()
        regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]iiitm{1}[.]ac{1}[.]in$'
        if user:
            raise ValidationError("Email is already in use. Pick another one.")
        if re.search(regex,email.data):
            return
        else:
            raise ValidationError("Invalid Email,Try again with collegeId!")

class AddLocationForm(FlaskForm):
    title   = StringField("Name", validators=[DataRequired()])
    IPaddress = StringField("IPaddress", validators=[DataRequired()])
    description = StringField("Describe", validators=[DataRequired(),Length(min=2,max=145)])
    category = SelectField("Category", validators=[DataRequired()],choices=["Public Place","Lecture Hall","Faculty Room","Others"])
    submit = SubmitField("Add Now")
