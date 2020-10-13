from flask import Blueprint, render_template, abort


classifier = Blueprint('classifier', __name__,template_folder='templates/classifier',static_folder='static/classifier')

@classifier.route('/classifier')
def classifier_page():
	return render_template('classifier.html')
