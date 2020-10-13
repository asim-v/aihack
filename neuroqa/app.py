from flask import Flask,render_template,send_from_directory

app = Flask('__name__')

@app.route('/')
def index():
	return render_template('index.html')

##STATIC
@app.route('/assets/js/<path:path>')
def send_js(path):
	return send_from_directory('static/js', path)
@app.route('/assets/images/<path:path>')
def send_img(path):
	return send_from_directory('static/images', path)
@app.route('/assets/css/<path:path>')
def send_css(path):
	return send_from_directory('static/css', path)
@app.route('/assets/fonts/<path:path>')
def send_fonts(path):
	return send_from_directory('static/fonts', path)
@app.route('/assets/favicons/<path:path>')
def send_icons(path):
	return send_from_directory('static/favicons', path)


if __name__ == '__main__':
	app.run(debug=True)