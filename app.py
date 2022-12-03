from flask import Flask, render_template, redirect, request
import os
import MakePredictions

# __name__ == __main__
app = Flask(__name__)


@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/', methods= ['POST'])
def marks():
	result_dic={}
	if request.method == 'POST':

		f = request.files['userfile']
		print("ffffffffffff",request.files['userfile'])
		path = "./static/{}".format(f.filename)# ./static/images.jpg
		f.save(path)
		
		
		file_name, file_extension = os.path.splitext(f.filename)
		grad_cam_path = "./static/{}".format(file_name+"_cam.jpg")

		prediction = MakePredictions.Predict(path)
		
		result_dic = {
		'image' : path,
		'grad_cam_image' : grad_cam_path,
		'caption' : prediction
		}

	return render_template("index.html", your_result =result_dic)

if __name__ == '__main__':
	# app.debug = True
	# due to versions of keras we need to pass another paramter threaded = Flase to this run function
	app.run(debug = True, threaded = False)
