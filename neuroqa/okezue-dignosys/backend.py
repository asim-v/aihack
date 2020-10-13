import framework


model_pneumoniaDetector = framework.model_pneumoniaDetector
model_covid19PneumoniaDetector = framework.model_covid19PneumoniaDetector

DIAGNOSE = [ "Pneumonia detected", "Covid19 detected", "Normal lungs detected", "Lung cancer detected" ]

def func_regularPneumonia (imagePath):
    test_data = []
    img = framework.cv2.imread(imagePath,0) 
    img = framework.cv2.resize(img, (framework.img_dims, framework.img_dims))
    img = framework.np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
    prediction = model_pneumoniaDetector.predict(framework.np.array(test_data))
    _prediction = round( prediction[0][0]*100, 3 )
    if ( _prediction > 50 ):
        _prediction = DIAGNOSE[0];
    elif ( _prediction < 50 ):
        _prediction = DIAGNOSE[2];  
    outputContent = _prediction + "\n"
    outputContent += "Raw Neural Network Output : " + str(prediction[0][0]) + ". These values demonstrate the accuracy of the model -Okezue Bell.\n\n"
    recordInferenceEvent (imagePath, outputContent)
    return outputContent


def func_covid19Pneumonia (imagePath):
    test_data = []
    img = framework.cv2.imread(imagePath,0)
    img = framework.cv2.resize(img, (framework.img_dims, framework.img_dims))
    img = framework.np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
    prediction = model_covid19PneumoniaDetector.predict(framework.np.array(test_data))
    _prediction = round( prediction[0][0]*100, 3 )
    if ( _prediction > 50 ):
        _prediction = DIAGNOSE[1];
    elif ( _prediction < 50 ):
        _prediction = DIAGNOSE[2];  
    outputContent = _prediction + "\n"
    outputContent += "Raw Neural Network Output : " + str(prediction[0][0]) + ". These values demonstrate the accuracy of the model -Okezue Bell.\n\n"
    recordInferenceEvent (imagePath, outputContent)
    return outputContent


def func_lungcancerPneumonia (imagePath):
    test_data = []
    img = framework.cv2.imread(imagePath,0)
    img = framework.cv2.resize(img, (framework.img_dims, framework.img_dims))
    img = framework.np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
    prediction = model_covid19PneumoniaDetector.predict(framework.np.array(test_data))
    _prediction = round( prediction[0][0]*100, 3 )
    if ( _prediction > 50 ):
        _prediction = DIAGNOSE[3];
    elif ( _prediction < 50 ):
        _prediction = DIAGNOSE[2];  
    outputContent = _prediction + "\n"
    outputContent += "Raw Neural Network Output : " + str(prediction[0][0]) + ". These values demonstrate the accuracy of the model -Okezue Bell.\n\n"
    recordInferenceEvent (imagePath, outputContent)
    return outputContent




import datetime
def recordInferenceEvent ( imagePath, outputContent ):
    currentDate = datetime.datetime.now()
    with open("inference_record.txt", "a") as text_file:
        text_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        text_file.write("DATE/TIME : " + str(currentDate.month) + " " + str(currentDate.day) + ", " + str(currentDate.year) + "..." + str(currentDate.hour) + ":" + str(currentDate.minute) + ":" + str(currentDate.second) + "\n\n") 
        text_file.write("IMAGE : " + imagePath + "\n\n")
        text_file.write("RESULT : \n" + outputContent + "\n\n\n\n")




