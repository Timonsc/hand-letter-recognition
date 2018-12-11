# MLCV Hand Recognition project 
> Timon Schneider 

#### Abstract
In this project I used OpenCV2 scripts to predict hand letters from live video. Using facedetect.cpp the program detects your face and saves a color palet of your skin color. Camshift.cpp then keeps following the face and covers the video so that another instance of camshift finds your hand because of the similarity in colors. That rectangle will be the input for the the trained MLP network that predicts the hand letter.

#### Results:
*	I trained the model to recognize a, b, c and d. 
*	I found out that the best precision on the test samples of my dataset can be reached with one hidden layer of 100 nodes. 100% precicion on the test samples of the abc.txt data set. The precision for live prediciction is also very high but only if the background colors are not too similar with my skin colors.
*  The model was trained with a method_param of 0.03.

#### Methods:
*	See code for more comments on how I did it.


#### Facedetect.cpp:
*	Maks sure to edit cascadeName on line 46, set the right folder for imagename on line 340 and set the name of the model on line 352.
*	Can be compiled using the following line: "sudo g++ facedetect.cpp -o facedetect -I /usr/local/include/ -L /usr/local/lib/ -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_ml -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_video"
*	If you want to predict your hand signs run ./facedetect. 
*	If you want to save samples to create a training set run with ./facedetect -letter=[lower case letter you want to create samples for]
*	Press 'x' if you want to save samples or predict your hand sign.

#### Letter_recog.cpp:
*	Can be compiled using the line: "sudo g++ letter_recog.cpp -o letter_recog -I /usr/local/include/ -L /usr/local/lib/ -lopencv_core -lopencv_ml"
*	Can be run with "./letter_recog -data=abc.txt -save=hallo.xml -load=hallo.xml" where 
		data is the file with samples, 
		save is the file name for the model that will be created and 
		load is the trained model (if you want to a data set)

#### abc.txt
*	The textfile with the samples will consist values with a lot of whitespaces.
*	Run "python replace.py" to transform textfile.txt into abc.txt to remove the whitespaces.
*	Then use a tool to shuffle the lines. (https://onlinerandomtools.com/shuffle-lines)
