#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp" // headers that have information of the methods input and output parameters
#include <iostream> 

#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio.hpp"
#include <fstream>
#include <cstdio>
#include <vector>
#include <iostream>
#include <ctime>

/* Compile command
sudo g++ facedetect.cpp -o facedetect -I /usr/local/include/ -L /usr/local/lib/ -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_ml -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_video
*/
using namespace std;
using namespace cv;
using namespace cv::ml;

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale, vector<Rect>& faces); // Declaration of the functions

void camShift(Mat& image, Rect selection, Rect newTrackWindow, int framecounter, String letter, bool training);

template<typename T> static Ptr<T> load_classifier(const string& filename_to_load);

string cascadeName = "/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";

int main( int argc, const char** argv )
{
    VideoCapture capture;       // Capital variable name means it's an object of a class
    Mat frame, image;           // matrice
    string inputName;
    String letter = "";         //
    //bool tryflip;    
    bool training = false;           
    CascadeClassifier cascade;  // , nestedCascade cascade boosting. has something to do with boosting, see first lecture about boosting
    double scale;               //
    Rect newTrackWindow;        //
    vector<int> returns;        //
    int framecounter;           // Frame counter that is used to give the sample picture jpeg captures of the hand unique names. 
    vector<Rect> faces;         // Vector of rectangles that is going to be filled with the faces recognized by the detectAndDraw function. 

    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{scale|1|}{@filename||}"
        "{letter||}"
    );
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    inputName = parser.get<string>("@filename");
    letter = parser.get<string>("letter");
    if( !letter.empty()){       // If the command line consists of the command 'letter', the program will save matrixes in the given textfile to use for training later. Otherwise the program will just predict.
        bool training = true;
        letter = parser.get<string>("letter");
        cout << "Training for letter '" << letter << "' started. Press 'x' to capture samples." << endl;
    }
    else{
        cout << "Prediction of hand signs. Press 'x' to predict you hand sign. Use your right hand." << endl;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    //if ( !nestedCascade.load( nestedCascadeName ) )
    //    cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) ) // no cascade names but try the camera anyway.
    {
        int camera = inputName.empty() ? 0 : inputName[0] - '0';
        if(!capture.open(camera))
            cout << "Capture from camera #" <<  camera << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            if(!capture.open( inputName ))
                cout << "Could not read " << inputName << endl;
        }
    }
    else
    {
        image = imread( "../data/lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read ../data/lena.jpg" << endl;
    }
    
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        
        for(;;) // infinite loop, no conditions
        {
            framecounter++; 
            capture >> frame; // Store capture in frame matrix
            if( frame.empty() )
                break;
            Mat frame1 = frame.clone();
            if(faces.empty()){
                detectAndDraw( frame1, cascade, scale, faces ); // If there are no recognized faces in faces, launch detectAndDraw to recognize a face. , tryflip
            }
            else{
                camShift(frame1, faces[0], newTrackWindow, framecounter,letter, training);      // Run Camshift to keep track of the recognized face without having to run detectAndDraw all the time.
            }
            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
    else // Detecting faces from images in directory given in command line using @filename
    {
        cout << "Detecting face(s) in " << inputName << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, scale, faces); //, tryflip
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf);
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        detectAndDraw( image, cascade, scale, faces);  //, tryflip
                        char c = (char)waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;

                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    return 0;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale, vector<Rect>& faces) // , CascadeClassifier& nestedCascade , bool tryflip
{
    double t = 0;
    vector<Rect> faces2; //vector<Rect> faces, faces2; move faces to line 156
    const static Scalar colors[] =
    { 
        Scalar(255,0,0), Scalar(255,128,0),Scalar(255,255,0),Scalar(0,255,0),Scalar(0,128,255),Scalar(0,255,255),Scalar(0,0,255),Scalar(255,0,255) // colors in BGR, blue green red
    }; 
    Mat gray, smallImg; 

    cvtColor( img, gray, COLOR_BGR2GRAY ); // convert colors, from BGR to grey 
    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR ); //linear interpolation, resize. so two or more pixels will become one pixel with the average color.
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount(); // launch the clock
    cascade.detectMultiScale( smallImg, faces, // launch the ada boost 
        1.1, 2, 0
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );

    t = (double)getTickCount() - t;
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i]; // store the recognized faces in rectangle r
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 ) // Size ratio conditions for the rectangle to be a face , if the rectangle is too wide it's probably not a face
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 ); // Put a circle around the face.
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), 
                cvRound(r.y*scale)),
                cvPoint(cvRound((r.x + r.width-1)*scale), 
                cvRound((r.y + r.height-1)*scale)),color, 3, 8, 0);
    }
    imshow( "result", img );
}

template<typename T> static Ptr<T> load_classifier(const string& filename_to_load) // Loading the classifier 
{
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";
    return model;
}


//vector<int> 
void camShift(Mat& image, Rect selection, Rect newTrackWindow, int framecounter, String letter, bool training)
{
    Mat hsv, hue, hist, mask, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

    // COUNTING THE NUMBER OF HAND PICTURES MADE
    int vmin = 10, vmax = 256, smin = 30;
    int trackObject = -1;
    //bool selectObject = false;
    bool showHist = true;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    Rect trackWindow;
    Point origin; // two variables.
    cvtColor(image, hsv, COLOR_BGR2HSV); // Convert the color space into hsv.
    
    int _vmin = vmin, _vmax = vmax;
    inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),Scalar(180, 256, MAX(_vmin, _vmax)), mask);
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    if( trackObject < 0 )
    {
                         
        // Object has been selected by user, set up CAMShift search properties once
        Mat roi(hue, selection), maskroi(mask, selection); 
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges); // Calculate the histogram of the selection that will used to follow the face.
        normalize(hist, hist, 0, 255, NORM_MINMAX);
        //cout << "3: trackobject <0 " << endl;
        trackWindow = selection;
        trackObject = 1; 

        histimg = Scalar::all(0);
        int binW = histimg.cols / hsize;
        Mat buf(1, hsize, CV_8UC3);
        for( int i = 0; i < hsize; i++ )
            buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
        cvtColor(buf, buf, COLOR_HSV2BGR);
        for( int i = 0; i < hsize; i++ )
        {
            int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
            rectangle( histimg, Point(i*binW,histimg.rows),Point((i+1)*binW,histimg.rows - val),
                       Scalar(buf.at<Vec3b>(i)), -1, 8 );
        }
    }

    // Perform CAMShift
    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    backproj &= mask;
    RotatedRect trackBox = CamShift(backproj, trackWindow,TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 )); // The selection of the face.
    Point2f vertices[4];
    trackBox.points(vertices); // Extracting the corner points of the rotated trackBox rotated Rect. I took a bigger rectangle around the retated rectangle to make sure the whole face is covered.
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0)); // make a vertice for every point that will be combined into a rectangle
    Rect nonRotatedRect = trackBox.boundingRect(); // Create a rectangle of the vertices.
    nonRotatedRect.height = nonRotatedRect.height + nonRotatedRect.y;
    rectangle(backproj, nonRotatedRect, Scalar(0,0,0),CV_FILLED); // Draw filled black rectangle around the rotated rect in backProj so that the camshift will look for other objects with the same color distribution as the face >> the hand.
    rectangle(image, nonRotatedRect, Scalar(0,56,144),2); // Draw the rectangle around the face/rotated rectanlg in image.
    newTrackWindow = Rect(0, trackWindow.y, trackWindow.x, trackWindow.height); // Create the new track window to start the camshift from. This is for the user on the right side of their face.
    RotatedRect hand = CamShift(backproj, newTrackWindow,TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 )); // Find the hand by using camshift.
    rectangle(image, newTrackWindow, Scalar(0,150,0),5); // Draw the rectangle around the hand.

    if( trackWindow.area() <= 1 )
    {
        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,trackWindow.x + r, trackWindow.y + r) &
        Rect(0, 0, cols, rows);
    }

    ellipse( image, hand, Scalar(0,0,255), 3, LINE_AA );

    imshow("CamShift Demo", image);
    imshow("Histogram", histimg);
    imshow("BackProject", backproj);

    char c = (char)waitKey(10);
    
    if( c == 'x')
    {    
        Mat hand, hand1616;
        int handx = newTrackWindow.x-25; // Create a border around the trackwindow to prevent cutting off fingers for example.
        if(handx<0){handx=0;} // Prevent reaching for pixels out of the backproj matrice (if the netTrackWindow starts at 0 the border will reach for -25)
        int handy = newTrackWindow.y-25;
        if(handy<0){handy=0;}
        int handwidth = newTrackWindow.width+50;
        int handheight = newTrackWindow.height+50;
        backproj(Rect(handx, handy, handwidth, handheight)).copyTo(hand);
        resize(hand, hand1616, Size(16, 16), 0, 0, INTER_CUBIC);
        hand1616.convertTo(hand1616,CV_32FC1);
        Mat img = hand1616.reshape(0,1);

        if(training)
        {
            String imagename =  "/home/user/mlcv/Saved_images/hand_"+ letter +"_" + to_string(framecounter)+".jpg"; // can be used to save the text in different files
            imshow("new hand", hand1616);
            ofstream os("textfile.txt", ios::out | ios::app); // Save the samples on new lines in textfile.txt
            os << letter+",";
            os << format(img, Formatter::FMT_CSV ) << endl;
            os.close(); 
            imwrite(imagename,hand1616);
            cout << "Sample saved in 'textfile.txt' and " << imagename << " stored in Saved_images" << endl;
        }
        
        else{
            Ptr<ANN_MLP> model;
            string filename_to_load = "hallo.xml";
            model = load_classifier<ANN_MLP>(filename_to_load);

            float r = model->predict( img ); // Predict the selected hand >> img.
            int prediction = static_cast<int>(r);
            int a = 'a';
            prediction = prediction + a; // Find the ascii code of the letter that is actually predicted
            cout << "Hand sign prediction: " << char(prediction) << endl;
        }

    }
}
