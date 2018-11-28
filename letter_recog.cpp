#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

#include <cstdio>
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;
using namespace cv::ml;


/*This code will be used to train the network based upon the textfile with the letters, 
backup weights in network_backup file to input the weights into facedetect where we are 
going to use the weights to predict the letters. Some lines of this code have to be copied 
to facedetect to predict the letter once we have trained the network.
*/

static void help()
{
    printf("\nThe sample demonstrates how to train Random Trees classifier\n"
    "(or Boosting classifier, or MLP, or Knearest, or Nbayes, or Support Vector Machines - see main()) using the provided dataset.\n"
    "\n"
    "We use the sample database letter-recognition.data\n"
    "from UCI Repository, here is the link:\n"
    "\n"
    "Newman, D.J. & Hettich, S. & Blake, C.L. & Merz, C.J. (1998).\n"
    "UCI Repository of machine learning databases\n"
    "[http://www.ics.uci.edu/~mlearn/MLRepository.html].\n"
    "Irvine, CA: University of California, Department of Information and Computer Science.\n"
    "\n"
    "The dataset consists of 20000 feature vectors along with the\n"
    "responses - capital latin letters A..Z.\n"
    "The first 16000 (10000 for boosting)) samples are used for training\n"
    "and the remaining 4000 (10000 for boosting) - to test the classifier.\n"
    "======================================================\n");
    printf("\nThis is letter recognition sample.\n"
            "The usage: letter_recog [-data=<path to letter-recognition.data>] \\\n" // argument -data=[]
            "  [-save=<output XML file for the classifier>] \\\n"
            "  [-load=<XML file with the pre-trained classifier>] \\\n"
            "  [-boost|-mlp|-knearest|-nbayes|-svm] # to use boost/mlp/knearest/SVM classifier instead of default Random Trees\n" );
}

// This function reads data and responses from the file <filename>
static bool
read_num_class_data( const string& filename, int var_count,
                     Mat* _data, Mat* _responses ) // Read the training samples.
{
    const int M = 1024;
    char buf[M+2];

    Mat el_ptr(1, var_count, CV_32F);
    int i;
    vector<int> responses;

    _data->release();
    _responses->release();

    FILE* f = fopen( filename.c_str(), "rt" );
    if( !f )
    {
        cout << "Could not read the database " << filename << endl;
        return false;
    }

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        responses.push_back((int)buf[0]);
        ptr = buf+2;
        for( i = 0; i < var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
            ptr += n + 1;
        }
        if( i < var_count )
            break;
        _data->push_back(el_ptr);
    }
    fclose(f);
    Mat(responses).copyTo(*_responses);

    cout << "The database " << filename << " is loaded.\n";

    return true;
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load) // Load the classifier.
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";

    return model;
}

static Ptr<TrainData>
prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
    Mat sample_idx = Mat::zeros( 1, data.rows, CV_8U );
    Mat train_samples = sample_idx.colRange(0, ntrain_samples);
    train_samples.setTo(Scalar::all(1));

    int nvars = data.cols;
    Mat var_type( nvars + 1, 1, CV_8U );
    var_type.setTo(Scalar::all(VAR_ORDERED));
    var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

    return TrainData::create(data, ROW_SAMPLE, responses,
                             noArray(), sample_idx, noArray(), var_type);
}

inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void test_and_save_classifier(const Ptr<StatModel>& model,
                                     const Mat& data, const Mat& responses,
                                     int ntrain_samples, int rdelta,
                                     const string& filename_to_save) // rdelta is the difference if ascii values. From a to b the difference is exactly 1 integer. a to c is 2
{
    int i, nsamples_all = data.rows;
    double train_hr = 0, test_hr = 0;

    // Compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i); // the equivalent of img mat in facedetect around line 411 in facedetect.
        
        float r = model->predict( sample );
        r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f; // Find the precision of the samples.
        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= nsamples_all - ntrain_samples;
    train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;

    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    if( !filename_to_save.empty() )
    {
        cout << "Saving model to " << filename_to_save << endl;
        model->save( filename_to_save );
    }
}

static bool 
build_mlp_classifier( const string& data_filename, const string& filename_to_save, const string& filename_to_load)
{
    const int class_count = 4; // Amount of different letters of the in my dataset. I tried to work with the whole alphabet but did not get a high precision.
    Mat data;
    Mat responses;
    bool ok = read_num_class_data( data_filename, 256, &data, &responses ); // 256 is the amount of features of the input
    if( !ok )
        return ok;
    Ptr<ANN_MLP> model; // Declaring the model

    int nsamples_all = data.rows; // Number of rows in textfile
    cout << ": Samples recognized: " << nsamples_all << endl;
    int ntrain_samples = (int)(nsamples_all*0.8);
    cout << ": Training samples: "<< ntrain_samples<<endl;
    
    // Create or load MLP classifier
    if( !filename_to_load.empty() ) // If the load parameter is used, just predict
    {
        cout << ": Loading model."<< endl;
        model = load_classifier<ANN_MLP>(filename_to_load); // used to load the model! transfer to facedetect 
        if( model.empty() )
            return false;
        ntrain_samples = 0;
    }
    else // Otherwise train and create a new model!
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // MLP does not support categorical variables by explicitly.
        // So, instead of the output class label, we will use
        // a binary vector of <class_count> components for training and,
        // therefore, MLP will give us a vector of "probabilities" at the
        // prediction stage
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Mat train_data = data.rowRange(0, ntrain_samples); 
        Mat train_responses = Mat::zeros( ntrain_samples, class_count, CV_32F ); // 

        // 1. unroll the responses
        cout <<": Unrolling the responses...\n";
        for( int i = 0; i < ntrain_samples; i++ ) // For all the train samples in the text file, associate the train samples with its labels
        {
            int cls_label = responses.at<int>(i) - 'a';
            train_responses.at<float>(i, cls_label) = 1.f;
        }

        // 2. train classifier
        int layer_sz[] = { data.cols, 100, class_count }; // Found out that the best precision on the test samples of my dataset can be reached with one hidden layer of 100 nodes. 100% precicion on the test samples.
        int nlayers = (int)(sizeof(layer_sz)/sizeof(layer_sz[0]));
        Mat layer_sizes( 1, nlayers, CV_32S, layer_sz );

        #if 1
            int method = ANN_MLP::BACKPROP;
            double method_param = 0.03; // First I set this parameter proportionally to the amount of samples I had compared to the dataset that was used in this code. Then I tweaked it to improve.
            int max_iter = 300;
        #else
            int method = ANN_MLP::RPROP;
            double method_param = 0.1;
            int max_iter = 1000;
        #endif

            Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

            cout << ": Training the classifier (may take a few minutes)...\n"; // Creating the model.
            model = ANN_MLP::create(); 
            model->setLayerSizes(layer_sizes);
            model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
            model->setTermCriteria(TC(max_iter,0));
            model->setTrainMethod(method, method_param);
            model->train(tdata);
            cout << ": Training finished." << endl;
        
    }

    test_and_save_classifier(model, data, responses, ntrain_samples, 'a', filename_to_save); // file_name_to_save has to be set as backupNetwork.txt (xml file) and will be used to predict in facedetect. 
    return true;
}

int main( int argc, char *argv[] )
{
    string filename_to_save = "/home/user/mlcv/backupNetwork.txt"; 
    string filename_to_load = ""; 
    string data_filename;
    int method = 2; // now we directly use mlp

    cv::CommandLineParser parser(argc, argv, "{data|abc.txt|}{save||}{load||}"); // SEE IF THE LETTER-RECOGNITION.DATA IS AT THIS DIRECTORY
    data_filename = parser.get<string>("data");
    
    if (parser.has("save"))
        filename_to_save = parser.get<string>("save");
    if (parser.has("load"))
        filename_to_load = parser.get<string>("load");

    help();
    build_mlp_classifier( data_filename, filename_to_save, filename_to_load );

    return 0;
}
