#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <sstream>

using namespace std;
using namespace cv;

extern "C" {
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba);
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_MotionAmplify(JNIEnv*, jobject, jlong addrGray, jint levels);

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;
    vector<KeyPoint> v;

    FastFeatureDetector detector(50);
    detector.detect(mGr, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }
}
}

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_MotionAmplify(JNIEnv*, jobject, jlong addrGray, jint levels)
{
	Mat& mGr  = *(Mat*)addrGray;

	vector<Mat> gaussPyr;
	buildPyramid(mGr, gaussPyr, levels);
	int l = gaussPyr.size();

	vector<Mat> laplacePyr;
	laplacePyr.resize(l);
	laplacePyr[l-1] = gaussPyr[l-1].clone();

	for (int i = 1; i < l; i++) {
		int cur = l-1-i;
		int prev = l-1-(i-1);
		laplacePyr[cur].create(gaussPyr[cur].size(), CV_32S);
		pyrUp(gaussPyr[prev], laplacePyr[cur], gaussPyr[cur].size());
		subtract(gaussPyr[cur], laplacePyr[cur], laplacePyr[cur], noArray(), CV_32S);
	}

	// --- felepult a laplace piramis

	//Size sz = mGr.size();
	//resize(laplacePyr[3], mGr, sz);

	Mat result = laplacePyr[l-1].clone();
	for (int i = 1; i < l; i++) {
		int cur = l-1-i;
		pyrUp(result, result, laplacePyr[cur].size());
		add(laplacePyr[cur], result, result, noArray(), CV_32S);
	}

	//mGr = result - mGr;

	//Size sz = mGr.size();
	//resize(laplacePyr[3], mGr, sz);

//	stringstream ss;
//	ss << gaussPyr[l-2].size().width << " " << gaussPyr[l-2].size().height << " " << gaussPyr[l-2].channels();
//	putText(mGr, ss.str(), Point(30,130), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
//
//	stringstream ss2;
//	ss2 << laplacePyr[l-2].size().width << " " << laplacePyr[l-2].size().height << " " << laplacePyr[l-2].channels();
//	putText(mGr, ss2.str(), Point(30,230), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);

	circle(mGr, Point(30, 30), 10, Scalar(255,255,255,255));
}
