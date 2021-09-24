
#include "dvs_meanshift/meanshift.h"
#include "dvs_meanshift/color_meanshift.h"
#include "image_reconstruct/image_reconst.h"
#include <time.h>

//remove this include, just for debugging
#include "graph3d/pnmfile.h"
#include <std_msgs/Float32.h>

#define min(x,y) ((x)<(y)?(x):(y))

double epsilon = 1E-10;
int counterGlobal = 0;
const int MAX_DISTANCE = 500; //actually the real distance is sqrt(MAX_DISTANCE)

bool firstevent = true;
double firsttimestamp;


namespace dvs_meanshift {

class Parallel_pixel_cosine: public cv::ParallelLoopBody
{

public:

	Parallel_pixel_cosine(cv::Mat imgg) : img(imgg)
	{
	}

	void operator() (const cv::Range &r) const
	{
		for(int j=r.start; j<r.end; ++j)
		{
			double* current = const_cast<double*>(img.ptr<double>(j));
			double* last = current + img.cols;

			for (; current != last; ++current)
				*current = (double) cos((double) *current);
		}
	}

private:
    cv::Mat img;
};

void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat1i &X, cv::Mat1i &Y)
{
  cv::repeat(xgv.reshape(1,1), ygv.total(), 1, X);
  cv::repeat(ygv.reshape(1,1).t(), 1, xgv.total(), Y);
}

// helper function (maybe that goes somehow easier)
 void meshgridTest(const cv::Range &xgv, const cv::Range &ygv, cv::Mat1i &X, cv::Mat1i &Y)
{
  std::vector<int> t_x, t_y;
  for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
  for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);

  meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}

void findClosestCluster(std::vector<int> * clustID, std::vector<double> clustX, std::vector<double> clustY,std::vector<double>clustX_old,std::vector<double> clustY_old)
{

	std::vector<double> distvalue (clustX.size());
	double tmp;

	for(int i=0; i<clustX.size(); i++)
	{
		distvalue[i]=MAX_DISTANCE;
		(*clustID).push_back(-1); //no match when clustID is -1
		for(int j=0; j<clustX_old.size(); j++)
		{
			tmp= (clustX[i]*DVSW-clustX_old[j]*DVSW)*(clustX[i]*DVSW-clustX_old[j]*DVSW) + (clustY[i]*DVSH-clustY_old[j]*DVSH)*(clustY[i]*DVSH-clustY_old[j]*DVSH);
			if(tmp<distvalue[i])
			{
				distvalue[i]=tmp;
				(*clustID)[i]=j;
			}
		}
	}
}


void Meanshift::assignClusterColor(std::vector<int> * positionClusterColor, int numClusters, std::vector<int> matches, std::vector<int> oldPositions, std::vector<int> activeTrajectories)
{

	if (matches.size()==0) //no mask --> assing all new colors
	{
		for(int i=0; i<numClusters; i++)
		{
			(*positionClusterColor).push_back(lastPositionClusterColor);
			lastPositionClusterColor = lastPositionClusterColor+1;
		}
	}
	else
	{
		for(int i=0; i<numClusters; i++)
		{
			if(matches[i]==-1) //No match with previous clusters --> new cluster, new color
			{
				////Check if the new position is being used
				//while ( activeTrajectories.end() != find (activeTrajectories.begin(), activeTrajectories.end(), (lastPositionClusterColor & (MAX_NUM_CLUSTERS-1))))
				//	lastPositionClusterColor++;

				(*positionClusterColor).push_back(lastPositionClusterColor);
				lastPositionClusterColor = lastPositionClusterColor+1;
			}
			else //Match with previous cluster --> assign color of previous cluster
			{
				(*positionClusterColor).push_back(oldPositions[matches[i]]);
			}
		}

	}
}

Meanshift::Meanshift(ros::NodeHandle & nh, ros::NodeHandle nh_private) : nh_(nh)
{
  used_last_image_ = false;

  // get parameters of display method
  std::string display_method_str;
  nh_private.param<std::string>("display_method", display_method_str, "");
  // setup subscribers and publishers
  //event_sub_ = nh_.subscribe("events", 1, &Meanshift::eventsCallback, this);
  event_sub_ = nh_.subscribe("events", 100, &Meanshift::eventsCallback_simple, this);
  camera_info_sub_ = nh_.subscribe("camera_info", 100, &Meanshift::cameraInfoCallback, this);

  image_transport::ImageTransport it_(nh_);
  
  image_pub_ = it_.advertise("dvs_rendering", 1);
  image_segmentation_pub_ = it_.advertise("dvs_segmentation", 1);


  //DEBUGGING
  image_debug1_pub_ = it_.advertise("dvs_debug1", 1);
  image_debug2_pub_ = it_.advertise("dvs_debug2", 1);
  //NOT DEBUGGING

  //Initializing params for color meanshift
  //computing image calibration
  numRows=DVSH; numCols=DVSW;
  mPixels=numRows*numCols;  maxPixelDim=3;

  spaceDivider=7; timeDivider = 5;
  maxIterNum=1;
  tolFun=0.0001;
  kernelFun[0] = 'g'; kernelFun[1] = 'a'; kernelFun[2] = 'u'; kernelFun[3] = 's';
  kernelFun[4] = 's'; kernelFun[5] = 'i'; kernelFun[6] = 'a'; kernelFun[7] = 'n';

  //Initialize pointers
  outputFeatures = NULL; imagePtr = NULL; pixelsPtr = NULL;
  initializationPtr=new double[maxPixelDim*mPixels/4];

  /*for (int i = 0; i < numRows; i++)
  	for (int j = 0; j < numCols; j++)
  	{
  		initializationPtr[i*numCols + j]=j/spaceDivider;//x
  		initializationPtr[mPixels+i*numCols+j]=i/spaceDivider;//y
  		initializationPtr[2*mPixels+i*numCols+j]=0;//img

  	}*/

  //Write xposition, yposition, and image in the same vector
  for (int i = 0; i < numRows/2; i++)
	for (int j = 0; j < numCols/2; j++)
	{
		initializationPtr[i*numCols/2 + j]=j/spaceDivider;//x
		initializationPtr[mPixels/4+i*numCols/2+j]=i/spaceDivider;//y
		initializationPtr[2*mPixels/4+i*numCols/2+j]=0;//img

	}

  //Initialize params for graph3d segmentation
  sigma = 0.75;
  k = 500;
  min_region = (int)(numRows*numCols*0.01);
  num_components = 0;

  //Initialization of random seed
  srand(time(NULL));//Random seed initialization
  //asm("rdtsc\n"
  //    "mov edi, eax\n"
  //    "call   srand");

  //Create first the RGB values and then assign it to the clusters
  //(hopefully, this will keep some stability in the cluster colors along consecutive frames)
  //Assume we won't have more than 50 clusters
  for(int i=0; i<MAX_NUM_CLUSTERS; i++)
  {
	  RGBColors.push_back(cv::Vec3b((uchar)random(), (uchar)random(), (uchar)random()));
	  //counterTrajectories.push_back(0);//initialize counter trajectories to all zeros
  }
  counterTrajectories=std::vector<int>(MAX_NUM_CLUSTERS,0);

  lastPositionClusterColor=0;

  //Initializing trajectories
  trajectories = cv::Mat(numRows, numCols, CV_8UC3);
  trajectories = cv::Scalar(128,128,128);

  //Initialize trajectory matrix
  allTrajectories = new std::vector<cv::Point>[MAX_NUM_CLUSTERS];
  for(int i = 0; i < MAX_NUM_CLUSTERS; i++)
  {
	  allTrajectories[i] = std::vector<cv::Point>(MAX_NUM_TRAJECTORY_POINTS);
  }
  prev_activeTrajectories=std::vector<int>(MAX_NUM_CLUSTERS,0);

  //Init Kalman filter
  //createKalmanFilter();
  vector_of_kf=std::vector<cv::KalmanFilter>(MAX_NUM_CLUSTERS);
  vector_of_meas=std::vector<cv::Mat>(MAX_NUM_CLUSTERS);
  vector_of_state=std::vector<cv::Mat>(MAX_NUM_CLUSTERS);
  for(int i=0; i<vector_of_kf.size(); i++)
	  createKalmanFilter(&(vector_of_kf[i]), &(vector_of_state[i]), &(vector_of_meas[i]));

  selectedTrajectory = -1;
  //foundBlobs = false;
  vector_of_foundBlobs=std::vector<bool>(MAX_NUM_CLUSTERS,false);
  notFoundBlobsCount = 0;
  //foundTrajectory=false;
  foundTrajectory = std::vector<bool>(MAX_NUM_CLUSTERS, false);
  //ticks = 0;
  vector_of_ticks=std::vector<double> (MAX_NUM_CLUSTERS,0.0);

  //BG activity filtering
  BGAFframe = cv::Mat(numRows, numCols, CV_32FC1);
  BGAFframe =cv::Scalar(0.);
}

Meanshift::~Meanshift()
{
  delete [] initializationPtr;

  image_pub_.shutdown();  
  image_segmentation_pub_.shutdown();
  image_debug1_pub_.shutdown();
}

void Meanshift::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
  camera_matrix_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      camera_matrix_.at<double>(cv::Point(i, j)) = msg->K[i+j*3];

  dist_coeffs_ = cv::Mat(msg->D.size(), 1, CV_64F);
  for (int i = 0; i < msg->D.size(); i++)
    dist_coeffs_.at<double>(i) = msg->D[i];
}


void Meanshift::createKalmanFilter(cv::KalmanFilter *kf, cv::Mat *state, cv::Mat *meas)
{
	/*
	 * This implementation takes 6 states position, velocity, and dimensions of the bounding box
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;

	unsigned int type = CV_32F;

	//cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
	kf.init(stateSize, measSize, contrSize, type);

	cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
	cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
	// [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

	// Transition State Matrix A
	// Note: set dT at each processing step!
	// [ 1 0 dT 0  0 0 ]
	// [ 0 1 0  dT 0 0 ]
	// [ 0 0 1  0  0 0 ]
	// [ 0 0 0  1  0 0 ]
	// [ 0 0 0  0  1 0 ]
	// [ 0 0 0  0  0 1 ]
	cv::setIdentity(kf.transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 ]
	// [ 0 0 0 0 1 0 ]
	// [ 0 0 0 0 0 1 ]
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(7) = 1.0f;
	kf.measurementMatrix.at<float>(16) = 1.0f;
	kf.measurementMatrix.at<float>(23) = 1.0f;

	// Process Noise Covariance Matrix Q
	// [ Ex   0   0     0     0    0  ]
	// [ 0    Ey  0     0     0    0  ]
	// [ 0    0   Ev_x  0     0    0  ]
	// [ 0    0   0     Ev_y  0    0  ]
	// [ 0    0   0     0     Ew   0  ]
	// [ 0    0   0     0     0    Eh ]
	//cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
	kf.processNoiseCov.at<float>(0) = 1e-2;
	kf.processNoiseCov.at<float>(7) = 1e-2;
	kf.processNoiseCov.at<float>(14) = 5.0f;
	kf.processNoiseCov.at<float>(21) = 5.0f;
	kf.processNoiseCov.at<float>(28) = 1e-2;
	kf.processNoiseCov.at<float>(35) = 1e-2;

	// Measures Noise Covariance Matrix R
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
	*/


	/*
	 * This implementation is only for one kalman filter
	 */
	int stateSize = 4;
	int measSize = 2;
	int contrSize = 0;

	unsigned int type = CV_32F;

	kf->init(stateSize, measSize, contrSize, type);

	//cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y]
	//cv::Mat meas(measSize, 1, type);    // [z_x,z_y]
	state->create(stateSize,1,type);
	meas->create(measSize,1,type);
	// [E_x,E_y,E_v_x,E_v_y]


	// Transition State Matrix A
	// Note: set dT at each processing step!
	// [ 1 0 dT 0 ]
	// [ 0 1 0  dT]
	// [ 0 0 1  0 ]
	// [ 0 0 0  1 ]
	cv::setIdentity(kf->transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 ]
	// [ 0 1 0 0 ]
	kf->measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf->measurementMatrix.at<float>(0) = 1.0f;
	kf->measurementMatrix.at<float>(5) = 1.0f;

	// Process Noise Covariance Matrix Q
	// [ Ex   0   0     0     ]
	// [ 0    Ey  0     0     ]
	// [ 0    0   Ev_x  0     ]
	// [ 0    0   0     Ev_y  ]
	//cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
	kf->processNoiseCov.at<float>(0) = 1e-2;//original values
	kf->processNoiseCov.at<float>(5) = 1e-2;
	kf->processNoiseCov.at<float>(10) = 5.0f;
	kf->processNoiseCov.at<float>(15) = 5.0f;

	kf->processNoiseCov.at<float>(0) = 1e-2;
	kf->processNoiseCov.at<float>(5) = 1e-2;
	kf->processNoiseCov.at<float>(10) = 7.0f;
	kf->processNoiseCov.at<float>(15) = 7.0f;

	// Measures Noise Covariance Matrix R
	cv::setIdentity(kf->measurementNoiseCov, cv::Scalar(1e-1));
}

void Meanshift::eventsCallback_simple(const dvs_msgs::EventArray::ConstPtr& msg)
{
	// only create image if at least one subscriber
	//ROS_INFO("event callback height %d width %d", msg->height, msg->width);
	// color the events and publish them so we see the events nicely 
	cv_bridge::CvImage cv_image;  
	cv_image.encoding = "bgr8";  
	if (last_image_.rows == msg->height && last_image_.cols == msg->width)
	{
	  last_image_.copyTo(cv_image.image);
	  used_last_image_ = true;
	}
	else
	{
	  cv_image.image = cv::Mat(msg->height, msg->width, CV_8UC3);
	  cv_image.image = cv::Scalar(128, 128, 128);
	}  
	for (int i = 0; i < msg->events.size(); ++i)
	{
	  // if the polarity is positive color red else blue
	  const int x = msg->events[i].x;
	  const int y = msg->events[i].y;  
	  cv_image.image.at<cv::Vec3b>(cv::Point(x, y)) = (
	      msg->events[i].polarity == true ? cv::Vec3b(255, 0, 0) : cv::Vec3b(0, 0, 255));
	}  	
	image_pub_.publish(cv_image.toImageMsg());

	// prepare segmentation image
	cv_bridge::CvImage cv_segments;
	// get the time stamp of the first event
	uint64_t first_timestamp = msg->events[0].ts.toNSec();

	// get the timestamp of the last event
	double final_timestamp =  (1E-6*(double)(msg->events[(msg->events.size())-1].ts.toNSec()-first_timestamp));
	// always have the active events
	std::vector<bool> activeEvents(msg->events.size());

	int counterIn = 0;
	int counterOut = 0;
	int beginEvent = 0;
	int packet = 1800;


	while(beginEvent < msg->events.size())
	{

	  	counterIn = 0;
	  	counterOut = 0;
	  	// handle a smaller amount of data
	  	cv::Mat data=cv::Mat(3, min(packet, msg->events.size()-beginEvent), CV_64F, cv::Scalar::all(0.));
	  	if(firstevent)
	  	{
			firsttimestamp = (1E-6*(double)(msg->events[0].ts.toNSec()));
			firstevent = false;
	  	}
	  	double maxTs;
	  	int posx, posy;
	  	double usTime = 1000.0;
	  	double ts;

	  	for (int i = beginEvent; i < min(beginEvent+packet, msg->events.size()); i++)
	  	{
		  	//select a smaller amount of events, size of the packet
		  	const int x = msg->events[counterIn].x;
		  	const int y = msg->events[counterIn].y;
			
		  	//get the current time stamp
		  	double event_timestamp =  (1E-6*(double)(msg->events[counterIn].ts.toNSec()-first_timestamp));//now in usecs
		  	// get the time that has passed
		  	ts = (1E-6*(double)(msg->events[i].ts.toNSec())) - firsttimestamp;


			#if BG_FILTERING // for each event, check all the neighboring events and remove the old ones

		  	BGAFframe.at<float>(cv::Point(x,y))=0.;
		  	maxTs = -1;
		  	//calculate maximum time in a 3x3 neighborhood
			for(int ii=-1; ii<=1; ii++)
			{  
				for(int jj=-1; jj<=1; jj++)
			  	{
				  	posx = x + ii;
				  	posy = y + jj;
				  	if(posx<0)
						  posx = 0;
				  	if(posy<0)
						  posy=0;
				  	if(posx>numRows-1)
						  posx = numRows-1;
				  	if(posy>numCols-1)
						  posy = numCols-1;
				  	if(BGAFframe.at<float>(cv::Point(posx,posy)) > maxTs)
						  maxTs = BGAFframe.at<float>(cv::Point(posx,posy));
				}
			}

			BGAFframe.at<float>(cv::Point(x,y))=ts;
			//if nothing happened in usTime, remove the event
			if(BGAFframe.at<float>(cv::Point(x,y)) >= (maxTs + usTime))
			{
			  activeEvents.at(counterIn)=false;
			  counterIn++;
			}
			else
			{
			#endif
				// normalize position in data 
				data.at<double>(cv::Point(counterOut, 0))= (double)x/numCols;
				data.at<double>(cv::Point(counterOut, 1))= (double)y/numRows;
				//data.at<double>(cv::Point(counterOut, 2))= event_timestamp/final_timestamp;//normalized
				double tau = 10000;
				data.at<double>(cv::Point(counterOut, 2))= exp(-(final_timestamp-event_timestamp)/tau);//normalized
				activeEvents.at(counterIn)=true;
				counterIn++;
				counterOut++;
			#if BG_FILTERING
			}
			#endif
		  	}
		  	// get the most recent time stamp
		  	double last_timestamp =  (1E-6*(double)(msg->events[counterIn-1].ts.toNSec()));//now in usecs

		  	// prepare the cluster centers and the output image
		  	cv::Mat clusterCenters;
		  	cv::Mat segmentation=cv::Mat(numRows, numCols, CV_8UC3);
		  	segmentation = cv::Scalar(128,128,128);

		  	cv::Mat traj = cv::Mat(numRows, numCols, CV_8UC3);
		  	traj = cv::Scalar(128,128,128);

		  	std::vector<double> clustCentX, clustCentY, clustCentZ;
		  	std::vector<int> point2Clusters;
		  	std::vector<int> positionClusterColor;
		  	std::vector<int> assign_matches;


		  	double bandwidth = 0.15;
			// do the clutering
		  	meanshiftCluster_Gaussian(data, &clustCentX, &clustCentY, &clustCentZ, &point2Clusters, bandwidth);

			//Assign new color or use color from previous clusters?
			//if(counterGlobal>0)//not the first time
			findClosestCluster(&assign_matches, clustCentX, clustCentY, prev_clustCentX, prev_clustCentY); //no match when clustID is -1

			assignClusterColor(&positionClusterColor, clustCentX.size(), assign_matches, prev_positionClusterColor, prev_activeTrajectories); //assign new colors to clusters


			int tmpColorPos;
			float estimClustCenterX=-1., estimClustCenterY=-1.;
			std::vector<int> activeTrajectories;

			if(point2Clusters.size()>100) //update positions only when there is enough change (>25%)
			{
			  for(int i=0; i<clustCentX.size(); i++)
			  {
				  estimClustCenterX=-1., estimClustCenterY=-1.;

				  tmpColorPos = (positionClusterColor[i])&(MAX_NUM_CLUSTERS-1);

				  //Check if the new position is being used
				  /*int idx = 0;
				  for (int idx=0; idx<prev_activeTrajectories.size(); idx++)
					  if(tmpColorPos == prev_activeTrajectories[idx])
						  break;*/
				  std::vector<int>::iterator it;

				  it = find (prev_activeTrajectories.begin(), prev_activeTrajectories.end(), tmpColorPos);

				  if(it==prev_activeTrajectories.end())//if the element tmpcolorpos is not found in the vector
				  {
					  counterTrajectories[tmpColorPos]=0;

#if KALMAN_FILTERING
						  foundTrajectory[tmpColorPos] = false;
						  createKalmanFilter(&(vector_of_kf[tmpColorPos]), &(vector_of_state[tmpColorPos]), &(vector_of_meas[tmpColorPos]));
						  vector_of_foundBlobs[tmpColorPos]=false;
						  vector_of_ticks[tmpColorPos]=0.0;
#endif
					  }


#if KALMAN_FILTERING
					  if(counterTrajectories[tmpColorPos]>50 & !foundTrajectory[tmpColorPos])
					  {
						  //selectedTrajectory = tmpColorPos;
						  foundTrajectory[tmpColorPos] = true;
					  }
					  //if(foundTrajectory[tmpColorPos] & selectedTrajectory == tmpColorPos) //More than 25 points in the trajectory, start w/ KF
					  if(foundTrajectory[tmpColorPos]) //More than 25 points in the trajectory, start w/ KF
					  {

						  if((last_timestamp - vector_of_ticks[tmpColorPos]) > 1)
							{
								double precTick = vector_of_ticks[tmpColorPos];
								vector_of_ticks[tmpColorPos] = last_timestamp;
								double dT = (vector_of_ticks[tmpColorPos] - precTick)/1000; //seconds
								//if (foundBlobs)
								if(vector_of_foundBlobs[tmpColorPos])
								{
									// >>>> Matrix A
									vector_of_kf[tmpColorPos].transitionMatrix.at<float>(2) = dT;
									vector_of_kf[tmpColorPos].transitionMatrix.at<float>(7) = dT;
									// <<<< Matrix A
									vector_of_state[tmpColorPos] = vector_of_kf[tmpColorPos].predict();
									estimClustCenterX = vector_of_state[tmpColorPos].at<float>(0);
									estimClustCenterY = vector_of_state[tmpColorPos].at<float>(1);
								}
								vector_of_meas[tmpColorPos].at<float>(0) = clustCentX[i]*DVSW;
								vector_of_meas[tmpColorPos].at<float>(1) = clustCentY[i]*DVSH; //[z_x, z_y]


								//if (!foundBlobs) // First detection!
								if(!vector_of_foundBlobs[tmpColorPos])
								{
									// >>>> Initialization
									vector_of_kf[tmpColorPos].errorCovPre.at<float>(0) = 1; // px
									vector_of_kf[tmpColorPos].errorCovPre.at<float>(5) = 1; // px
									vector_of_kf[tmpColorPos].errorCovPre.at<float>(10) = 2;
									vector_of_kf[tmpColorPos].errorCovPre.at<float>(15) = 2;

									vector_of_state[tmpColorPos].at<float>(0) = vector_of_meas[tmpColorPos].at<float>(0);
									vector_of_state[tmpColorPos].at<float>(1) = vector_of_meas[tmpColorPos].at<float>(1);
									vector_of_state[tmpColorPos].at<float>(2) = 0;
									vector_of_state[tmpColorPos].at<float>(3) = 0; //[z_x, z_y, v_x, v_y]
									// <<<< Initialization

									vector_of_kf[tmpColorPos].statePost = vector_of_state[tmpColorPos];

									//foundBlobs = true;
									vector_of_foundBlobs[tmpColorPos]=true;
								}
								else
									vector_of_kf[tmpColorPos].correct(vector_of_meas[tmpColorPos]); // Kalman Correction

								if(estimClustCenterX>=0.) //initialized to -1
								{
									//ROS_ERROR_STREAM("Estimated difference = ["<<abs(estimClustCenterX/DVSW -clustCentX[i]) <<", "<<abs(estimClustCenterY/DVSH-clustCentY[i])<<"];");
									//ROS_ERROR_STREAM("Estimated point = ["<<estimClustCenterX<<", "<<estimClustCenterY<<"];");
									//ROS_ERROR_STREAM("Measured point = ["<<clustCentX[i]<<", "<<clustCentY[i]<<"];");
									cv::circle(trajectories, cv::Point(clustCentX[i]*DVSW, clustCentY[i]*DVSH), 2, cv::Scalar( 0, 0, 255 ),-1,8);
									cv::circle(trajectories, cv::Point(estimClustCenterX, estimClustCenterY), 2, cv::Scalar( 255, 0, 0),-1,8);

									//foutX<<tmpColorPos<<" "<<clustCentX[i]*DVSW<<" "<<clustCentY[i]*DVSH<<" "<<estimClustCenterX<<" "<<estimClustCenterY<<std::endl;

									//foutX_estim<<estimClustCenterX<<" "<<estimClustCenterY<<std::endl;
									//foutT<<dT<<std::endl;
									//foutX<<"Estimated difference = ["<<abs(estimClustCenterX/DVSW -clustCentX[i]) <<", "<<abs(estimClustCenterY/DVSH-clustCentY[i])<<"];"<<std::endl;

									// !!!!!!CAREFUL ****************************************************************************************
									//ACTIVATE THIS !!!!!
									//clustCentX[i] = estimClustCenterX/DVSW;
									//clustCentY[i] = estimClustCenterY/DVSH;
									// CAREFUL ****************************************************************************************
								}
							}
					  }
#endif

					  cv::Point end(clustCentX[i]*DVSW, clustCentY[i]*DVSH);



					  allTrajectories[tmpColorPos][(counterTrajectories[tmpColorPos]) & (MAX_NUM_TRAJECTORY_POINTS-1)]=end;
					  counterTrajectories[tmpColorPos]++;
					  activeTrajectories.push_back(tmpColorPos);
				  }

				  prev_clustCentX = clustCentX;
				  prev_clustCentY = clustCentY;
				  prev_positionClusterColor = positionClusterColor;
				  prev_activeTrajectories = activeTrajectories;

				  trajectories = cv::Scalar(128,128,128);
				  int first=1;
				  for(int i=0; i<activeTrajectories.size(); i++)
				  {
					int tmpval = activeTrajectories[i];
					if (counterTrajectories[tmpval]>1)
					{
						if(counterTrajectories[tmpval]<=MAX_NUM_TRAJECTORY_POINTS)
						{
							for(int j=1; j<counterTrajectories[tmpval]; j++) //instead of % I use &(power of 2 -1): it is more efficient
								cv::line( trajectories, allTrajectories[tmpval][j-1], allTrajectories[tmpval][j], cv::Scalar( RGBColors[tmpval].val[0], RGBColors[tmpval].val[1], RGBColors[tmpval].val[2]), 2, 1);
						}
						else
						{
							int j=1;
							int pos = counterTrajectories[tmpval];
							while(j<MAX_NUM_TRAJECTORY_POINTS)
							{
								cv::line( trajectories, allTrajectories[tmpval][pos & (MAX_NUM_TRAJECTORY_POINTS-1)], allTrajectories[tmpval][(pos+1)&(MAX_NUM_TRAJECTORY_POINTS-1)], cv::Scalar( RGBColors[tmpval].val[0], RGBColors[tmpval].val[1], RGBColors[tmpval].val[2]), 2, 1);
								pos = pos+1;
								j++;
							}
						}
					}

				  }
			  }

			  //trajectories.copyTo(segmentation); //Always keep trajectories

			  counterIn =0;
			  counterOut=0;
			  for (int i = beginEvent; i < min(beginEvent+packet, msg->events.size()); i++)
			  {
				  if(activeEvents.at(counterIn)) //Label it only if it is not noise
				  {
					  const int x = msg->events[counterIn].x;
					  const int y = msg->events[counterIn].y;
					  double ts =  (1E-6*(double)(msg->events[counterIn].ts.toNSec()-first_timestamp));//now in usecs

					  //if(ts<15) //This is just for painting, we are processing all events
					  //{
						  //segmentation.at<cv::Vec3b>(cv::Point(x,y))=RGBColors[(positionClusterColor[point2Clusters[counter]])%MAX_NUM_CLUSTERS];
						  segmentation.at<cv::Vec3b>(cv::Point(x,y))=RGBColors[(positionClusterColor[point2Clusters[counterOut]])&(MAX_NUM_CLUSTERS-1)]; //cheaper than % (mod operation)
						  counterOut++;
				  }

				  counterIn++;
			  }
			counterOut=0;
			  for(int i=0; i<clustCentX.size(); i++)
			  {
				  	for(int jj = -1; jj<1; jj++)
					  { 
						  
				  	for(int ii = -1; ii<1; ii++)
					  { 
				  		const int x = clustCentX[i]*DVSW;
						const int y = clustCentY[i]*DVSH;
						posx = x + ii;
				  		posy = y + jj;
				  		if(posx<0)
							  posx = 0;
				  		if(posy<0)
							  posy=0;
				  		if(posx>numCols-1)
						  {
							  posx = numCols-1;
						  }
				  		if(posy>numRows-1)
							  posy = numRows-1;
						segmentation.at<cv::Vec3b>(cv::Point(posx,posy))=cv::Vec3b(255, 255, 255); //cheaper than % (mod operation)
					  }
					  }
					counterOut++;


			  }
		  cv_segments.encoding = "bgr8";
		  cv_segments.image = segmentation;
		  image_segmentation_pub_.publish(cv_segments.toImageMsg());


		  beginEvent +=packet;

		  counterGlobal++;
	  }
}



//
} // namespace
