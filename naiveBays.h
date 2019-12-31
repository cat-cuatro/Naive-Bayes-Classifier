/*****************************************/
// John Lorenz IV // Program #2 // CS445 //
/*****************************************/
// Naive Bayes classifier program. Header file contains
// included libraries as well as class definitions, methods, etc.
// No implementations in this file.
#include <iostream>
#include <math.h>
#include <fstream>
#include <time.h>
using namespace std;
#define PRIOR_TRUE 39.4 // P(1) == 39.4% (That the entry is spam)
#define PRIOR_FALSE 60.6// P(0) == 60.6% (That the entry is not spam)
#define MAX_FEATURES 58 // Range: [0-57], where 57th index is the class label
#define MAX_ROWS 2300   // Technically 2301 in one of the files, but easier to round to 2300.
#define MEAN 1
#define STDEV 2
#define TRAIN 1
#define TEST 2
class naiveMatrix{
  public:
    //methods
    naiveMatrix();
    ~naiveMatrix();
    void loadData(int flag, double data[][MAX_FEATURES]);
    double compute_means(double data[][MAX_FEATURES]);
    double compute_stdevs(double data[][MAX_FEATURES]);
    void print_features(int flag);
    double assertValidNumber(double value);
    int classify(double data[][MAX_FEATURES], int index);
    bool assertClass(int spam, int index, double data[][MAX_FEATURES]);
    // variables
  private:
    double mean_of_features_neg[MAX_FEATURES];
    double mean_of_features_pos[MAX_FEATURES];
    double stdev_of_features_neg[MAX_FEATURES];
    double stdev_of_features_pos[MAX_FEATURES];
};
void printConfusionMatrix(int arr[][2]);
void incrementMatrix(int prediction, int target, int confusionMatrix[][2]);
