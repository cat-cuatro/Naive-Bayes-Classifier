// John Lorenz IV // CS 445 // Program #2
// Implementation file. Contains implementations for functions
// contained in the naiveBays.h file. No function prototypes or
// class definitions exist in this file.
#include "naiveBays.h"
//////////////////////////////
// Class Methods
//////////////////////////////
naiveMatrix::naiveMatrix(){
  int i = 0;
  for(i = 0; i < MAX_FEATURES; ++i){
    mean_of_features_neg[i] = 0;
    stdev_of_features_neg[i] = 0;
    mean_of_features_pos[i] = 0;
    stdev_of_features_pos[i] = 0;
  }
}
naiveMatrix::~naiveMatrix(){
  int i = 0;
  for(i = 0; i < MAX_FEATURES; ++i){
    mean_of_features_neg[i] = 0;
    stdev_of_features_neg[i] = 0;
    mean_of_features_pos[i] = 0;
    stdev_of_features_pos[i] = 0;
  }
}
// Prints features respective to the class . . an early debugging method
void naiveMatrix::print_features(int flag){
  int i,j;
  if(flag == MEAN){
    cout << "#\tMean of Features:" << endl;
  }
  else{
    cout << "#\tStandard Deviation of Features:" << endl;
  }
  for(i = 0; i < MAX_FEATURES; ++i){
    cout << i << "\t";
    if(flag == MEAN){
      cout << mean_of_features_pos[i] << endl;
    }
    else{
      cout << stdev_of_features_pos[i] << endl;
    }
  }
  cout << endl;
}
// Load the specified data file into the data array
void naiveMatrix::loadData(int flag, double data[][MAX_FEATURES]){
  int i,j,k;
  ifstream spam_data;
  if(flag == TRAIN){
    spam_data.open("train.txt");
  }
  else{
    spam_data.open("test.txt");
  }
  for(i = 0; i < MAX_ROWS; ++i){
    for(j = 0; j < MAX_FEATURES; ++j){
      spam_data >> data[i][j];
      if(j != MAX_FEATURES-1){ // last value is not comma-separated!
        spam_data.ignore(100, ',');
      }
    }
    spam_data.ignore(100, '\n'); // position file reader at beginning of next row
  }
  spam_data.close();
}
// Computes the mean for each feature (57 of them, 1 pos and 1 negative for a total of 114)
// Mean calculations are stored in the class data structure
double naiveMatrix::compute_means(double data[][MAX_FEATURES]){
  double num = 0;
  double sums_pos[MAX_FEATURES] = {0};
  double sums_neg[MAX_FEATURES] = {0};
  int i,j;
  double negative_entries = 0;
  double positive_entries = 0;
  for(i = 0; i < MAX_FEATURES-1; ++i){ 
    for(j = 0; j < MAX_ROWS; ++j){ 
      if(data[j][MAX_FEATURES-1] == 1){ // last 'feature' is a label for the class type
        sums_pos[i] += data[j][i]; // Sum all of the positive class features
        if(i == 0){ // <-- only want to count the # of positives once
          ++positive_entries; 
        }
      }
      else{
        sums_neg[i] += data[j][i]; // Sum all of the negative class features
        if(i == 0){ // <-- only want to count these once
          ++negative_entries;
        }
      }
    }
  }
  for(i = 0; i < MAX_FEATURES-1; ++i){
    mean_of_features_pos[i] =  sums_pos[i]/positive_entries; // compute mean of positive class features
    mean_of_features_neg[i] = sums_neg[i]/negative_entries;  // compute mean of negative class features
  }
  return num;
}
// Computes the standard deviation for each feature
double naiveMatrix::compute_stdevs(double data[][MAX_FEATURES]){
  double num = 0;
  int i,j,k;
  double sums_pos[MAX_FEATURES] = {0};
  double sums_neg[MAX_FEATURES] = {0};
  double negative_entries = 0;
  double positive_entries = 0;
  for(i = 0; i < MAX_FEATURES-1; ++i){ 
    for(j = 0; j < MAX_ROWS; ++j){ 
      if(data[j][MAX_FEATURES-1] == 1){ // if this is feature set is in the 'spam' class
        sums_pos[i] += pow(data[j][i]-mean_of_features_pos[i], 2.0); // same as sum_x->n((Xn-MEANn)^2)
        if(i == 0){
          ++positive_entries;
        }
      }
      else{ // if it isn't..
        sums_neg[i] += pow(data[j][i]-mean_of_features_neg[i], 2.0); // <-- mean of negative class features!
        if(i == 0){
          ++negative_entries;
        }
      }
    }
  }
  // Now we square root all of the values for both positive and negative classes

  for(i = 0; i < MAX_FEATURES-1; ++i){
    stdev_of_features_pos[i] = assertValidNumber(sqrt(sums_pos[i]/positive_entries)); // computes positive standard deviations
    stdev_of_features_neg[i] = assertValidNumber(sqrt(sums_neg[i]/negative_entries)); // same, except for negative
    // assertValidNumber() just makes sure that we don't divide by 0 later.
    // 0 values are changed to .00001 (close to zero)
  }  
  
  return num;
}
// Change 0 or nan values to .00001 (very close to 0)
double naiveMatrix::assertValidNumber(double value){
  // We want to be sure that we're not feeding 0s into the classifier.
  // Instead, replace instances of 0 or nan with close to 0 values.
  if(isnan(value)){ // if not a number
    value = .00001;
    cout << "Found nan" << endl;
  }
  else if(value == 0){ // or if a 0
    value = .00001;
    cout << "Found 0" << endl;
  }
  return value; 
}
int naiveMatrix::classify(double data[][MAX_FEATURES], int index){
  int spam = 0; // set to 1 if its spam, set to 0 if it isn't.
  double probability_pos[MAX_FEATURES];
  double probability_neg[MAX_FEATURES];
  double sumPos = 0;
  double sumNeg = 0;
  int i,j;
  double numerator = 0; // these will make the code easier to read.
  double denominator = 0; // ""
  double probTrue = log(PRIOR_TRUE); // normalize the data to prevent underflow
  double probFalse = log(PRIOR_FALSE);
  for(i = 0; i < MAX_FEATURES-1; ++i){
    // compute positive probabilities
    numerator = data[index][i] - mean_of_features_pos[i];
    numerator = -1.0 * pow(numerator,2.0);
    denominator = 2.0*pow(stdev_of_features_pos[i],2.0);
    probability_pos[i] = (1/(sqrt(2.0*M_PI)*stdev_of_features_pos[i]))*exp(numerator/denominator);
    // compute negative probabilities
    numerator = data[index][i] - mean_of_features_neg[i];
    numerator = -1.0 * pow(numerator, 2.0);
    denominator = 2.0*pow(stdev_of_features_neg[i], 2.0);
    probability_neg[i] = (1/(sqrt(2.0*M_PI)*stdev_of_features_neg[i]))*exp(numerator/denominator);
    // take the log of each term as we're calculating
    probability_pos[i] = log(probability_pos[i]);
    probability_neg[i] = log(probability_neg[i]);
  }
  for(i = 0; i < MAX_FEATURES-1; ++i){ // add together all 57 probabilities
    sumPos += probability_pos[i];
    sumNeg += probability_neg[i];
  }
  sumPos += probTrue; // same as log(P(class)+sum(log(P(x|true)))
  sumNeg += probFalse;
  if(sumPos >= sumNeg){
    spam = 1;
  }
  else{
    spam = 0;
  }
  return spam;
}
// Returns whether or not our prediction was correct
bool naiveMatrix::assertClass(int spam, int index, double data[][MAX_FEATURES]){
  bool correct = false;
  if(spam == data[index][MAX_FEATURES-1]){
    correct = true;
  }
  else{
    correct = false;
  } 
  return correct;
}
/////////////////////////////
// Non-class methods
/////////////////////////////
void printConfusionMatrix(int arr[][2]){
  int i,j;
  cout << "Confusion Matrix: " << endl;
  cout << "#\t0\t1\t" << endl;
  for(i = 0; i < 2; ++i){
    cout << i << "\t";
    for(j = 0; j < 2; ++j){
      cout << arr[i][j] << "\t";
    }
    cout << endl;
  }
}
// Increment confusion matrix based on naive bayes prediction
void incrementMatrix(int prediction, int target, int confusionMatrix[][2]){
  if(prediction == target){
    ++confusionMatrix[prediction][prediction];
  }
  else{
    ++confusionMatrix[prediction][target];
  }
}
