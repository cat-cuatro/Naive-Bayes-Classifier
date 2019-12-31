// John Lorenz IV // CS 445 // Program #2
// This is the main implementation of the program. 
// In this file there are functions related to looping through
// the algorithm. This program uses naive bays to classify data
// that is given to it
#include "naiveBays.h"

int main(){
  naiveMatrix bayesClassifier;
  double data[MAX_ROWS][MAX_FEATURES]; // data[row][columnn]
  int i,j;
  int spam = 0;
  bool correct = false;
  int incorrect = 0;
  double accuracy = 0;
  double precision = 0;
  double recall = 0;
  int confusionMatrix[2][2];
  bayesClassifier.loadData(TRAIN, data); // load the training data
  bayesClassifier.compute_means(data);   // Start training the classifier
  bayesClassifier.compute_stdevs(data);  // ..
  bayesClassifier.loadData(TEST, data);  // Load the testing data

  for(i = 0; i < MAX_ROWS; ++i){
    spam = bayesClassifier.classify(data, i); // Begin testing
    correct = bayesClassifier.assertClass(spam, i, data); // Are we correct?
    if(!correct){
      ++incorrect;
    }
    incrementMatrix(spam, data[i][MAX_FEATURES-1], confusionMatrix);
  }

  // Calculate metrics and display confusion matrix:
  accuracy = (double)MAX_ROWS - (double)incorrect;
  accuracy = (accuracy/(double)MAX_ROWS) * 100.0;
  precision = ((double)confusionMatrix[1][1]/((double)confusionMatrix[1][1]+(double)confusionMatrix[0][1])) * 100.0;
  recall = (double)confusionMatrix[1][1]/((double)confusionMatrix[1][1]+(double)confusionMatrix[1][0]) * 100.0;
  cout << "Incorrect: " << incorrect << " times." << endl;
  printConfusionMatrix(confusionMatrix);
  cout << "Accuracy: " << accuracy << "% // " << "Precision: " << precision << "% // " << "Recall: " << recall << "%" << endl;
  return 0;
}

