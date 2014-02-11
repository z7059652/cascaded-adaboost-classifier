#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
using std::vector;

//////////////////////////////////////////////////////////////////////////
// adaboost
#define THRSH_SPACE 200

// struct of a weak classifier
typedef struct WEAKCLASSIFIER
{
	int iDim;
	float fThr;
	float fThreshold;
	int iDirection;
	float fAlpha;
	float fError;
}WEAKCLASSIFIER;

#define CLASSIFIER_TYPE vector<vector<WEAKCLASSIFIER>>

// generate the pool of weak classifiers
void GetWeakClassifierPool(vector<WEAKCLASSIFIER> &wc_pool, CvMat *features_matrix);

// clear the pool of weak classifiers
void ClearWCpool(vector<WEAKCLASSIFIER> &wc_pool);
 
// train one adaboost classifier, composed of a set of weak classifiers
void AdaBoost_train(vector<WEAKCLASSIFIER> &sc, CvMat *features_matrix, vector<int> &vecLabel, vector<WEAKCLASSIFIER> &wc_pool);
void TrainClassifier(vector<WEAKCLASSIFIER> &sc, CvMat *features_matrix, CvMat *label);

// apply one adaboost classifier to a feature vector
int  Adaboost_apply(CvMat *feature_vector, vector<WEAKCLASSIFIER>  &sc, float sc_thr);
int  ApplyClassifier(CvMat *feature_vector, vector<WEAKCLASSIFIER> &sc);

// apply a weak classifier to a feature vector
void ApplyWeakClassifier(WEAKCLASSIFIER &wc, vector<int> &vecResult, CvMat *features_matrix, vector<int> &vecLabel, vector<float> &vecWeight);

// file IO, one adaboost classifier
void WriteClassifier(vector<WEAKCLASSIFIER> &sc, const char *sPath);
void ReadClassifier(vector<WEAKCLASSIFIER> &sc, const char *sPath);

// train cascaded adaboost classifier, composed of a set of adaboost classifiers
void AdaBoost_train_cascade(vector<vector<WEAKCLASSIFIER>> &sc, CvMat *features_matrix, vector<int> &vecLabel, vector<WEAKCLASSIFIER> &wc_pool, const char *sPath);
void TrainClassifier_cascade(vector<vector<WEAKCLASSIFIER>> &sc, CvMat *features_matrix, CvMat *label, const char *sPath);

// apply cascaded adaboost classifier to a feature vector
int Adaboost_apply_cascade(CvMat *feature_vector, vector<vector<WEAKCLASSIFIER>> &sc);
CvMat *UpdateNegSamples(vector<vector<WEAKCLASSIFIER>> &sc, CvMat *features_matrix_neghard);
int  ApplyClassifier(CvMat *feature_vector, vector<vector<WEAKCLASSIFIER>> &sc);

// file IO, one adaboost classifier
void WriteClassifier(vector<vector<WEAKCLASSIFIER>> &sc, const char *sPath);
void ReadClassifier(vector<vector<WEAKCLASSIFIER>> &sc, const char *sPath);

// train and evaluate one adaboost classifier in cascaded model, and each adaboost classifier can be regarded as a node in cascaded adaboost classifier
int  EvalNodeClassifier(float *pos_aspos_rate, float *neg_aspos_rate, float &node_threshold, vector<WEAKCLASSIFIER> &nc, CvMat *features_matrix, vector<int> &vecLabel);

void AdaBoost_train_node(vector<WEAKCLASSIFIER> &nc, CvMat *features_matrix, vector<int> &vecLabel, vector<WEAKCLASSIFIER> &wc_pool);


#endif
