#include "adaboost.h"
#include <algorithm>

static float pos_limit = 0.995; 
static float neg_limit = 0.6;

void AdaBoost_train_node(vector<WEAKCLASSIFIER> &nc, CvMat *features_matrix, vector<int> &vecLabel, vector<WEAKCLASSIFIER> &wc_pool)
{
	int numSamples = features_matrix->rows;
	int numFeatures = features_matrix->cols;

	vector<float> vecWeight; vecWeight.resize(numSamples);
	for (int s = 0;s < numSamples;s++)
		vecWeight[s] = 1.0 / numSamples;

	float node_threshold = 0;
	int nodeitr = 0;
	for (; nodeitr < 5000; nodeitr++) // the maximum iteration of node classifier is 5000
	{
		ClearWCpool(wc_pool);
		float minerr = 100000.0; int optwix = 0;
		for (int wix = 0; wix < wc_pool.size(); wix++)
		{
			WEAKCLASSIFIER &wc = wc_pool[wix];
			vector<int> vecResult;
			ApplyWeakClassifier(wc, vecResult, features_matrix, vecLabel, vecWeight);
			if (wc.fError < minerr)
			{
				minerr = wc.fError;
				optwix = wix;
			}
		}

		WEAKCLASSIFIER &wc = wc_pool[optwix];
		vector<int> vecResult;
		ApplyWeakClassifier(wc, vecResult, features_matrix, vecLabel, vecWeight);
		nc.push_back(wc);
		float fsum = 0;
		for (int s = 0;s < numSamples;s++)
		{
			vecWeight[s] = vecWeight[s] * exp(-wc.fAlpha*vecResult[s]*vecLabel[s]);
			fsum = fsum + vecWeight[s];
		}
		for (int s = 0;s < numSamples;s++)
			vecWeight[s] = vecWeight[s] / fsum;

		float pos_aspos_rate = 0; float neg_aspos_rate = 0; 
		int iret = EvalNodeClassifier(&pos_aspos_rate, &neg_aspos_rate, node_threshold, nc, features_matrix, vecLabel);
		if (pos_aspos_rate >= pos_limit && neg_aspos_rate <= neg_limit){
			printf("nodeitr=%d, pos_aspos_rate=%.3f, neg_aspos_rate=%.3f, node_threshold = %.3f\n", nodeitr, pos_aspos_rate, neg_aspos_rate, node_threshold);
			WEAKCLASSIFIER thr; thr.fAlpha = -1; thr.fError = -1; thr.fThr = -1; thr.iDim = -1; thr.iDirection = -1;
			thr.fThreshold = node_threshold; nc.push_back(thr);
			break;
		}
		else{
			printf("nodeitr=%d, pos_aspos_rate=%.3f, neg_aspos_rate=%.3f, node_threshold=%.3f\n", wc.iDim, pos_aspos_rate, neg_aspos_rate, node_threshold);
		}
	}
}

void AdaBoost_train_cascade(vector<vector<WEAKCLASSIFIER>> &sc, CvMat *features_matrix, vector<int> &vecLabel, vector<WEAKCLASSIFIER> &wc_pool, const char *sPath)
{
	int num_pos = 0;
	while (vecLabel[num_pos] != -1)
		num_pos++;
	int num_neg = vecLabel.size() - num_pos;
	CvMat *features_matrix_pos = cvCreateMat(num_pos, features_matrix->cols, CV_32FC1);
	for (int kx = 0;kx < num_pos; kx++){
		for (int j = 0;j < features_matrix_pos->cols;j++)
			features_matrix_pos->data.fl[kx*features_matrix_pos->cols + j] = features_matrix->data.fl[kx*features_matrix->cols+j];
	}
	CvMat *features_matrix_neghard = cvCreateMat(num_neg, features_matrix->cols, CV_32FC1);
	for (int kx = num_pos;kx < num_pos + num_neg; kx++){
		for (int j = 0;j < features_matrix_neghard->cols;j++)
			features_matrix_neghard->data.fl[(kx-num_pos)*features_matrix_neghard->cols + j] = features_matrix->data.fl[kx*features_matrix->cols+j];
	}
	vector<int> vecLabel_parts;

	int itr = 0;
	do{
		printf("***********itr=%d************\n", itr);
		CvMat *features_matrix_neg = cvCreateMat(MIN(num_pos, features_matrix_neghard->rows), features_matrix->cols, CV_32FC1);
		for (int kx = 0;kx < MIN(num_pos, features_matrix_neghard->rows); kx++)
			for (int j = 0;j < features_matrix_neg->cols;j++)
				features_matrix_neg->data.fl[kx*features_matrix_neg->cols+j] = features_matrix_neghard->data.fl[kx*features_matrix_neghard->cols+j];

		CvMat *features_matrix_trainparts  = MergeMatrixByRows(features_matrix_pos, features_matrix_neg, 0);
		vector<WEAKCLASSIFIER> nc; vecLabel_parts.clear();
		for (int kx = 0;kx < features_matrix_pos->rows;kx++)
			vecLabel_parts.push_back(1);
		for (int kx = 0;kx< features_matrix_neg->rows;kx++)
			vecLabel_parts.push_back(-1);
		float node_threshold = 0;
		AdaBoost_train_node(nc, features_matrix_trainparts, vecLabel_parts, wc_pool);
		sc.push_back(nc);
		WriteClassifier(sc, sPath);
		
		cvReleaseMat(&features_matrix_trainparts);
		cvReleaseMat(&features_matrix_neg);
		CvMat *features_matrix_neghard_update = UpdateNegSamples(sc, features_matrix_neghard);
		if (features_matrix_neghard_update != NULL){
			cvReleaseMat(&features_matrix_neghard);
			features_matrix_neghard = cvCloneMat(features_matrix_neghard_update); 
			cvReleaseMat(&features_matrix_neghard_update);
		}
		else{
			break;
		}

		itr++;
	}
	while(itr < 20);

	cvReleaseMat(&features_matrix_neghard);
	cvReleaseMat(&features_matrix_pos);
}

int Adaboost_apply_cascade(CvMat *feature_vector, vector<vector<WEAKCLASSIFIER>> &sc)
{
	int iret = 1;
	for (int nx = 0;nx < sc.size();nx++)
	{
		vector<WEAKCLASSIFIER> &nc = sc[nx];
		WEAKCLASSIFIER thr = nc[nc.size()-1];
		vector<WEAKCLASSIFIER> nc_body;
		for (int j = 0;j < nc.size()-1;j++)
			nc_body.push_back(nc[j]);
		int inoderet = Adaboost_apply(feature_vector, nc_body, thr.fThreshold);
		if (inoderet == -1){
			iret = -1;
			break;
		}
	}
	return iret;
}

CvMat *UpdateNegSamples(vector<vector<WEAKCLASSIFIER>> &sc, CvMat *features_matrix_neghard)
{
	CvMat *features_matrix_neghard_update = NULL;
	for (int kx = 0;kx < features_matrix_neghard->rows;kx++)
	{
		CvMat *feature_vector = cvCreateMat(1, features_matrix_neghard->cols, CV_32FC1);
		for (int j = 0;j < features_matrix_neghard->cols;j++)
			feature_vector->data.fl[j] = features_matrix_neghard->data.fl[kx*features_matrix_neghard->cols+j];

		int iret = Adaboost_apply_cascade(feature_vector, sc);
		if (iret == 1)
			features_matrix_neghard_update = MergeMatrixByRows(features_matrix_neghard_update, feature_vector);
	}
	return features_matrix_neghard_update;
}

int  EvalNodeClassifier(float *pos_aspos_rate, float *neg_aspos_rate, float &node_threshold, vector<WEAKCLASSIFIER> &nc, CvMat *features_matrix, vector<int> &vecLabel)
{
	WEAKCLASSIFIER &wc_last = nc[nc.size()-1];
	float node_threshold_update1 = node_threshold + wc_last.fAlpha*(-1);
	float node_threshold_update2 = node_threshold + wc_last.fAlpha*(1);

	vector<int> vecResults1, vecResults2;
	for (int kx = 0;kx < features_matrix->rows;kx++)
	{
		CvMat *feature_vector = cvCreateMat(1, features_matrix->cols, CV_32FC1);
		for (int j = 0;j < features_matrix->cols;j++)
			feature_vector->data.fl[j] = features_matrix->data.fl[kx*features_matrix->cols+j];

		int w = 0;
		float fHsum = 0;
		for (w = 0;w < nc.size();w++)
		{
			WEAKCLASSIFIER &wc = nc[w];
			float fval = feature_vector->data.fl[wc.iDim];
			if (wc.iDirection == 1)
			{
				if (fval >= wc.fThreshold)
					fHsum += wc.fAlpha * 1;
				else
					fHsum += wc.fAlpha * (-1);
			}
			else if (wc.iDirection == -1)
			{
				if (fval >= wc.fThreshold)
					fHsum += wc.fAlpha * (-1);
				else
					fHsum += wc.fAlpha * 1;
			}
		}

		if (fHsum >= node_threshold_update1)
			vecResults1.push_back(1);
		else
			vecResults1.push_back(-1);

		if (fHsum >= node_threshold_update2)
			vecResults2.push_back(1);
		else
			vecResults2.push_back(-1);

		cvReleaseMat(&feature_vector);
	}

	int total_pos = 0; int result_pos_aspos = 0; int total_neg = 0; int result_neg_aspos = 0;
	for (int kx = 0;kx < vecLabel.size();kx++)
	{
		if (vecLabel[kx] == 1){
			total_pos++;
			if (vecResults2[kx] == 1)
				result_pos_aspos++;
		}
		if (vecLabel[kx] == -1){
			total_neg++;
			if (vecResults2[kx] == 1){
				result_neg_aspos++;
			}
		}
	}
	if (1.0 * result_pos_aspos / total_pos >= pos_limit){
		*pos_aspos_rate = 1.0 * result_pos_aspos / total_pos;
		*neg_aspos_rate = 1.0*result_neg_aspos/total_neg;
		node_threshold = node_threshold_update2;
		return 2;
	}

    total_pos = 0;  result_pos_aspos = 0;  total_neg = 0;  result_neg_aspos = 0;
	for (int kx = 0;kx < vecLabel.size();kx++)
	{
		if (vecLabel[kx] == 1){
			total_pos++;
			if (vecResults1[kx] == 1)
				result_pos_aspos++;
		}
		if (vecLabel[kx] == -1){
			total_neg++;
			if (vecResults1[kx] == 1){
				result_neg_aspos++;
			}
		}
	}
	if (1.0 * result_pos_aspos / total_pos >= pos_limit){
		*pos_aspos_rate = 1.0 * result_pos_aspos / total_pos;
		*neg_aspos_rate = 1.0*result_neg_aspos/total_neg;
		node_threshold = node_threshold_update1;
		return 1;
	}
}

void TrainClassifier_cascade(vector<vector<WEAKCLASSIFIER>> &sc, CvMat *features_matrix, CvMat *label, const char *sPath)
{
	vector<WEAKCLASSIFIER> wc_pool;
	GetWeakClassifierPool(wc_pool, features_matrix);
	vector<int> vecLabel;
	for (int s = 0;s < label->rows;s++)
		vecLabel.push_back(label->data.i[s]);

	AdaBoost_train_cascade(sc, features_matrix, vecLabel, wc_pool, sPath);
}

void WriteClassifier(vector<vector<WEAKCLASSIFIER>> &sc, const char *sPath)
{
	FILE *fp = fopen(sPath, "w");
	for (int nx = 0;nx < sc.size(); nx++)
	{
		vector<WEAKCLASSIFIER> &nc = sc[nx];
		fprintf(fp, "%d\n", nx);
		for (int k = 0;k < nc.size()-1;k++)
		{
			WEAKCLASSIFIER &wc = nc[k];
			fprintf(fp, "%d, %d, %d, %f, %f, %f\n", k, wc.iDim, wc.iDirection, wc.fThreshold, wc.fAlpha, wc.fError);
		}
		fprintf(fp, "@, %f\n", (nc[nc.size()-1]).fThreshold);
	}
	fprintf(fp, "#\n");
	fclose(fp);
}

void ReadClassifier(vector<vector<WEAKCLASSIFIER>> &sc, const char *sPath)
{
	FILE *fp = fopen(sPath, "r");

	while (1)
	{
		vector<WEAKCLASSIFIER> nc;
		char buff[200];
		string sz = fgets(buff, 200, fp);
		sz = sz.substr(0, sz.length()-1);
		if (!strcmp(sz.c_str(), "#"))
			break;
		else {
			while(1)
			{
				int k;
				WEAKCLASSIFIER wc;
				char buff[200];
				string sz = fgets(buff, 200, fp);
				sz = sz.substr(0, sz.length()-1);
				if (!strcmp((sz.substr(0, 1)).c_str(), "@")){
					sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); 
					wc.fThreshold = atof(sz.c_str());
					wc.fAlpha = -1; wc.fError = -1; wc.fThr = -1; wc.iDim = -1; wc.iDirection = -1;
					nc.push_back(wc);
					break;
				}

				string s0 = sz.substr(0, sz.find(", "));
				sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s1 = sz.substr(0, sz.find(", "));
				sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s2 = sz.substr(0, sz.find(", "));
				sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s3= sz.substr(0, sz.find(", "));
				sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s4= sz.substr(0, sz.find(", "));
				sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s5= sz.substr(0, sz.find(", "));

				wc.iDim = atoi(s1.c_str());
				wc.iDirection = atoi(s2.c_str());
				wc.fThreshold = atof(s3.c_str());
				wc.fAlpha = atof(s4.c_str());
				wc.fError = atof(s5.c_str());

				nc.push_back(wc);
			}
		}
		sc.push_back(nc);
	}

	fclose(fp);
}

int  ApplyClassifier(CvMat *feature_vector,  vector<vector<WEAKCLASSIFIER>> &sc)
{
	return Adaboost_apply_cascade(feature_vector, sc);
}

void AdaBoost_train(vector<WEAKCLASSIFIER> &sc, CvMat *features_matrix, vector<int> &vecLabel, vector<WEAKCLASSIFIER> &wc_pool)
{
	int numSamples = features_matrix->rows;
	int numFeatures = features_matrix->cols;

	vector<float> vecWeight; vecWeight.resize(numSamples);
	for (int s = 0;s < numSamples;s++)
		vecWeight[s] = 1.0 / numSamples;

	int itr = 0;
	for (; itr < 200; itr++)
	{
		ClearWCpool(wc_pool);
		float minerr = 100000.0; int optwix = 0;
		for (int wix = 0; wix < wc_pool.size(); wix++)
		{
			WEAKCLASSIFIER &wc = wc_pool[wix];
			 vector<int> vecResult;
			ApplyWeakClassifier(wc, vecResult, features_matrix, vecLabel, vecWeight);
			if (wc.fError < minerr)
			{
				minerr = wc.fError;
				optwix = wix;
			}
		}

		printf("itr=%d\n", itr);

		WEAKCLASSIFIER &wc = wc_pool[optwix];
		sc.push_back(wc);

		vector<int> vecResult;
		ApplyWeakClassifier(wc, vecResult, features_matrix, vecLabel, vecWeight);
		float fsum = 0;
		for (int s = 0;s < numSamples;s++)
		{
			vecWeight[s] = vecWeight[s] * exp(-wc.fAlpha*vecResult[s]*vecLabel[s]);
			fsum = fsum + vecWeight[s];
		}
		for (int s = 0;s < numSamples;s++)
			vecWeight[s] = vecWeight[s] / fsum;

		if (minerr == 0)
			break;
	}
}

void GetWeakClassifierPool(vector<WEAKCLASSIFIER> &wc_pool, CvMat *features_matrix)
{
	for (int dim = 0;dim < features_matrix->cols;dim++)
	{
		for (int dir = 0;dir < 2;dir++)
		{
			for (int thr = 10;thr < THRSH_SPACE-9; thr++)
			{
				WEAKCLASSIFIER wc;
				wc.iDim = dim;
				wc.iDirection = dir*2-1;
				wc.fThr = 1.0*thr / THRSH_SPACE;
				wc.fAlpha = 0.0;
				wc.fError = 0.0;
				wc.fThreshold = 0.0;
				wc_pool.push_back(wc);
			}
		}
	}
}

void ApplyWeakClassifier(WEAKCLASSIFIER &wc, vector<int> &vecResult, CvMat *features_matrix, vector<int> &vecLabel, vector<float> &vecWeight)
{
	int numSamples = features_matrix->rows;
	int numFeatures = features_matrix->cols;

	vector<float> feature_dim, feature_dim_sort;
	for (int s = 0;s < numSamples;s++)
	{
		feature_dim.push_back(features_matrix->data.fl[s*numFeatures+wc.iDim]);
		feature_dim_sort.push_back(features_matrix->data.fl[s*numFeatures+wc.iDim]);
	}
	std::sort(feature_dim_sort.begin(), feature_dim_sort.end());
	float fval_thresh = feature_dim_sort[cvFloor(numSamples*wc.fThr)];

	vecResult.resize(vecLabel.size());

	if (wc.iDirection == 1)
	{
		for (int s = 0;s < numSamples;s++)
		{
			if (feature_dim[s] >= fval_thresh)
				vecResult[s] = 1;
			else
				vecResult[s] = -1;
		}
	}
	else if (wc.iDirection == -1)
	{
		for (int s = 0;s < numSamples;s++)
		{
			if (feature_dim[s] >= fval_thresh)
				vecResult[s] = -1;
			else
				vecResult[s] = 1;
		}
	}

	wc.fError = 0;
	for (int s = 0; s < numSamples; s++)
		wc.fError = wc.fError + abs(vecLabel[s] - vecResult[s]) / 2 * vecWeight[s];
	if (wc.fError == 0)
		wc.fAlpha = 0.5 * log((1-wc.fError)/ 1e-5);
	else
		wc.fAlpha = 0.5 * log((1-wc.fError)/(wc.fError));

	wc.fThreshold = fval_thresh;
}

int  Adaboost_apply(CvMat *feature_vector,  vector<WEAKCLASSIFIER>  &sc, float sc_thr)
{
	int iClassifyResult;

	int w = 0;
	float fHsum = 0;
	for (w = 0;w < sc.size();w++)
	{
		WEAKCLASSIFIER &wc = sc[w];
		float fval = feature_vector->data.fl[wc.iDim];
		if (wc.iDirection == 1)
		{
			if (fval >= wc.fThreshold)
				fHsum += wc.fAlpha * 1;
			else
				fHsum += wc.fAlpha * (-1);
		}
		else if (wc.iDirection == -1)
		{
			if (fval >= wc.fThreshold)
				fHsum += wc.fAlpha * (-1);
			else
				fHsum += wc.fAlpha * 1;
		}
	}

	if (fHsum >= sc_thr)
		iClassifyResult = 1;
	else
		iClassifyResult = -1;

	return iClassifyResult;
}

void ClearWCpool(vector<WEAKCLASSIFIER> &wc_pool)
{
	for (int k = 0;k < wc_pool.size();k++)
	{
		WEAKCLASSIFIER &wc = wc_pool[k];
		wc.fAlpha = 0.0;
		wc.fError = 0.0;
	}
}

void TrainClassifier(vector<WEAKCLASSIFIER> &sc, CvMat *features_matrix, CvMat *label)
{
	vector<WEAKCLASSIFIER> wc_pool;
	GetWeakClassifierPool(wc_pool, features_matrix);
	vector<int> vecLabel;
	for (int s = 0;s < label->rows;s++)
		vecLabel.push_back(label->data.i[s]);

	AdaBoost_train(sc, features_matrix, vecLabel, wc_pool);
}

void WriteClassifier(vector<WEAKCLASSIFIER> &sc, const char *sPath)
{
	FILE *fp = fopen(sPath, "w");
	for (int k = 0;k < sc.size();k++)
	{
		WEAKCLASSIFIER &wc = sc[k];
		fprintf(fp, "%d, %d, %d, %f, %f, %f\n", k, wc.iDim, wc.iDirection, wc.fThreshold, wc.fAlpha, wc.fError);
	}

	fprintf(fp, "#\n");
	fclose(fp);
}

void ReadClassifier(vector<WEAKCLASSIFIER> &sc, const char *sPath)
{
	FILE *fp = fopen(sPath, "r");

	while(1)
	{
		int k;
		WEAKCLASSIFIER wc;
		char buff[200];
		string sz = fgets(buff, 200, fp);
		sz = sz.substr(0, sz.length()-1);
		if (!strcmp(sz.c_str(), "#"))
			break;

		string s0 = sz.substr(0, sz.find(", "));
		sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s1 = sz.substr(0, sz.find(", "));
		sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s2 = sz.substr(0, sz.find(", "));
		sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s3= sz.substr(0, sz.find(", "));
		sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s4= sz.substr(0, sz.find(", "));
		sz = sz.substr(sz.find(", ")+2, sz.length()-sz.find(", ")-2); string s5= sz.substr(0, sz.find(", "));

		wc.iDim = atoi(s1.c_str());
		wc.iDirection = atoi(s2.c_str());
		wc.fThreshold = atof(s3.c_str());
		wc.fAlpha = atof(s4.c_str());
		wc.fError = atof(s5.c_str());

		sc.push_back(wc);
	}

	fclose(fp);
}

int  ApplyClassifier(CvMat *feature_vector,  vector<WEAKCLASSIFIER> &sc)
{
	return Adaboost_apply(feature_vector, sc, 0);
}
