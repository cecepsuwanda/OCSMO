#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

using namespace std;

#ifndef Included_credal_H

#define Included_credal_H

struct ProbInterval {
	int obs;
	vector<int> freq;
	vector<double> lower;
	vector<double> upper;
	vector<double> prob;

	void clear()
	{
		lower.clear();
		lower.shrink_to_fit();

		upper.clear();
		upper.shrink_to_fit();

		freq.clear();
		freq.shrink_to_fit();

		prob.clear();
		prob.shrink_to_fit();
	}

	~ProbInterval()
	{
		clear();
	}
};

class credal
{
private:
	double _s, _min_ent, _max_ent, _ent;

	ProbInterval _prob_int;
	vector<double> _max_prob;
	vector<double> _min_prob;

	double entropy(vector<double> x) const;

	vector<double> minVals(const std::vector<double>& array);

	ProbInterval probabilityInterval(const vector<int>& classtable);
	vector<double> maxEntropyDist(const ProbInterval& probint, const bool exact = true);
	vector<double> minEntropyDist(const ProbInterval& probint);
	double correctionEntropy(const vector<double>& probs, const int n);


	void cetak_vec(vector<double> &vec);

	void clear();

public:
	credal();
	~credal();
	credal(double s);
	void input_frec(vector<int> frec);

	vector<double> get_max_prob();
	vector<double> get_min_prob();


	double get_max_ent();
	double get_min_ent();
	double get_ent();
	double get_overall_metric(vector<double> ent,vector<double> max_ent);

	void info();

};

#endif