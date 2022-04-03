#include "credal.h"

credal::credal()
{
	_s = 0.5;
}

credal::~credal()
{
	clear();
}

credal::credal(double s)
{
	_s = s;    
}

void credal::clear()
{
	_prob_int.clear();

	_max_prob.clear();
	_max_prob.shrink_to_fit();

	_min_prob.clear();
	_min_prob.shrink_to_fit();
}

double credal::entropy(std::vector<double> x) const {

	std::transform(x.begin(), x.end(), x.begin(),
	[](double d) -> double {
		return ((d > 0.0) ? (d * std::log2(d)) : 0.0);
	});
	return -std::accumulate(x.begin(), x.end(), 0.0);
}

vector<double> credal::minVals(const vector<double>& array) {

	/* Initianlizing the minimal values with a not reachable one in the array */
	double nmin = 0.0;
	double min1 = 2.0;
	double min2 = 2.0;
	bool samemin = true;
	for (const double val : array) {
		/* In case a new minimum is identified, the old minimum in 'min1'
		** is passed to 'min2' and 'min1' gets the new one */
		if (val < min1) {
			min2 = min1;
			min1 = val;
			nmin = 1;
		} else if (val == min1) {
			++nmin;
		} else if (val < min2) {
			/* In case a the value is greater than the minimum,
			** but smaller than the second minimal value, then
			** it is assigned to 'min2' */
			min2 = val;
			samemin = false;
		}
	}
	if (samemin) {
		min2 = min1;
	}
	/* return the minimum and second minimal */
	vector<double> res{min1, min2, nmin};
	return res;
}


/* Functions called by R, see description above */

vector<double> credal::maxEntropyDist(const ProbInterval &probint, const bool /*exact*/) {

	vector<double> lower = probint.lower;
	int lsize = lower.size();

	double nmin, minval, sminval;

	// Due to the nature of IDM the intial free mass is s/(N+s)
	double assignMass, freeMass = _s / (static_cast<double>(probint.obs) + _s);

	bool hasFree = true;

	// Keep iterating till all assigned
	while (hasFree) {
		// Vector of minimal values
		vector<double> minvals = minVals(lower);
		minval = minvals[0];
		sminval = minvals[1];
		nmin = minvals[2];

		if (minval == sminval || !((sminval - minval) < (freeMass / nmin))) {
			// All values in lower have the same value
			//   OR
			// not enough free mass to lift the minimum value(s) to the second minimal value
			assignMass = (freeMass / nmin);
			// We have used up all free mass, so signal end
			hasFree = false;
		} else {
			// assign as much mass to lift the minimum value(s) to the second minimal value
			assignMass = (sminval - minval);
			// substract the lifts from the free mass
			freeMass -= (assignMass * nmin);
		}
		// Update lower
		for (int i = 0; i < lsize; ++i) {
			if (lower[i] == minval) {
				lower[i] += assignMass;
			}
		}
	}
	return lower;
}


vector<double> credal::minEntropyDist(const ProbInterval &probint) {

	vector<double> lower = probint.lower;
	// get the index with the (first) maximum
	size_t index = std::distance(lower.begin(), std::max_element(lower.begin(), lower.end()));

	// if valid index then set the value of the lower boundary to the
	// value of the upper one
	if (index < lower.size()) {
		lower[index] = probint.upper[index];
	}
	return lower;
}

double credal::correctionEntropy(const vector<double>& probs, const int n) {
	if (_s > 0 && n > 0) {
		double ent = entropy(probs);
		//ent += (_s * log2(probs.size())) / (n + _s);
		return ent;
	}
	return -1;
}


ProbInterval credal::probabilityInterval(const vector<int>& classtable) {

	ProbInterval prob;
	prob.obs = std::accumulate(classtable.begin(), classtable.end(), 0);
	prob.freq.clear();
	prob.upper.clear();
	prob.lower.clear();
	double dobs = static_cast<double>(prob.obs);
	for (int classObs : classtable) {
		prob.freq.push_back(classObs);
		prob.prob.push_back(classObs / dobs);
		prob.upper.push_back((classObs + _s) / (dobs + _s));
		prob.lower.push_back(static_cast<double>(classObs) / (dobs + _s));
	}
	return prob;
}

void credal::input_frec(vector<int> frec)
{
	clear();
	_prob_int = probabilityInterval(frec);
	_max_prob = maxEntropyDist(_prob_int, true);
	_min_prob = minEntropyDist(_prob_int);
	_max_ent  = correctionEntropy(_max_prob, _prob_int.obs);
	_min_ent  = correctionEntropy(_min_prob, _prob_int.obs);
	_ent = entropy(_prob_int.prob);

}

void credal::cetak_vec(vector<double> &vec)
{
	for (int i = 0; i < vec.size(); ++i)
	{
		cout << vec[i];
		if (i < (vec.size() - 1))
		{
			cout << ",";
		} else {
			cout << endl;
		}
	}
}

void credal::info()
{
	cout << "frek : ";

	vector<int> freq = _prob_int.freq;

	for (int i = 0; i < freq.size(); ++i)
	{
		cout << freq[i];
		if (i < (freq.size() - 1))
		{
			cout << ",";
		} else {
			cout << endl;
		}
	}

	cout << "max entropy : " << _max_ent << endl;
	cout << "min entropy : " << _min_ent << endl;
	cout << "entropy     : " << _ent << endl;

	cout << "lower prob : ";
	cetak_vec(_prob_int.lower);

	cout << "upper prob : ";
	cetak_vec(_prob_int.upper);

	cout << "max prob : ";
	cetak_vec(_max_prob);

	cout << "min prob : ";
	cetak_vec(_min_prob);
}

vector<double> credal::get_max_prob()
{
	return _max_prob;
}

vector<double> credal::get_min_prob()
{
	return _min_prob;
}

double credal::get_max_ent()
{
	return _max_ent;
}

double credal::get_min_ent()
{
	return _min_ent;
}

double credal::get_ent()
{
	return _ent;
}

double credal::get_overall_metric(vector<double> ent,vector<double> max_ent)
{
  double max = *std::max_element(ent.begin(),ent.end());

  double sum = 0.0;
  for (int i = 0; i < max_ent.size(); ++i)
  {
  	if(ent[i]<max)
  	{
       sum+=(_prob_int.lower[i]*max_ent[i]);
       //cout << _prob_int.lower[i]<<","<<max_ent[i] << endl;
  	}else{
  		if(ent[i]==max)
  		{
          sum+=(_prob_int.upper[i]*max_ent[i]);
          //cout << _prob_int.upper[i]<<","<<max_ent[i] << endl;
  		}
  	}
  }

  return sum;
}




