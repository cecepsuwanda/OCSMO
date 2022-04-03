#include "Tconf_metrix.h"

Tconf_metrix::Tconf_metrix()
{
	
	failed = 0;
	tepat = 0;
	tdk_tepat = 0;
	jml_data = 0;
}

Tconf_metrix::~Tconf_metrix()
{
	
	map<string, map<string, int>>::iterator it, it1;

	if (matrik.size() > 0)
	{
		for (it = matrik.begin(); it != matrik.end(); ++it)
		{
			it->second.clear();
		}
		matrik.clear();
	}

	matrik1.clear();
	label.clear();
}

void Tconf_metrix::add_jml(string asli, string tebakan, int jml)
{
	std::lock_guard<std::mutex> lock(v_mutex);

	if (konversi_asli.size() > 0)
	{
		auto itr = konversi_asli.find(asli);
		if (itr != konversi_asli.end())
		{
			asli = konversi_asli[asli];
		}
	}

	if (konversi_tebakan.size() > 0)
	{
		auto itr = konversi_tebakan.find(tebakan);
		if (itr != konversi_tebakan.end())
		{
			tebakan = konversi_tebakan[tebakan];
		}
	}

	if ((tebakan.compare("dfs_failed.") != 0))
	{
		
        auto itr = label.find(asli);
        if (itr == label.end())
        {
          label.insert(pair<string, int>(asli, 0));	
        }

        auto itr1 = label.find(tebakan);
        if (itr1 == label.end())
        {
          label.insert(pair<string, int>(tebakan, 0));	
        }
				

		jml_data++;
		if (asli == tebakan)
		{
			tepat++;
		} else {
			tdk_tepat++;
		}

		auto it = matrik.find(asli);
		if (it == matrik.end())
		{
			map<string, int> tmp_map;
			tmp_map.insert(pair<string, int>(tebakan, jml));
			matrik.insert(pair<string, map<string, int>>(asli, tmp_map));
		} else {
			auto it1 = it->second.find(tebakan);
			if (it1 == it->second.end())
			{
				it->second.insert(pair<string, int>(tebakan, jml));
			} else {
				it1->second += jml;
			}
		}



	} else {
		failed++;
	}

   

}

void Tconf_metrix::kalkulasi()
{
	
	if (matrik.size() > 0) {

		int TP = 0;
		int FN = 0;
		int FP = 0;
		int TN = 0;

		for (auto it = label.begin(); it != label.end(); ++it)
		{
			Tdata tmp_data;
			tmp_data.TP = get_TP(it->first);
			TP += tmp_data.TP;
			tmp_data.FN = get_FN(it->first);
			FN += tmp_data.FN;
			tmp_data.FP = get_FP(it->first);
			FP += tmp_data.FP;
			tmp_data.TN = get_TN(it->first);
			TN += tmp_data.TN;
			tmp_data.jml = tmp_data.TP + tmp_data.FN;
			tmp_data.accuracy = 0;
			if ((tmp_data.TP + tmp_data.FP) > 0) {
			 	 tmp_data.accuracy =  (tmp_data.TP + tmp_data.TN) / (double)(tmp_data.TP + tmp_data.FP + tmp_data.TN + tmp_data.FN);
			}
			tmp_data.recall = 0;
			if ((tmp_data.jml) > 0) {
				tmp_data.recall =  tmp_data.TP / (double) tmp_data.jml;
			}

			tmp_data.precision = 0;
			if ((tmp_data.TP+tmp_data.FP) > 0) {
				tmp_data.precision =  tmp_data.TP / ((double) (tmp_data.TP+tmp_data.FP));
			}

			// tmp_data.specificity = 0;
			// if ((tmp_data.TN + tmp_data.FP) > 0) {
			// 	tmp_data.specificity =  tmp_data.TN / (double)(tmp_data.TN + tmp_data.FP);
			// }

			tmp_data.f1 = 0;
			if ((tmp_data.recall + tmp_data.precision) > 0) {
			  	tmp_data.f1 =  (2*tmp_data.recall*tmp_data.precision) / (double)(tmp_data.recall + tmp_data.precision);
			}

			matrik1.insert(pair<string, Tdata> (it->first, tmp_data));
		}

		accuracy = (TP + TN) / (double)(TP + TN + FP + FN);
		precision = TP / (double)(TP + FP);
		recall = TP / (double)(TP + FN);
		f1 = (2*recall*precision)/(double)(recall+precision);
	}
}

int Tconf_metrix::get_TP(string kelas)
{
	
	int total = 0;

	if (matrik.size() > 0) {
		auto it = matrik.find(kelas);
		if (it != matrik.end())
		{
			auto it1 = it->second.find(kelas);
			if (it1 != it->second.end())
			{
				total = it1->second;
			}
		}
	}

	return total;
}

int Tconf_metrix::get_TN(string kelas)
{
	
	int total = 0;

	if (matrik.size() > 0) {
		for (auto it = matrik.begin(); it != matrik.end(); ++it)
		{
			if (it->first != kelas)
			{
				for (auto it1 = it->second.begin(); it1 != it->second.end(); ++it1)
				{
					if (it1->first != kelas) {
						total += it1->second;
					}
				}
			}
		}
	}

	return total;
}

int Tconf_metrix::get_FP(string kelas)
{
	
	int total = 0;

	if (matrik.size() > 0) {

		for (auto it = matrik.begin(); it != matrik.end(); ++it)
		{
			if (it->first != kelas) {
				auto it1 = it->second.find(kelas);
				if (it1 != it->second.end()) {
					total += it1->second;
				}
			}
		}
	}

	return total;
}

int Tconf_metrix::get_FN(string kelas)
{
	//std::lock_guard<std::mutex> lock(v_mutex);
	int total = 0;

	if (matrik.size() > 0) {
		auto it = matrik.find(kelas);
		if (it != matrik.end())
		{
			for (auto it1 = it->second.begin(); it1 != it->second.end(); ++it1)
			{
				if (it1->first != kelas) {
					total += it1->second;
				}
			}
		}
	}

	return total;
}

float Tconf_metrix::get_F1()
{
	return f1;
}

ostream & operator << (ostream &out, const Tconf_metrix &tc)
{
	out << "Jumlah Data: " << tc.jml_data;
	out << " Prediksi Tepat: " << tc.tepat;
	out << " Prediksi Tidak Tepat: " << tc.tdk_tepat;
	out << " Prediksi Failed: " << tc.failed;
	//out << " Prosentase: " << ((tc.tepat / (double) tc.jml_data) * 100);
	out << " Akurasi: " << tc.accuracy ;
	out << " Precision: " << tc.precision ;
	out << " Recall: " << tc.recall ;
	out << " F1 : " << tc.f1 << endl;

	if (tc.matrik1.size() > 0) {
		out << "   Confusion Metrik     : " << endl;
		out << setw(30) << "kelas" << setw(10) << "TP" << setw(10) << "FN" << setw(10) << "jml" << setw(10) << "FP" << setw(10) << "TN" << setw(10) << "accuracy" << setw(10)  << "recall" << setw(13) << "precision" << setw(13) << "f1" << endl; 
		for (auto it = tc.matrik1.begin(); it != tc.matrik1.end(); ++it)
		{
			out << setw(30) << it->first;
			out << setw(10) << it->second.TP;
			out << setw(10) << it->second.FN;
			out << setw(10) << it->second.jml;
			out << setw(10) << it->second.FP;
			out << setw(10) << it->second.TN;
			out << setprecision(3) << setw(10) << it->second.accuracy;
			out << setprecision(3) << setw(10) << it->second.recall;
			out << setprecision(3) << setw(13) << it->second.precision;
			out << setprecision(3) << setw(13) << it->second.f1 << endl;
		}
	}

	out << endl;

	if (tc.matrik.size() > 0)
	{
		out << setw(20);

		for (auto it = tc.label.begin(); it != tc.label.end(); it++)
		{
			out << it->first << setw(10);
		}

		out << endl;

		for (auto it = tc.label.begin(); it != tc.label.end(); it++)
		{
			out << setw(10) << it->first;
			for (auto it1 = tc.label.begin(); it1 != tc.label.end(); it1++)
			{
				auto it2 = tc.matrik.find(it->first);
				if (it2 == tc.matrik.end())
				{
					out << setw(10) << 0;
				} else {
					auto it3 = it2->second.find(it1->first);
                    if(it3==it2->second.end()){
					   out << setw(10) << 0;
                    }else{
                       out << setw(10) << it3->second;	
                    }
				}
			}

			out << endl;

		}


	}



	return out;
}

void Tconf_metrix::save(string nm_file, string param_nm_file, int param_depth, int param_min_sample, double param_gamma, double param_nu, double param_credal_s)
{
	
	Twrite_file tmp_wf;
	tmp_wf.setnm_f(nm_file);

	for (auto it = matrik.begin(); it != matrik.end(); ++it)
	{
		string tmp_str = "";
		for (auto it1 = it->second.begin(); it1 != it->second.end(); it1++)
		{
			string tmp_str = param_nm_file + "," + to_string(param_depth) + "," + to_string(param_min_sample) + "," + to_string(param_gamma) + "," + to_string(param_nu) + "," + to_string(param_credal_s) +  "," + it->first + "," + it1->first + "," + to_string(it1->second);
			tmp_wf.write_file(tmp_str);
		}
	}

	tmp_wf.close_file();
}

void Tconf_metrix::add_konversi_asli(string dari, string ke)
{	
	konversi_asli.insert(pair<string, string>(dari, ke));
}

void Tconf_metrix::add_konversi_tebakan(string dari, string ke)
{	
	konversi_tebakan.insert(pair<string, string>(dari, ke));
}