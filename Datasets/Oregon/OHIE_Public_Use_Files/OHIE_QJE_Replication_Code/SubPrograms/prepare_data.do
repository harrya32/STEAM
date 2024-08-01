
* ***********************************************
* Merging Data
* ***********************************************

use Data/oregonhie_descriptive_vars.dta, clear
foreach dat in 0m 6m 12m {
merge 1:1 person_id using Data/oregonhie_survey`dat'_vars.dta, assert(match master) nogenerate
}
merge 1:1 person_id using Data/oregonhie_stateprograms_vars.dta, assert(match) nogenerate

***********************************************
*Rename insurance variables for use with paper replication code
***********************************************

rename ohp_all_ever_matchn_30sep2009 ohp_all_ever_admin 
rename ohp_all_mo_matchn_30sep2009 ohp_all_mo_admin 
rename ohp_all_mo_firstn_30sep2009 ohp_all_mo_survey 
rename ohp_all_ever_firstn_30sep2009 ohp_all_ever_survey 
rename ohp_all_ever_firstn_survey0m ohp_all_ever_survey0m 
rename ohp_all_ever_firstn_survey6m ohp_all_ever_survey6m 
rename ohp_all_end_30sep2009 ohp_all_end_admin 
rename ohp_all_mo_firstn_survey0m ohp_all_mo_survey0m 
rename ohp_all_mo_firstn_survey6m ohp_all_mo_survey6m 

gen ohp_all_end_survey=ohp_all_end_admin

rename ohp_std_ever_matchn_30sep2009 ohp_std_ever_admin 
*rename ohp_std_mo_matchn_30sep2009 ohp_std_mo_admin 
rename ohp_std_ever_firstn_30sep2009 ohp_std_ever_survey 
*rename ohp_std_mo_firstn_survey0m ohp_std_mo_survey0m
*rename ohp_std_mo_firstn_survey6m ohp_std_mo_survey6m

rename tanf_ever_matchn_30sep2009 postn_tanf_bin 
rename tanf_ever_prenotify07 prenany_tanf_bin 
rename tanf_ever_firstn_survey12m postn_survey12m_tanf_bin 
rename tanf_ever_presurvey12m pren_survey12m_tanf_bin
rename tanf_tot_hh_firstn_survey12m postn_survey12m_tanf_hh_amt 
rename tanf_tot_hh_30sep2009 postn_tanf_hh_amt 
rename tanf_tot_hh_prenotify07 prenany_tanf_hh_amt 
rename tanf_tot_hh_presurvey12m pren_survey12m_tanf_hh_amt 

rename snap_ever_matchn_30sep2009 postn_snap_bin
rename snap_ever_presurvey12m pren_survey12m_snap_bin 
rename snap_tot_hh_30sep2009 postn_snap_hh_amt
rename snap_tot_hh_prenotify07 prenany_snap_hh_amt 
rename snap_tot_hh_presurvey12m pren_survey12m_snap_hh_amt 
*rename ohp_all_at_survey12m ohp_all_at_12m\
rename snap_ever_prenotify07 prenany_snap_bin 
rename snap_ever_firstn_survey12m postn_survey12m_snap_bin
rename snap_tot_hh_firstn_survey12m postn_survey12m_snap_hh_amt

rename zip_msa_list zip_msa

rename wave_survey0m draw
rename wave_survey12m draw_survey_12m

* ***********************************************
* Fixed effects for regressions
* ***********************************************

	gen noweight = 1
	
	gen constant = 1

	xi, prefix(ddd) i.draw_survey_12m*i.numhh_list
	desc ddd* 
	drop ddddraXnum_4_3 ddddraXnum_5_3 ddddraXnum_6_3 ddddraXnum_7_3 
	
	xi, prefix(lll) i.draw_lottery
	desc lll*

	xi, prefix(nnn) i.numhh_list
	desc nnn*
	
gen weight_0m = 1
replace weight_0m =. if sample_0m!=1
gen sample_0m_resp = (returned_0m==1 & weight_0m!=0)
gen sample_6m_resp = (returned_6m==1 & weight_6m!=0)

xi, prefix(eee) i.draw*i.numhh_list
desc eee* 
drop eeedraXnum_4_3 eeedraXnum_5_3 eeedraXnum_6_3 eeedraXnum_7_3 eeedraXnum_8_3 


* ************************************************
* Other Variables
* ************************************************

foreach x in 0m 6m {
gen health_poor_`x' = (health_gen_`x'==1) if !missing(health_gen_`x')

recode health_gen_bin_`x' (1=0) (0=1), gen(health_genflip_bin_`x')
recode health_poor_`x' (1=0) (0=1), gen(health_notpoor_`x')
recode health_chg_bin_`x' (1=0) (0=1), gen(health_chgflip_bin_`x')
foreach stub in tot phys ment {
	gen notbaddays_`stub'_`x'=30-baddays_`stub'_`x'
	}
}

count

 gen older = (birthyear_list >= 1945 & birthyear_list <= 1958)
 label var older "Age 50-64 in 2008"
 
 gen younger = (birthyear_list >= 1959 & birthyear_list <= 1989)
 label var younger "Age 19-49 in 2008"
 
 gen female_cb = (female_list==1 & birthyear_list>1968 & birthyear_list<=1989) // females aged 19-40
 label var female_cb "females aged 19-40 in 2008"
 
 gen first_week_list = week_list==1
 label var first_week_list "Signed up for lottery during first week: lottery list"
 
 gen scaled_week_list=week_list-1
 label var scaled_week_list "Week of lottery sign up - 1"
	
	
********************************
*12m timing variables
********************************

foreach x in 0m 6m 12m {
replace dt_returned_`x' = round(dt_returned_`x')
}
gen mail_to_response_12m = dt_returned_12m - dt_mail_12m
tab mail_to_response_12 // value <= are likely entry mistakes, replace them with average 
egen mail_to_response_12_mean = mean(mail_to_response_12m) if mail_to_response_12m>=0, by(draw)
egen mail_to_response_12_maxmean = max(mail_to_response_12_mean), by(draw) // fill in missing values
replace mail_to_response_12m=round(mail_to_response_12_maxmean) if mail_to_response_12m<=0
drop mail_to_response_12_mean mail_to_response_12_maxmean

tab mail_to_response_12


gen sample_hdd=1

gen zip_hh_inc_list=0


********************************************
*12m Survey variables recoded for QJE Paper
********************************************

/* health poor */
	gen health_poor_12m = (health_gen_12m==1) if !missing(health_gen_12m)

* education
	tab edu_12m, gen(edu_12m_)
   * inc as % of FPL
	recode hhinc_pctfpl_12m (min/50 = 1 "below 50% of FPL") (50/75 = 2 "50-75% FPL") (75/100 = 3 "75-100% FPL") (100/150 = 4 "100-150% of FPL") (150/max = 5 "above 150% of FPL"), gen(fpl_categ_12m)
	tab fpl_categ_12m, gen(fpl_categ_12m_)
	* employment - moved in survey data cleaning
	tab employ_hrs_12m, gen(employ_hrs_12m_)

	recode health_gen_bin_12m (1=0) (0=1), gen(health_genflip_bin_12m)
	recode health_chg_bin_12m (1=0) (0=1), gen(health_chgflip_bin_12m)
	
	recode health_poor_12m (1=0) (0=1), gen(health_notpoor_12m)

	foreach stub in tot phys ment {
	gen notbaddays_`stub'_12m=30-baddays_`stub'_12m
								}
	recode happiness_12m (1/2 = 0 "very/pretty happy") (3 = 1 "not too happy"), gen(happiness_bin_12m)
	recode happiness_bin_12m (1=0) (0=1), gen(poshappiness_bin_12m)

	/* Recode preventive care vars */


	recode chl_chk_12m (1/2 = 1 "yes") (3 = 0 "never"), gen(chl_chk_bin_12m)
	recode dia_chk_12m (1/2 = 1 "yes") (3 = 0 "never"), gen(dia_chk_bin_12m)

	recode mam_chk_12m (1 = 1 "yes, last year") (2/3 = 0 "more than 1yr/no"), gen(mam_chk_bin_12m)
	replace mam_chk_bin_12m = . if birthyear_list>1968	//mamogram only >=40

	recode pap_chk_12m (1 = 1 "yes, last year") (2/3 = 0 "more than 1yr/no"), gen(pap_chk_bin_12m)
	

	/* Recode health behavior variables */
	recode smk_curr_12m (1/2 = 1 "every day/some days") (3 = 0 "not at all"), gen(smk_curr_bin_12m)
	recode smk_curr_bin_12m (1=0) (0=1), gen(nonsmk_curr_12m)
	recode physical_act_12m  (1/2 = 0 "more/same") (3 = 1 "less") (nonmissing = .) (missing = .), gen(physical_act_bin_12m)
	recode physical_act_bin_12m (1=0) (0=1), gen(more_active_12m)
	
	
	gen not_er_noner_0m=0 if !missing(er_noner_0m)
	replace not_er_noner_0m=1 if er_noner_0m==0
	
	gen not_er_noner_6m=0 if !missing(er_noner_6m)
	replace not_er_noner_6m=1 if er_noner_6m==0
	
	gen not_er_noner_12m=0 if !missing(er_noner_12m)
	replace not_er_noner_12m=1 if er_noner_12m==0
	
	gen dep_screen_6m=(dep_interest_6m + dep_sad_6m)>=5 if dep_interest_6m~=. & dep_sad_6m~=.
	gen dep_screen_12m=(dep_interest_12m + dep_sad_12m)>=5 if dep_interest_12m~=. & dep_sad_12m~=.
	
	recode dep_screen_12m (1=0) (0=1), gen(nodep_screen_12m)
	
	qui: sum cost_tot_oop_12m, det
	local cost_oop_cutoff = 2*r(p99)
	gen cost_tot_oop_mod_12m = cost_tot_oop_12m if cost_tot_oop_12m <= `cost_oop_cutoff' & cost_tot_oop_12m >= 0
	
	qui: sum cost_tot_owe_12m, det 
	local cost_owe_cutoff = 2*r(p99)
	gen cost_tot_owe_mod_12m = cost_tot_owe_12m if cost_tot_owe_12m<= `cost_owe_cutoff' & cost_tot_owe_12m >= 0
	
	gen hhinc_mid_12m = hhinc_cat_12m
	replace hhinc_mid_12m = 0 if hhinc_mid_12m == 1
	replace hhinc_mid_12m = 1250 + 2500 * (hhinc_mid_12m - 2) if hhinc_mid_12m !=0
	replace hhinc_mid_12m = 50001 if hhinc_cat_12m==22
	
	
	* recode death var
	recode postn_death (1=0) (0=1), gen(postn_alive)
	label var postn_alive "Alive post-notification - based on Oregon VitalStats data"
	label define alive 0 "Dead" 1 "Alive"
	label values postn_alive alive


	
save Data/data_for_analysis.dta, replace


