
************************************************
*12m Mail Survey Data
************************************************

clear
use Data/oregonhie_survey12m_vars.dta

*Merge in descriptive variables
merge 1:1 person_id using Data/oregonhie_descriptive_vars.dta, keepusing(household_id treatment numhh_list birthyear_list)
assert _merge==3
drop _merge

*Merge in insurance variables
merge 1:1 person_id using Data/oregonhie_stateprograms_vars.dta, keepusing(ohp_all_ever_matchn_30sep2009 ohp_all_ever_firstn_30sep2009)
assert _merge==3
drop _merge

rename ohp_all_ever_matchn_30sep2009 ohp_all_ever_admin 
rename ohp_all_ever_firstn_30sep2009 ohp_all_ever_survey

*generate, recode, or rename analysis variables
	gen constant=1

	gen dep_screen_12m=(dep_interest_12m + dep_sad_12m)>=5 if dep_interest_12m~=. & dep_sad_12m~=.
	recode dep_screen_12m (1=0) (0=1), gen(nodep_screen_12m)

	foreach x in 12m {
	gen health_poor_`x' = (health_gen_`x'==1) if !missing(health_gen_`x')

	recode health_gen_bin_`x' (1=0) (0=1), gen(health_genflip_bin_`x')
	recode health_poor_`x' (1=0) (0=1), gen(health_notpoor_`x')
	recode health_chg_bin_`x' (1=0) (0=1), gen(health_chgflip_bin_`x')
	}

	recode happiness_12m (1/2 = 0 "very/pretty happy") (3 = 1 "not too happy"), gen(happiness_bin_12m)
	recode happiness_bin_12m (1=0) (0=1), gen(poshappiness_bin_12m)

	recode chl_chk_12m (1/2 = 1 "yes") (3 = 0 "never"), gen(chl_chk_bin_12m)

	recode mam_chk_12m (1 = 1 "yes, last year") (2/3 = 0 "more than 1yr/no"), gen(mam_chk_bin_12m)
	replace mam_chk_bin_12m = . if birthyear_list>1968	// mamogram only >=40

	recode pap_chk_12m (1 = 1 "yes, last year") (2/3 = 0 "more than 1yr/no"), gen(pap_chk_bin_12m)

	recode smk_curr_12m (1/2 = 1 "every day/some days") (3 = 0 "not at all"), gen(smk_curr_bin_12m)

	rename wave_survey12m draw_survey_12m

	xi, prefix(ddd) i.draw_survey_12m*i.numhh_list
		desc ddd* 
		drop ddddraXnum_4_3 ddddraXnum_5_3 ddddraXnum_6_3 ddddraXnum_7_3 

save Data/survey12m_inperson_vars.dta, replace

clear
use Data/oregonhie_inperson_vars.dta

***************************
*Renaming/Generating variables to make the replication code run

rename sample_resp_inp sample_inp_resp
rename in_data_inp in_inperson_data
rename meds_miss_inp meds_miss_comp
rename pcs8_score_inp pcs8_score
rename mcs8_score_inp mcs8_score
rename scale_id_inp scale_id
rename stadio_id_inp stadio_id
rename omron_id_inp omron_id
rename cvd_risk_point_inp cvd_risk_point
rename any_oop_spending_inp any_oop_spending

foreach var in ast dia hbp chl ami chf emp kid cancer dep {
	rename `var'_dx_pre_lottery_inp `var'_dx_pre_lottery 
	}
	
foreach var in dia hbp chl dep {
	rename `var'_dx_post_lottery_inp `var'_dx_post_lottery 
	}

* ***********************************************
* Merging Data
* ***********************************************

merge 1:1 person_id using Data/oregonhie_descriptive_vars.dta, keepusing(treatment household_id draw_lottery numhh_list english_list self_list first_day_list have_phone_list pobox_list)
drop if _merge==2
assert _N==20745
drop _merge


merge 1:1 person_id using  Data/oregonhie_survey12m_vars.dta, keepusing(sample_12m_resp weight_12m)
drop if _merge==2
assert _N==20745
drop _merge

merge 1:1 person_id using  Data/oregonhie_stateprograms_vars.dta, keepusing(ohp_all_ever_matchn_30sep2009 ohp_all_ever_firstn_30sep2009 ///
																	ohp_all_ever_inperson ohp_std_ever_inperson ohp_all_mo_inperson ohp_all_end_inperson)
drop if _merge==2
assert _N==20745
drop _merge

rename ohp_all_ever_matchn_30sep2009 ohp_all_ever_admin 
rename ohp_all_ever_firstn_30sep2009 ohp_all_ever_survey

* ***********************************************
* Limiting Sample
* ***********************************************

count
keep if in_inperson_data == 1

*Rename variables

rename language_inp language_capi_inp


**************************
*List variables
xtile age_decile_inp=age_inp if sample_inp_resp==1, nq(10)
label var age_decile_inp "Age decile, 1 is the youngest"

tab age_decile_inp, gen(age_decile_dum_inp)
drop age_decile_dum_inp1

gen older = (age_inp>49 & age_inp<65)
gen younger = (age_inp>18 & age_inp<50)
label variable older "50 or older"
label variable younger "Younger than 50"

gen age_19_34_inp = (age_inp>18 & age_inp<35)
replace age_19_34_inp = . if age_inp==.
label var age_19_34_inp "Age 19-34"

gen age_35_49_inp = (age_inp>34 & age_inp<50)
replace age_35_49_inp = . if age_inp==.
label var age_35_49_inp "Age 35-49"

gen age_50_64_inp = (age_inp>49 & age_inp<65)
replace age_50_64_inp = . if age_inp==.
label var age_50_64_inp "Age 50-64"
	
    *Interview location
	tab interview_location_inp, gen(int_loc_cat_inp_) 
	
	*Interview season
	gen interview_season=.
	replace interview_season=1 if month(dt_completed_inp)==3 | month(dt_completed_inp)==4 | month(dt_completed_inp)==5 /* Spring*/
	replace interview_season=2 if month(dt_completed_inp)==6 | month(dt_completed_inp)==7 | month(dt_completed_inp)==8 /* Summer*/
	replace interview_season=3 if month(dt_completed_inp)==9 | month(dt_completed_inp)==10 | month(dt_completed_inp)==11 /* Fall*/
	replace interview_season=4 if month(dt_completed_inp)==12 | month(dt_completed_inp)==1 | month(dt_completed_inp)==2 /* Winter*/
	label define interview_season_lbl 1 "Spring" 2 "Summer" 3 "Fall" 4 "Winter"
	label values interview_season interview_season_lbl
	tab interview_season, gen(interview_season_)
	label variable interview_season_1 "Interviewed in the spring"
	label variable interview_season_2 "Interviewed in the summer"
	label variable interview_season_3 "Interviewed in the fall"
	label variable interview_season_4 "Interviewed in the winter"
	
	*Interview day
	gen interview_weekend=.
	replace interview_weekend=0 if dow(dt_completed_inp)<6
	replace interview_weekend=1 if dow(dt_completed_inp)==6|dow(dt_completed_inp)==7
	label variable interview_weekend "Interviewed on the weekend"
	
	*Interviewer fixed effect
	tab interviewer_inp, gen(itvr_)
	
	*Omron fixed effect
	tab omron_id, gen(omron_)
	
	*Scale fixed effect
	tab scale_id, gen(scale_)
	
	*Stadio machine fixed effect
	tab stadio_id, gen(stadio_)
	 
	
* Demographic
	*gender (change transgender to birth gender)
	replace gender_inp=0 if gender_inp==3
	replace gender_inp=1 if gender_inp==4

	*interviewed in English (CAPI is English and no interpreter)
	gen itvw_english_inp=.
	replace itvw_english_inp=1 if language_capi_inp==0 & interpreter_inp!=1 // 673 don't know if interpreter was present
	replace itvw_english_inp=0 if language_capi_inp==1 
	replace itvw_english_inp=0 if interpreter_inp==1

	tab itvw_english_inp	
	label var itvw_english_inp "Inerviewed in English"

* Medication

    *has valid med data (a response rate variable)
	gen valid_meds_inp = 1-meds_miss_comp
	label variable valid_meds_inp "Has valid medication data (or does not need medication data)"
	

* Self -Reported Health
    *low levels of pain
	gen pain_low_inp = .
	replace pain_low_inp = 1 if inlist(sf4_inp, 0, 1, 2)
	replace pain_low_inp = 0 if pain_low_inp==. & sf4_inp!=.
	label variable pain_low_inp "No or mild pain"
	
	*Health 12 months

	recode health_last12_inp (1 2 3 = 0 "0: Very poor/poor/fair") (4 5 6 = 1 "1: Good/very good/excellent"), gen(health_last12_good)
	label variable health_last12_good "Health in the last 12 mos. is good/very good/excellent"
	
	recode health_last12_inp (1 2 = 0 "0: Very poor/poor") (3 4 5 6 = 1 "Fair or better"), gen(health_last12_notbad)
	label variable health_last12_notbad "Health in the last 12 mos is not poor/very poor"
	
	*Health Change
	recode health_change_inp (0 2 = 1 "Same or better") (1 = 0 "Worse"), gen(health_change_noworse)
	label variable health_change_noworse "Health about the same or gotten better (than last year)"

* Health measurement

	*Obese
	gen obese = 0 if bmi_inp<30
	replace obese = 1 if bmi_inp>=30 & bmi_inp!=.
	label variable obese "BMI is greater than or equal to 30 (obese)"
		
	*Prehypertension or hypertension
	gen bp_prehyper = 0 if (bp_sar_inp <120 & bp_dar_inp<80)
	replace bp_prehyper = 1 if (bp_prehyper ==. & (bp_sar_inp !=. | bp_dar_inp!=.))
	label variable bp_prehyper "Prehypertensive or hypertensive (dias bp >= 80 OR sys bp >=120)" 
	
	*Hypertensive
	gen bp_hyper = 0 if (bp_sar_inp < 140 & bp_dar_inp <90)
	replace bp_hyper = 1 if (bp_hyper == . & (bp_sar_inp !=. | bp_dar_inp!=.))
	label variable bp_hyper "Hypertensive (dias bp >=90 OR sys bp >= 140)"
	
	*Diabetic A1c
	gen a1c_dia = 0 if a1c_inp < 6.5 // this is now from the MTO formula
	replace a1c_dia = 1 if a1c_inp>=6.5 & a1c_inp!=.
	label variable a1c_dia "Diabetic a1c (>=6.5%)"
	
	
	*Pre-Diabetic A1c
	gen a1c_pre_dia = 0 if a1c_inp<5.7
	replace a1c_pre_dia = 1 if a1c_inp>=5.7 & a1c_inp!=.
	label variable a1c_pre_dia "Pre diabetic or diabetic a1c (>=5.7)"

	*chl elevated
	gen chl_high = 0 if chl_inp <200 // this is now from the MTO formula
	replace chl_high = 1 if chl_inp >=200 & chl_inp!=.
	label variable chl_high "Total cholesterol is elevated(>=200)"

	*chl high
	gen chl_h = 0 if chl_inp<240 // this is now from the MTO formula
	replace chl_h = 1 if chl_inp>=240 & chl_inp!=.
	label variable chl_h "Total cholesterol is high (>=240)"
	
	*hdl low // this is now from the MTO formula
	gen hdl_low = 0 if hdl_inp >=40 & hdl_inp!=.
	replace hdl_low = 1 if hdl_inp <40
	label variable hdl_low "HDL cholesterol is low (< 40)"
	
	*hdl high // this is now from the MTO formula
	gen hdl_high = 1 if hdl_inp >= 60 & hdl_inp!=.
	replace hdl_high = 0 if hdl_inp <60
	label variable hdl_high "HDL cholesterol is high (>60)"

	*phqtot_high
	gen phqtot_high = 1 if phqtot_inp >= 10 & phqtot_inp!=.
	replace phqtot_high = 0 if phqtot_inp <10
	label variable phqtot_high "Screened positive for depression (phq8>=10)"
	 
	*liklihood of depression given PHQ-8 responses
	gen phq_prob = 0 if phqtot_inp==0
	replace phq_prob = .002 if phqtot_inp>=1 & phqtot_inp<=4
	replace phq_prob = .129 if phqtot_inp>=5 & phqtot_inp<=9
	replace phq_prob = .58 if phqtot_inp>=10 & phqtot_inp<=14
	replace phq_prob = .915 if phqtot_inp>=15 & phqtot_inp<=19
	replace phq_prob = .989 if phqtot_inp>=20 & phqtot_inp<=27
	label var phq_prob "depression diagnosis probability"
	
	*Diagnosis, any of diabetes, hypertension, high cholesterol, heart attack, or congestive heart failure
	gen any_dx_pre_lottery = (dia_dx_pre_lottery==1|hbp_dx_pre_lottery==1|chl_dx_pre_lottery==1|ami_dx_pre_lottery==1|chf_dx_pre_lottery==1)
	label variable any_dx_pre_lottery "Diagnosed with any of dia, hbp, chl, ami, or chf pre randomization"

	
* Quality

gen med_qual_bin_inp =.
	replace med_qual_bin_inp = 0 if inlist(med_qual_inp, 1, 2)
	replace med_qual_bin_inp = 1 if inlist(med_qual_inp , 3, 4, 5)
	
	label variable med_qual_bin_inp "RECODE of med_qual"
	label define med_qual_bin_inp_l 0 "fair/poor" 1 "good/vgood/excellent"
	label values med_qual_bin_inp med_qual_bin_inp_l
	
* Health behavior

gen smk_curr_bin_inp =.
	
	replace smk_curr_bin_inp = 0 if smk_curr_inp==2
	replace smk_curr_bin_inp = 1 if inlist(smk_curr_inp, 0, 1)
	
	label variable smk_curr_bin_inp "RECODE of smk_curr"
	label define smk_curr_bin_inp_l 0 "not at all" 1 "everyday/some days"
	label values smk_curr_bin_inp smk_curr_bin_inp_l
	 
	 
* Happiness
recode happiness_inp (0/1 = 1 "very/pretty happy") (2 = 0 "not too happy"), gen(poshappiness_bin_inp)

gen mam50_chk_inp = mam_chk_inp
replace mam50_chk_inp =. if gender_inp ==1 & age_inp <50
label variable mam50_chk_inp "Had a mammogram in the last 12 months (female, 50+)"

*Truncate the total spending variable

sum tot_med_spend_other_inp, det
local cutoff = 2*r(p99)
gen tr_tot_med_spend_other_inp=.
replace tr_tot_med_spend_other_inp=tot_med_spend_other_inp if tot_med_spend_other_inp<=`cutoff'
label var tr_tot_med_spend_other_inp "Total medical spending, including other, truncated at 2*99th percentile" 

*Truncate usage variables

sum doc_num_incl_probe_inp, det
local cutoff = 2*r(p99)
gen doc_num_mod_inp=.
replace doc_num_mod_inp=doc_num_incl_probe_inp if doc_num_incl_probe<=`cutoff'
label var doc_num_mod_inp "Num. of doctor's visits, truncated at 2*99th percentile" 

sum ed_num_incl_probe_inp, det
local cutoff = 2*r(p99)
gen ed_num_mod_inp=.
replace ed_num_mod_inp=ed_num_incl_probe_inp if ed_num_incl_probe<=`cutoff'
label var ed_num_mod_inp "Num. of ED visits, truncated at 2*99th percentile" 

sum surg_num_incl_probe_inp, det
local cutoff = 2*r(p99)
gen surg_num_mod_inp=.
replace surg_num_mod_inp=surg_num_incl_probe_inp if surg_num_incl_probe<=`cutoff'
label var surg_num_mod_inp "Num. of surgeries, truncated at 2*99th percentile" 

sum hosp_num_incl_probe_inp, det
local cutoff = 2*r(p99)
gen hosp_num_mod_inp_2=.
replace hosp_num_mod_inp=hosp_num_incl_probe_inp if hosp_num_incl_probe<=`cutoff'
label var hosp_num_mod_inp "Num. of hospitalizations, truncated at 2*99th percentile" 


* Response time
    gen response_time = dt_completed_inp - dt_release_inp
	assert response_time>=0
	label variable response_time "Response time"
	
* Interview components (keep has_* valid_meds_inp)
	gen has_anthro_inp=.
	replace has_anthro_inp=1 if has_hght_wght_inp*has_bp_inp*has_waist_inp==1
	replace has_anthro_inp=0 if has_hght_wght_inp*has_bp_inp*has_waist_inp==0
	label var has_anthro_inp "Has height, weight, bp, and waist measures"
	
* give stuff shorter names so the table-making files will all work
	gen any_oop_inp = any_oop_spending
	label var any_oop_inp "Any out of pocket spending"
	gen tr_tot_spend_inp = tr_tot_med_spend_other_inp
	label var tr_tot_spend_inp "Total medical spending, truncated at 2*99"
	gen tot_spend_inp=tot_med_spend_other_inp
	label var tot_spend_inp "Total medical spending, including other"

	
**************************
* Regression fixed effects

gen constant = 1

xi, prefix(nnn) i.numhh_list
desc nnn*

xi, prefix(lll) i.draw_lottery
desc lll*

gen sampbase2_inp=1
label var sampbase2_inp "Eligible for inp sample (incl. all hhold members)"

saveold Data/data_for_analysis.dta, replace 


