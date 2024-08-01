
*********************************************************
*Admissions-level file, cleaned and with zip code restriction
*********************************************************

***********************
*Individual-level file
***********************

clear
use Data/oregonhie_ed_vars.dta


*Rename variables (back to internal OHS names) so code will run

foreach date in 09mar2008 {

	rename any_visit_pre_ed ed_visit_`date'
	rename any_hosp_pre_ed ed_hosp_`date' 
	rename any_out_pre_ed ed_out_`date' 
	rename any_off_pre_ed ed_off_`date' 
	rename any_on_pre_ed ed_on_`date' 
	
	rename num_ne_pre_ed tr_ed_ne_`date' 
	rename num_epct_pre_ed tr_ed_epct_`date' 
	rename num_edcnpa_pre_ed tr_ed_edcnpa_`date' 
	rename num_edcnnp_pre_ed tr_ed_edcnnp_`date' 
	rename num_unclas_pre_ed tr_ed_unclas_`date' 
	
	rename any_chron_pre_ed ed_chron_`date' 
	rename any_acsc_pre_ed ed_acsc_`date' 
	
	rename any_inj_pre_ed ed_inj_`date' 
	rename any_skin_pre_ed ed_skin_`date' 
	rename any_abdo_pre_ed ed_abdo_`date' 
	rename any_back_pre_ed ed_back_`date' 
	rename any_heart_pre_ed ed_heart_`date' 
	rename any_head_pre_ed ed_head_`date' 
	rename any_depres_pre_ed ed_depres_`date' 
	rename any_psysub_pre_ed ed_psysub_`date' 
	
	rename charg_tot_pre_ed tr_charg_`date'
	rename ed_charg_tot_pre_ed tr_ed_charg_`date' 
	
	rename any_hiun_pre_ed ed_hiun_`date' 
	rename any_loun_pre_ed ed_loun_`date' 
	
		}

foreach date in 30sep2009 {

	rename any_visit_ed ed_visit_`date' 
	rename any_hosp_ed ed_hosp_`date' 
	rename any_out_ed ed_out_`date' 
	rename any_off_ed ed_off_`date'
	rename any_on_ed ed_on_`date' 
	
	rename num_ne_ed tr_ed_ne_`date' 
	rename num_epct_ed tr_ed_epct_`date' 
	rename num_edcnpa_ed tr_ed_edcnpa_`date' 
	rename num_edcnnp_ed tr_ed_edcnnp_`date' 
	rename num_unclas_ed tr_ed_unclas_`date' 
	
	rename any_chron_ed ed_chron_`date' 
	rename any_acsc_ed ed_acsc_`date' 
	
	rename any_inj_ed ed_inj_`date' 
	rename any_skin_ed ed_skin_`date' 
	rename any_abdo_ed ed_abdo_`date' 
	rename any_back_ed ed_back_`date'  
	rename any_heart_ed ed_heart_`date' 
	rename any_head_ed ed_head_`date' 
	rename any_depres_ed ed_depres_`date' 
	rename any_psysub_ed ed_psysub_`date' 
		
	rename charg_tot_ed tr_charg_`date' 
	rename ed_charg_tot_ed tr_ed_charg_`date'
	
	rename any_hiun_ed ed_hiun_`date' 
	rename any_loun_ed ed_loun_`date' 
	
		}
		
rename any_mail_match_ed survey_any_mail 
rename num_mail_match_ed tr_survey_tot_mail
rename any_inp_match_ed survey_any_inp 
rename num_inp_match_ed tr_survey_tot_inp

/*Base Data */

merge 1:1 person_id using Data/oregonhie_descriptive_vars.dta, keepusing(household_id numhh_list female_list birthyear_list ///
						english_list birthyear_list self_list first_day_list have_phone_list pobox_list ///
						dt_notify_lottery dt_app_decision ///
						female_list first_day_list draw_lottery treatment) 

drop _merge

/* Generate Additional Variables and Sample Cuts */

*need household size fixed effects
xi, prefix(nnn) i.numhh_list
desc nnn*

*need lottery draw fixed effects
xi, prefix(lll) i.draw_lottery
	desc lll*

merge 1:1 person_id using Data/oregonhie_stateprograms_vars.dta, ///
				keepusing(ohp_all_ever_firstn_30sep2009 ohp_all_end_30sep2009 ohp_std_ever_firstn_30sep2009 ///
				ohp_all_mo_firstn_30sep2009 ohp_all_ever_inperson)
drop if _merge==2
assert _N==74922
drop _merge

gen ohp_all_ever_mail=ohp_all_ever_firstn_30sep2009 
label var ohp_all_ever_mail "Ever on Medicaid from first notification date (10mar2008) to 30 September 2009"
rename ohp_all_ever_firstn_30sep2009 ohp_all_ever_30sep2009
rename ohp_all_ever_inperson ohp_all_ever_inp
rename ohp_std_ever_firstn_30sep2009 ohp_std_ever_30sep2009
rename ohp_all_mo_firstn_30sep2009 ohp_all_mo_30sep2009 

replace sample_ed=0 if missing(sample_ed)

assert _N==74922

gen constant=1

**********
*Subpopulations for Heterogeneity Analysis			
**********

/*Inperson Data (for additional subsample variables) */ 
merge 1:1 person_id using Data/oregonhie_inperson_vars.dta, ///
			keepusing(race_white_inp race_nwother_inp smk_curr_inp dia_dx_pre_lottery_inp ///
			hbp_dx_pre_lottery_inp chl_dx_pre_lottery_inp ami_dx_pre_lottery_inp chf_dx_pre_lottery_inp ast_dx_pre_lottery_inp ///
			kid_dx_pre_lottery_inp emp_dx_pre_lottery_inp cancer_dx_pre_lottery_inp dep_dx_pre_lottery_inp edu_inp)
drop if _merge==2
drop _merge
assert _N==74922

*rename some variables to make the code run
rename dia_dx_pre_lottery_inp dia_dx_pre_lottery
rename hbp_dx_pre_lottery_inp hbp_dx_pre_lottery
rename chl_dx_pre_lottery_inp chl_dx_pre_lottery
rename ami_dx_pre_lottery_inp ami_dx_pre_lottery
rename chf_dx_pre_lottery_inp chf_dx_pre_lottery
rename ast_dx_pre_lottery_inp ast_dx_pre_lottery
rename kid_dx_pre_lottery_inp kid_dx_pre_lottery
rename emp_dx_pre_lottery_inp emp_dx_pre_lottery
rename cancer_dx_pre_lottery_inp cancer_dx_pre_lottery
rename dep_dx_pre_lottery_inp dep_dx_pre_lottery

*Inperson survey cuts
gen morehs=0 if !missing(edu_inp)
replace morehs=1 if edu_inp==3 | edu_inp==4
label var morehs "More than HS/GED"

gen lesshs=0 if !missing(edu_inp)
replace lesshs=1 if morehs==0
label var lesshs "HS/GED or less"

gen smoke=0 if !missing(smk_curr_inp)
replace smoke=1 if smk_curr_inp==1 | smk_curr_inp==0
label var smoke "Smoker (in-person interview response)"

gen nosmoke=0 if !missing(smk_curr_inp)
replace nosmoke=1 if smk_curr_inp==2
label var nosmoke "Non smoker (in-person interview response)"

gen white= (race_white_inp==1) if !missing(race_white_inp)
label var white "Self-reported race as white (in-person survey)"
gen nonwhite= (race_nwother_inp==1) if !missing(race_nwother_inp)
label var nonwhite "Self_reported race as non-white (in-person survey)"

*Any Diagnosis pre-lottery:
gen dx_prelot= (dia_dx_pre_lottery==1|hbp_dx_pre_lottery==1|chl_dx_pre_lottery==1| ///
					ami_dx_pre_lottery==1|chf_dx_pre_lottery==1|ast_dx_pre_lottery==1| ///
					kid_dx_pre_lottery==1|emp_dx_pre_lottery==1|cancer_dx_pre_lottery==1| ///
					dep_dx_pre_lottery==1) if !missing(dia_dx_pre_lottery)
label var dx_prelot "Any diagnosis pre-lottery (inp)"

assert dx_prelot==1 if dia_dx_pre_lottery==1
assert dx_prelot==1 if emp_dx_pre_lottery==1
			
drop dia_dx_pre_lottery hbp_dx_pre_lottery chl_dx_pre_lottery ami_dx_pre_lottery ///
	ast_dx_pre_lottery chf_dx_pre_lottery kid_dx_pre_lottery emp_dx_pre_lottery ///
	cancer_dx_pre_lottery dep_dx_pre_lottery 

*List data subpopulations
gen old = (birthyear_list >= 1945 & birthyear_list <= 1958)
label var old "Age 50-64 in 2008"

gen young = (birthyear_list >= 1959 & birthyear_list <= 1989)
label var young "Age 19-49 in 2008"
assert old==1 if young==0 & sample_ed==1

gen male= (female_list==0)
label var male "Male (lottery list variable)"
gen female= (female_list==1)
label var female "Female (lottery list variable)"

gen first_day= (first_day_list==1)
gen notfirst_day=(first_day_list==0)
label var first_day "Signed up for lottery on the first day the list was open"
label var notfirst_day "Signed up for lottery, but not on the first day the list was open"

 ****Save balance data
	count
	save Data/balance_data.dta, replace
 
	keep if sample_ed==1
	assert _N==24646


*****
*Sample cuts on prelottery usage
*****
	gen all=1
	label var all "Full ed sample"
	gen zero=0
	replace zero=1 if num_visit_pre_cens_ed==0 
	label var zero "Individual had zero ED visits in the preperiod"
	gen one=0
	replace one=1 if num_visit_pre_cens_ed==1 
	label var one "Individual had one ED visit in the preperiod"
	gen twoplus=0
	replace twoplus=1 if num_visit_pre_cens_ed>=2 & !missing(num_visit_pre_cens_ed) 
	replace twoplus=1 if missing(num_visit_pre_cens_ed) // if preperiod visits are missing, variable was truncated (therefore >2)
	label var twoplus "Individual had two or more ED visits in the preperiod"
	gen twoplus_out=0
	replace twoplus_out=1 if num_out_pre_cens_ed>=2 & !missing(num_out_pre_cens_ed)
	replace twoplus_out=1 if missing(num_out_pre_cens_ed) // if preperiod outpatient visits are missing, variable was truncated (therefore >2)
	label var twoplus_out "Individual had two or more outpatient ED visits in the preperiod"
	gen fiveplus=0
	replace fiveplus=1 if num_visit_pre_cens_ed>=5 & !missing(num_visit_pre_cens_ed)
	replace fiveplus=1 if missing(num_visit_pre_cens_ed) // if preperiod visits are missing, variable was truncated (therefore >5)
	label var fiveplus "Individual had five or more ED visits in the preperiod" 

*****
*Missing pre-period data indicator (and recode as the mean)
*****
/*Most analyses control for pre-period outcomes (i.e. number of ED visits 
from 1 January 2007 to 30 September 2009). The missing pre-period data 
indicator is included as a regression covariate in order to prevent individuals 
from being dropped from the analysis in the event that they are missing data in the preperiod.*/

local all ed_visit ed_hosp ed_out ed_off ed_on ed_inj ed_skin ed_heart ed_back ed_head ed_abdo ed_depres ed_psysub ed_chron ed_acsc tr_ed_ne tr_ed_epct tr_ed_edcnpa tr_ed_edcnnp ed_hiun ed_loun tr_ed_unclas tr_charg tr_ed_charg
	foreach date in 09mar2008 {
		foreach var in `all' {
			gen `var'_`date'_m= (missing(`var'_`date'))
			label var `var'_`date'_m "Indicates Missing `var'_`date' (regression covariate)"
			sum `var'_`date'
			replace `var'_`date'=r(mean) if missing(`var'_`date')  

			}
		}
		
**************************************
*Add Survey data to compare responses with administrative data
***************************************

*Mail survey data
merge 1:1 person_id using Data/oregonhie_survey12m_vars.dta, keepusing(returned_12m er_any_12m er_num_mod_12m weight_12m wave_survey12m)
drop if _merge==2
tab returned_12m if _merge==3, mi

	*Survey indicator/weights
	gen mail_survey= (returned_12m==1)
	label var mail_survey "Mail survey respondent"

	assert _merge==3 if mail_survey==1
	drop _merge

	rename weight_12m weight_mail
	label var weight_mail "12m Mail Survey weights"
	
	/* Fixed Effects for regressions */
	*survey draw
	gen draw_survey_12m=wave_survey12m
	label var draw_survey_12m "Survey wave for 12m survey- adjusted for simultaneous mailing of waves 7 and 8"
	replace draw_survey_12m=7 if draw_survey_12m==8
	
	xi, prefix(ddd) i.draw_survey_12m*i.numhh_list
	desc ddd* 
	drop ddddraXnum_4_3 ddddraXnum_5_3 ddddraXnum_6_3 ddddraXnum_7_3 

	*ER use variables
	rename er_any_12m er_any_mail
	rename er_num_mod_12m er_num_mail
	label var er_any_mail "Any ED visit in the 6m preceeding survey response (12m survey variable)"
	label var er_num_mail "Number of ED visits in the 6m preceeding survey response (12m survey variable)"

	replace er_any_mail=0 if missing(er_any_mail) & mail_survey==1
	replace er_num_mail=0 if missing(er_num_mail) & mail_survey==1

assert _N==24646

*Inperson data
	merge 1:1 person_id using Data/oregonhie_inperson_vars.dta, keepusing(dt_completed_inp ed_any_incl_probe_inp ed_num_incl_probe_inp weight_total_inp)
	
	drop if _merge==2
	count if !missing(dt_completed_inp) & _merge==3
	
	*Truncate number of ED visits at 2*99th%ile
	sum ed_num_incl_probe_inp, det
	local cutoff = 2*r(p99)
	gen ed_num_mod_inp=.
	replace ed_num_mod_inp=ed_num_incl_probe_inp if ed_num_incl_probe<=`cutoff'
	label var ed_num_mod_inp "Num. of ED visits, truncated at 2*99th percentile" 

	*survey indicator/weights
	gen inp_survey= (!missing(dt_completed_inp))
	label var inp_survey "Inperson survey respondent"
	assert _merge==3 if inp_survey==1
	drop _merge

	label var dt_completed_inp "Date in-person interview completed" 
	rename weight_total_inp weight_inp
	label var weight_inp "Inperson weights"

	*ER use variables
	rename ed_any_incl_probe_inp er_any_inp
	rename ed_num_mod_inp er_num_inp
	label var er_any_inp "Any ED visit in 12m preceeding survey response (inperson survey variable)"
	label var er_num_inp "Number of ED visits in 12m preceeding survey response (inperson survey variable)"

	replace er_any_inp=0 if missing(er_any_inp) & inp_survey==1
	replace er_num_inp=0 if missing(er_num_inp) & inp_survey==1

assert _N==24646


assert !missing(weight_inp) if inp_survey==1
assert inp_survey==0 if missing(weight_inp)
assert !missing(weight_mail) if mail_survey==1
assert missing(dt_completed_inp) if inp_survey==0
assert !missing(dt_completed_inp) if inp_survey==1


	save Data/data_for_analysis.dta, replace

