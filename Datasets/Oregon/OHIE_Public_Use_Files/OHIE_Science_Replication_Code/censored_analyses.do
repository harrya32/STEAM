
cap log close
log using censored_results.log, text replace

clear
set more off

use ../../Data/ED/oregonhie_ed_vars.dta

merge 1:1 person_id using ../../Data/OYP/oregonhie_descriptive_vars.dta, keepusing(treatment household_id numhh_list)
drop if _merge==2
assert _N==24646
drop _merge

merge 1:1 person_id using ../../Data/OYP/oregonhie_stateprograms_vars.dta, keepusing(ohp_all_ever_firstn_30sep2009 ohp_all_ever_inperson)
drop if _merge==2
assert _N==24646
drop _merge

merge 1:1 person_id using Data/oregonhie_survey12m_vars.dta, keepusing(returned_12m er_any_12m er_num_mod_12m weight_12m wave_survey12m)
drop if _merge==2
assert _N==24646
drop _merge


*Survey indicator/weights
gen mail_survey= (returned_12m==1)
label var mail_survey "Mail survey respondent"


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

*need household size fixed effects
xi, prefix(nnn) i.numhh_list
desc nnn*


*Inperson data
	merge 1:1 person_id using Data/oregonhie_inperson_vars.dta, keepusing(weight_total_inp)
	
	drop if _merge==2
	assert _N==24646
	drop _merge

	rename weight_total_inp weight_inp
	label var weight_inp "Inperson weights"
	

*Table 2

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
	label var twoplus "Individual had two or more ED visits in the preperiod"
	gen twoplus_out=0
	replace twoplus_out=1 if num_out_pre_cens_ed>=2 & !missing(num_out_pre_cens_ed)
	label var twoplus_out "Individual had two or more outpatient ED visits in the preperiod"
	gen fiveplus=0
	replace fiveplus=1 if num_visit_pre_cens_ed>=5 & !missing(num_visit_pre_cens_ed)
	label var fiveplus "Individual had five or more ED visits in the preperiod" 

matrix define censored_table2=J(12,4,.)

local row=1
foreach subpop in all zero one twoplus fiveplus twoplus_out {

	reg num_visit_cens_ed if treatment==0 & `subpop'==1
	local mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
	local sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
	matrix censored_table2[`row',1]= `mean'
	matrix censored_table2[`row'+1,1]= `sd'
	
	reg num_visit_cens_ed treatment num_visit_pre_cens_ed nnn* if `subpop'==1, cluster(household_id)
	local itt_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
	local itt_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
	
	matrix censored_table2[`row',2]= `itt_beta'
	matrix censored_table2[`row'+1,2]= `itt_se'

	ivregress 2sls num_visit_cens_ed (ohp_all_ever_firstn_30sep2009=treatment) num_visit_pre_cens_ed nnn* if `subpop'==1, cluster(household_id)

	local tot_beta = round(_b[ohp_all_ever_firstn_30sep2009],10^(min(-3, int(log10(abs(_b[ohp_all_ever_`source'])))-2)))
	local tot_se = round(_se[ohp_all_ever_firstn_30sep2009], 10^(min(-3, int(log10(abs(_se[ohp_all_ever_`source'])))-2)))
	matrix postest=r(table)
	local tot_p = round(postest[4,1],10^(min(-3, int(log10(abs(postest[4,1])))-2)))
	
	matrix censored_table2[`row',3]= `tot_beta'
	matrix censored_table2[`row'+1,3]= `tot_se'
	matrix censored_table2[`row',4]= `tot_p'
	
	local row=`row'+2

	}
	matrix rownames censored_table2= "all" "" "zero" "" "one" "" "twoplus" "" "fiveplus" "" "twoplus_out" "" 
	matrix list censored_table2

*Tables 3, S11, S13

matrix define censored=J(36,4,.)
local row=1
foreach var in num_visit num_hosp num_out num_off num_on num_acsc num_chron num_inj num_skin num_abdo num_back num_heart num_head num_depres num_psysub num_hiun num_loun {

	di "`var'_ed"
	
	di "reg `var'_cens_ed if treatment==0"
	reg `var'_cens_ed if treatment==0
	local mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
	local sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
	matrix censored[`row',1]= `mean'
	matrix censored[`row'+1,1]= `sd'
	
	di "reg `var'_cens_ed treatment `var'_pre_cens_ed nnn*, cluster(household_id)"
	reg `var'_cens_ed treatment `var'_pre_cens_ed nnn*, cluster(household_id)
	
	local itt_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
	local itt_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
	
	matrix censored[`row',2]= `itt_beta'
	matrix censored[`row'+1,2]= `itt_se'

	di "ivregress 2sls `var'_cens_ed (ohp_all_ever_firstn_30sep2009=treatment) `var'_pre_cens_ed nnn*, cluster(household_id)"
	ivregress 2sls `var'_cens_ed (ohp_all_ever_firstn_30sep2009=treatment) `var'_pre_cens_ed nnn*, cluster(household_id)

	local tot_beta = round(_b[ohp_all_ever_firstn_30sep2009],10^(min(-3, int(log10(abs(_b[ohp_all_ever_`source'])))-2)))
	local tot_se = round(_se[ohp_all_ever_firstn_30sep2009], 10^(min(-3, int(log10(abs(_se[ohp_all_ever_`source'])))-2)))
	matrix postest=r(table)
	local tot_p = round(postest[4,1],10^(min(-3, int(log10(abs(postest[4,1])))-2)))
	
	matrix censored[`row',3]= `tot_beta'
	matrix censored[`row'+1,3]= `tot_se'
	matrix censored[`row',4]= `tot_p'
	
	local row=`row'+2
	
	
				}
	matrix rownames censored="num_visit" " "  "num_hosp" " " "num_out" " " "num_off" " " "num_on" " " "num_acsc" " " "num_chron" " " ///
							"num_inj" "" "num_skin" "" "num_abdo" "" "num_back" "" "num_heart" "" "num_head" "" ///
							"num_depres" "" "num_psysub" "" "num_hiun" "" "num_loun" "" 

	matrix list censored 
	
*Table 5

matrix define censored_table5=J(4,4,.)
local row=1
local num_mail_match_weight "weight_mail"
local num_mail_match_controls "draw_survey_12m ddd*"
local num_mail_match_restrict "mail_survey==1"
local num_mail_match_insure "ohp_all_ever_firstn_30sep2009"

local num_inp_match_weight "weight_inp"
local num_inp_match_restrict "1==1"
local num_inp_match_insure "ohp_all_ever_inperson"

foreach var in num_mail_match num_inp_match {

di "`var'_ed"
	
	di "reg `var'_cens_ed if treatment==0 & ``var'_restrict' [pw=``var'_weight']"
	reg `var'_cens_ed if treatment==0 & ``var'_restrict' [pw=``var'_weight']
	local mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
	local sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
	matrix censored_table5[`row',1]= `mean'
	matrix censored_table5[`row'+1,1]= `sd'
	
	di "reg `var'_cens_ed treatment nnn* ``var'_controls' if ``var'_restrict' [pw=``var'_weight'], cluster(household_id)"
	reg `var'_cens_ed treatment nnn* ``var'_controls' if ``var'_restrict' [pw=``var'_weight'], cluster(household_id)
	
	local itt_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
	local itt_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
	
	matrix censored_table5[`row',2]= `itt_beta'
	matrix censored_table5[`row'+1,2]= `itt_se'

	di "ivregress 2sls `var'_cens_ed (``var'_insure'=treatment) nnn* ``var'_controls' if ``var'_restrict' [pw=``var'_weight'], cluster(household_id)"
	ivregress 2sls `var'_cens_ed (``var'_insure'=treatment) nnn* ``var'_controls' if ``var'_restrict' [pw=``var'_weight'], cluster(household_id)

	local tot_beta = round(_b[``var'_insure'],10^(min(-3, int(log10(abs(_b[ohp_all_ever_`source'])))-2)))
	local tot_se = round(_se[``var'_insure'], 10^(min(-3, int(log10(abs(_se[ohp_all_ever_`source'])))-2)))
	matrix postest=r(table)
	local tot_p = round(postest[4,1],10^(min(-3, int(log10(abs(postest[4,1])))-2)))
	
	matrix censored_table5[`row',3]= `tot_beta'
	matrix censored_table5[`row'+1,3]= `tot_se'
	matrix censored_table5[`row',4]= `tot_p'
	
	local row=`row'+2
	
	
				}
	matrix rownames censored_table5="num_mail_match" " "  "num_inp_match" " " 
	
		matrix list censored_table5
	
	
	matrix list censored_table2
	matrix list censored
	matrix list censored_table5
	

	log close
	
	
	clear
	use ../../../Lottery2008/AnalyticData/emergency_department_data.dta
	
	tab tr_survey_tot_mail, mi
	tab tr_survey_tot_inp, mi
	
	clear
	use ../../Data/ED/oregonhie_ed_vars.dta

	tab num_mail_match_cens_ed, mi
	tab num_inp_match_cens_ed, mi

	
	
	
	
	
	
