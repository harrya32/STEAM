/**************************************
*This code replicates tables from Baicker et al (NEJM 2013).
It runs on Stata versions 13 and earlier. 

Tables replicated are:
	Table 1
	Table 2
	Table 3
	Table 4
	Table 5 - To replicate the spending estimate, the switch "local STE=1" 
				inside the Table 5 switch must be equal to 1 
				(this is its default position). 
			
	Table S1
	Table S2
	Table S3
	Table S4
	Table S5
	Table S6
	Table S7 - "median household income in zip code" is omitted. 
				Note that due to this the pooled F-statistic does not replicate Baicker et al (2013).
	Table S8
	Table S9
	Table S10
	Table S11
	Table S14
	Table s15
	Table S16
	Table S17
	Table S18
	Table S19


The table-generating code is divided among several .do files, all of which are 
called by this master file.

*To run this file:*

Save this .do file to the folder where you wish to do your work.

Save the following .do file to a subfolder named "Programs"
	prepare_data.do
	tables_analysis_inp.do
	tables_balance.do
	tables_balance_NEJM.do
	tables_compilers.do
	tables_firststage_pval.do
	tables_logit.do
	tables_observational_inp.dp
	
Save the following datasets to a subfolder named "Data"

	oregonhie_descriptive_vars.dta
	oregonhie_stateprograms_vars.dta
	oregonhie_survey12m_vars.dta
	oregonhie_inperson_nejm_vars.dta
	
Edit the local "switches" below to choose which tables to run. For example, 
leaving the code as "local T1_alt_switch=1" will replicate Table 1. Changing the
code to "local T1_alt_switch=0" will cause Table 1 to not be replicated. 

The full log file outputs to oregon_hie_nejm_replication.log. In addition, a 
log file with just the generated tables outputs to paper_tables.log. 

**************************************/

set more off
capture log close
log using oregon_hie_nejm_replication.log, text replace

clear
clear matrix
matrix drop _all

set mem 800m
set linesize 150
set matsize 800

set seed 493767

global ALT=1  // Displays confidence intervals instead of standard errors (Leave this set to 1)

********************************
* SWITCHES
********************************

* Paper Tables
local T1_alt_switch=1 // Balance of population characteristics
local T2_alt_switch=1 // Tables 2 and S1: Physical Health Measures
local T3_alt_switch=1 // Tables 3 and S2: Health Related Quality of Life
local T4_alt_switch=1 // Tables 4 and S3: Finances
local T5_alt_switch=1 // Tables 5 and S4: Utilization, Access and Quality, Prevention, and Behavior

* Appendix Tables
local TA5_alt_switch=1 // Table S5: Characteristics of compliers   
local TA6_alt_switch=1 // Table S6: Summary of Weights
local TA7_alt_switch=1 // Table S7: Treatment and control balance
local TA8_alt_switch=1 // Table S8: Pre-randomization DX balance
local TA9_alt_switch=1 // Table S9: First Stage	
local TA10_alt_switch=1 // Table S10: Summary of analytic variables
local TA11_alt_switch=1 // Table S11: Distribution of analytic variables  
local TA14_alt_switch=1 // Table S14: Mean Val/Abs change in clinical measures (over 50, "high risk", and specific diagnoses)
local TA15a_alt_switch=1 // Table S15a Logit: Physical Health Measures
local TA15b_alt_switch=1 // Table S15b Logit: Health Related Quality of Life
local TA15c_alt_switch=1 // Table S15c Logit: Finances
local TA15d_alt_switch=1 // Table S15d Logit: Utilization, Access and Quality, Prevention, and Behavior
local TA16a_alt_switch=1 // Table S16: Alternative specifications: Physical Health Measures // took out median household income in zip code from the local "control_all" in all 4 of these panels
local TA16b_alt_switch=1 // Table S16: Alternative specifications: Health Related Quality fo Life
local TA16c_alt_switch=1 // Table S16: Alternative specifications: Finances
local TA16d_alt_switch=1 // Table S16: Alternative specifications: Utilization, access and quality, Prevention, and behavior
local TA17_alt_switch=1 // Table S17: Other clinical outcomes
local TA18a_alt_switch=1 // Table S18: Panels A and E (Physical Health Measures, Health Related Quality fo Life)
local TA18b_alt_switch=1 // Table S18: Panels B and F (Finances)
local TA18c_alt_switch=1 // Table S18: Panels C and G (Utilization) 
local TA18d_alt_switch=1 // Table S18: Panels D and H (Access and quality)
local TA18e_alt_switch=1 // Table S18: Panels C and H 
local TA19_alt_switch=1 // OLS on physical health measures

********************************
* GLOBALS AND LOCALS
********************************

**Regression settings
global iterations = 3
local mult_inf_switch = "no" // "" to turn on, "no" to turn off

global weight "weight_total_inp" 
global adjustments "nnn*"
global sample "sample_inp_resp"


global chars_list "gender_inp age_19_34_inp age_35_49_inp age_50_64_inp race_white_inp race_black_inp race_nwother_inp hispanic_inp itvw_english_inp"
global prevention_list "chl_chk_inp fobt_chk_inp col_chk_inp did_flu_inp pap_chk_inp mam50_chk_inp psa_chk_inp"
*global desc_var "female_list age_19_34_inp age_35_49_inp age_50_64_inp english_list race_white_inp race_black_inp hispanic_inp edu_inp_1 edu_inp_2 edu_inp_3 edu_inp_4 employ_cat_inp_1 employ_cat_inp_2 employ_cat_inp_3 employ_cat_inp_4  hh_income_val_inp fpl_cat_inp_1 fpl_cat_inp_2 fpl_cat_inp_3 fpl_cat_inp_4 fpl_cat_inp_5 dia_dx_inp ast_dx_inp hbp_dx_inp chl_dx_inp emp_dx_inp dep_dx_inp ins_any_inp ins_ohp_inp ins_private_inp ins_other_inp"
global baseline_list "age_inp gender_inp race_black_inp race_nwother_inp hispanic_inp english_list self_list first_day_list have_phone_list pobox_list" // took out med hh income in zip
global fielding "dt_completed_inp response_time interview_season_4 interview_season_1 interview_season_3 interview_weekend int_loc_cat_inp_3 int_loc_cat_inp_1 int_loc_cat_inp_4 language_capi_inp interpreter_inp"

global interviewer_fix "itvr_1"
	foreach i of numlist 2/49 {
	global interviewer_fix "$interviewer_fix itvr_`i'"
	}
dis "$interviewer_fix"

global scale_fix "scale_1"
	foreach i of numlist 2/43 {
    global scale_fix "$scale_fix scale_`i'"
	}
dis "$scale_fix"

global stadio_fix "stadio_1"
	foreach i of numlist 2/43 {
    global stadio_fix "$stadio_fix stadio_`i'"
	}
dis "$stadio_fix"

global omron_fix "omron_1"
	foreach i of numlist 2/41 {
    global omron_fix "$omron_fix omron_`i'"
	}
dis "$omron_fix"


global diagnoses_pre_lottery "ast_dx_pre_lottery dia_dx_pre_lottery hbp_dx_pre_lottery chl_dx_pre_lottery ami_dx_pre_lottery chf_dx_pre_lottery emp_dx_pre_lottery kid_dx_pre_lottery cancer_dx_pre_lottery dep_dx_pre_lottery"
global oop_spending "any_oop_inp tr_tot_spend_inp catastrophic_exp_inp"
global debt_details "owe_inp borrow_inp"

**************************
* PREPARE DATA 
**************************

** This piece - Programs/prepare_data.do - does the following
***** 1. merges the various analytic data sets
***** 2. creates any other variables needs for the analysis
***** 3. saves a limited local dataset called "data_for_analysis.dta"

include Programs/prepare_data.do

***************************
* ANALYSIS
****************************

** The analysis is coded by table
** Some tables share a form, when these happens, there is generic code to run the table in a separate file

* ***************************
* Table 1: Balance of Population Characteristics
* ***************************

if `T1_alt_switch' ==1 {
	
****** Balance of treament and controls in inperson survey responders

	* Characteristics of the study population
	local balance_table T1
	local balance_list "$chars_list"
	local balance_subset "$chars_list"
	local balance_matchvar "$sample"
	local balance_condition "& $sample==1"
	local balance_controls "$adjustments"
	local balance_weight "$weight"
	local list_binary "$chars_list"
	
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Run table
	include Programs/tables_balance_NEJM.do
	
	} // End of T1 switch


* ***************************
* Tables 2 and S1: Physical Health Measures
* ***************************
if `T2_alt_switch' == 1 {
        dis "test2"

	foreach var in  bp_sar_inp bp_dar_inp  bp_hyper hbp_dx_post_lottery hbp_diure_med_inp {
	local `var'_controls "gender_inp age_decile_dum*"
        }
	
    * Panel A
        local table T2_A
	* Locals for running the control distribution code
	local list_survey "bp_sar_inp bp_dar_inp bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_inp chl_h hdl_inp hdl_low chl_dx_post_lottery antihyperlip_med a1c_inp a1c_dia dia_dx_post_lottery diabetes_med phqtot_high  dep_dx_post_lottery antidep_med"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_h hdl_low chl_dx_post_lottery antihyperlip_med a1c_dia dia_dx_post_lottery diabetes_med phqtot_high dep_dx_post_lottery antidep_med"
	local ste=0
	local inp_12m_spend=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
	local table T2_B
	* Locals for running the control distribution code
	local list_survey "cvd_risk_point"
	local condition_survey "& $sample==1 & age_inp>=30"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample age_inp"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "cvd_risk_point"
	local ste=0

	
	* Set up data
	clear
	use Data/data_for_analysis.dta
	

	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do

	* Panel C
	local table T2_C
	* Locals for running the control distribution code
	local list_survey "cvd_risk_point"
	local condition_survey "& $sample==1 & any_dx_pre_lottery==1 & age_inp>=30"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample any_dx_pre_lottery age_inp"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "cvd_risk_point"
	local ste=0

	
	* Set up data
	clear
	use Data/data_for_analysis.dta


	* Run table
	drop ohp_all_ever_survey 
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
	* Panel D
     local table T2_D
	* Locals for running the control distribution code
	local list_survey "cvd_risk_point"
	local condition_survey "& $sample==1 & age_inp>=50 & age_inp<=64"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample  age_inp"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "cvd_risk_point"
	local ste=0

	
	* Set up data
	clear
	use Data/data_for_analysis.dta

	* Run table
	drop ohp_all_ever_survey 
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
	foreach var in  bp_sar_inp bp_dar_inp  bp_hyper hbp_dx_post_lottery hbp_diure_med_inp {
	local `var'_controls ""
        }
} // End of T2 switch


* ***************************
* Tables 3 and S2: Health Related Quality of Life
* ***************************
if `T3_alt_switch' == 1 {
 
    * Panel A
        local table T3
	* Locals for running the control distribution code
	
	local list_survey "health_change_noworse mcs8_score pcs8_score pain_low_inp poshappiness_bin_inp"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample"
	local exclude_from_ste "poshappiness_bin_inp"
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "health_change_noworse pain_low_inp poshappiness_bin_inp"
	local ste=0
			
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do	
	
} // End of T3 switch

*********************************************
* Tables 4 and S3: Finances
*********************************************	
if `T4_alt_switch'==1 {

    * Panel A
    local table T4
	* Locals for running the control distribution code
	local list_survey "$oop_spending $debt_details"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample"
	local exclude_from_ste "any_oop_inp catastrophic_exp_inp"
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "any_oop_inp owe_inp borrow_inp catastrophic_exp_inp"
	local ste=0 

	
	* Set up data
	clear
	use Data/data_for_analysis.dta

	* Run table
	drop ohp_all_ever_survey 
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do

} // end of T4 switch

	
* *****************************************************************
* Tables 5 and S4: Utilization, Access and Quality, Prevention, and Behavior
* *****************************************************************
if `T5_alt_switch' ==1 {

        local table T5
	* Locals for running the control distribution code
	local list_survey "rx_num_mod_inp doc_num_mod_inp surg_num_mod_inp ed_num_mod_inp hosp_num_mod_inp $prevention_list usual_clinic_inp needmet_med_inp med_qual_bin_inp smk_curr_bin_inp obese"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 1
	local list_reverse "hdl_inp"
	local list_binary "usual_clinic_inp needmet_med_inp med_qual_bin_inp $prevention_list smk_curr_bin_inp obese"
	local ste=1
	local inp_12m_spend=0 // need this condition if spend_est is 1
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
} // End of T5 Switch


* **********************************************************************************************************************************************************************************************************************************

*******************************
* Appendix Tables
*******************************
* ***************************************
* Table S5: Compliers
* ***************************************
if `TA5_alt_switch'==1 {

	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Survey responders
	
	local table TA5
	local list "$chars_list"
	local insurance "ohp_all_ever_inperson"  
	local sample "$sample"
	local controls "$adjustments"
	local weight "weight_total_inp"

	include Programs/tables_compliers.do
	
} // end of TA15 switch


* ***************************************
* Table S6: Summary of Weights
* ***************************************
if `TA6_alt_switch'==1 {
clear
use Data/data_for_analysis.dta

gen samp_recruit=weight_total_inp>0 & weight_total_inp<.
gen samp_resp_ctrl=sample_inp_resp*(1-treatment)
gen samp_resp_treat=sample_inp_resp*treatment

assert samp_resp_ctrl==1-samp_resp_treat if sample_inp_resp==1
        
matrix define sum_weights=J(5, 10, .)
matrix rownames sum_weights = sampbase2_inp samp_recruit sample_inp_resp samp_resp_ctrl samp_resp_treat
matrix colnames sum_weights = mean sd min p5 p25 p50 p75 p95 max N
	local i=0
	
	foreach sample in sampbase2_inp samp_recruit sample_inp_resp samp_resp_ctrl samp_resp_treat {
	local i = `i'+1
	sum weight_total_inp if `sample'==1, detail
	
	matrix sum_weights[`i', 1]=round(r(mean), .001)
	matrix sum_weights[`i', 2]=round(r(sd), .001)
	matrix sum_weights[`i', 3]=round(r(min), .001)
	matrix sum_weights[`i', 4]=round(r(p5), .001)
	matrix sum_weights[`i', 5]=round(r(p25), .001)
	matrix sum_weights[`i', 6]=round(r(p50), .001)
	matrix sum_weights[`i', 7]=round(r(p75),.001)
	matrix sum_weights[`i', 8]=round(r(p95), .001)
	matrix sum_weights[`i', 9]=round(r(max), .001)
	matrix sum_weights[`i', 10]=round(r(N), .001)
	}
} // end of TA6 switch
	
* ***************************************
* Table S7: Treatment and control balance
* ***************************************
if `TA7_alt_switch' ==1 {
	
****** Balance of treament and controls in inperson survey responders

  * Part 1: Survey variables & fielding variables
	* Locals for running the balance table code
  
	local balance_table TA7_1
	local balance_list "$baseline_list $fielding"
	local balance_subset "$baseline_list $fielding"
	local balance_matchvar "$sample"
	local balance_condition "& $sample==1"
	local balance_controls "$adjustments"
	local balance_weight "$weight"
	local list_binary " gender_inp race_black_inp race_nwother_inp hispanic_inp  english_list self_list first_day_list have_phone_list pobox_list interview_season_4 interview_season_1 interview_season_3 interview_weekend int_loc_cat_inp_3 int_loc_cat_inp_1 int_loc_cat_inp_4 language_capi_inp interpreter_inp"

   * Set up data
    clear
 	use Data/data_for_analysis.dta
	compress
	
	* Run table
	include Programs/tables_balance.do
	
  * Part 1.5 : Interview component
   local balance_table TA7_1A
	local balance_list "has_anthro_inp has_dbs_inp has_all_dbs_inp valid_meds_inp"
	local balance_subset "has_anthro_inp has_dbs_inp has_all_dbs_inp valid_meds_inp"
	local balance_matchvar "$sample"
	local balance_condition "& $sample==1"
	local balance_controls "$adjustments"
	local balance_weight "$weight"
	local list_binary "has_anthro_inp has_dbs_inp has_all_dbs_inp valid_meds_inp"

   * Set up data
    clear
 	use Data/data_for_analysis.dta
	compress
	
	* Run table
	include Programs/tables_balance.do

  * Part 2: Interviewer fixed effect
    local balance_table TA7_2
	local balance_list "$interviewer_fix"
	local balance_subset "$interviewer_fix"
	local balance_matchvar "$sample"
	local balance_condition "& $sample==1 "
	local balance_controls "$adjustments"
	local balance_weight "$weight"
	local list_binary "$interviewer_fix"
	
   * Set up data
   clear
 	use Data/data_for_analysis.dta
	compress
	tab interviewer_inp
	
	* Run table
	include Programs/tables_balance.do
	
    * Part 3:Equipment fixed effect: scale
    local balance_table TA7_3
	local balance_list "$scale_fix"
	local balance_subset "$scale_fix"
	local balance_matchvar "$sample"
	local balance_condition "& $sample==1"
	local balance_controls "$adjustments"
	local balance_weight "$weight"
	local list_binary "$scale_fix"
	
   * Set up data
   clear
 	use Data/data_for_analysis.dta
	compress
	tab scale_id
	
	* Run table
	include Programs/tables_balance.do
	
	* Part 4: Equipment fixed effect: stadio
    local balance_table TA7_4
	local balance_list "$stadio_fix"
	local balance_subset "$stadio_fix"
	local balance_matchvar "$sample"
	local balance_condition "& $sample==1 "
	local balance_controls "$adjustments"
	local balance_weight "$weight"
	local list_binary "$stadio_fix"
	
   * Set up data
   clear
 	use Data/data_for_analysis.dta
	compress
	tab stadio_id
	
	* Run table
	include Programs/tables_balance.do
	
	* Part 5: Equiptment fixed effect: omron
    local balance_table TA7_5
	local balance_list "$omron_fix"
	local balance_subset "$omron_fix"
	local balance_matchvar "$sample"
	local balance_condition "& $sample==1"
	local balance_controls "$adjustments"
	local balance_weight "$weight"
	local list_binary "$omron_fix"
	
   * Set up data
   clear
 	use Data/data_for_analysis.dta
	compress
	tab omron_id
	
	* Run table
	include Programs/tables_balance.do	

} // end of TA2 switch





* ***************************************
* Table S8: Pre-randomization Dx balance
* ***************************************
if `TA8_alt_switch' ==1 {
	
****** Balance of treament and controls in inperson survey responders

  * Part 1: Balance Table
	* Locals for running the balance table code
	local balance_table TA8_1
	local balance_list "$diagnoses_pre_lottery any_dx_pre_lottery"
	local balance_subset "$diagnoses_pre_lottery"
	local balance_matchvar "$sample"
	local balance_condition "& $sample==1"
	local balance_controls "$adjustments"
	local balance_weight "$weight"
	local list_binary "$diagnoses_pre_lottery any_dx_pre_lottery"

   * Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Run table
	include Programs/tables_balance.do
	

  * Part 2: STE (add to the bottom)
	* Locals for running the balance table code
	local table TA8_2
	local list_survey "$diagnoses_pre_lottery"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	local list_binary "$diagnoses_pre_lottery"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local ste=1
	
	* Set up data
	clear
	use Data/data_for_analysis.dta

	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	

	} 	// end of Table A8 switch


	
* ***************************************
* Table S9: First Stage
* ***************************************
if `TA9_alt_switch'==1 {

	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Survey responders
	
	local table TA9_A
	local list  "ohp_all_ever_inperson ohp_std_ever_inperson ohp_all_mo_inperson ohp_all_end_inperson ins_any_inp ins_ohp_inp ins_private_inp"  
	local sample "$sample"
	local controls "$adjustments"
	local weight "weight_total_inp"
	local list_binary "ohp_all_ever_inperson ohp_std_ever_inperson ohp_all_end_inperson ins_any_inp ins_ohp_inp ins_private_inp"   
	
	include Programs/tables_firststage_pval.do
	
	* Non-missing sample size
	clear 
	use Data/data_for_analysis.dta
	
	matrix define TableTA9_N = J(`rows', 1, .)
	matrix colnames TableTA9_N = "N"
	
    local i=1
    foreach yvar of local list {
	count if `sample'==1 & `yvar'!=.
	local `i'_N = r(N)
	matrix TableTA9_N[`i',1] = ``i'_N'
	local i = `i'+1
	}
	
	* Just control means and treatment means
	foreach var in ohp_all_ever_inperson ohp_std_ever_inperson {
	noi dis "`var' control"
	noi sum `var' if $sample==1 & treatment==0 [aw=weight_total_inp]
	noi dis "`var' treatment"
	noi sum `var' if $sample==1 & treatment==1 [aw=weight_total_inp]
	}

	
	} // end of TA4 switch
	

	
	
	

* ***************************************
* Table S10: Summary of analytic variables (Table S5)
* ***************************************
if `TA10_alt_switch' == 1 {
clear
use Data/data_for_analysis.dta
keep if $sample==1

local i=0
local rownames ""
foreach var in bp_sar_inp bp_dar_inp  chl_inp hdl_inp  a1c_inp phqtot_high ///
	        hbp_dx_post_lottery chl_dx_post_lottery dia_dx_post_lottery dep_dx_post_lottery ///
		hbp_diure_med_inp  antihyperlip_med diabetes_med antidep_med ///
		cvd_risk_point ///
		health_change_noworse mcs8_score pcs8_score pain_low_inp poshappiness_bin_inp ///
		any_oop_inp tr_tot_spend_inp catastrophic_exp_inp owe_inp borrow_inp ///
		rx_num_mod_inp doc_num_mod_inp surg_num_mod_inp ed_num_mod_inp hosp_num_mod_inp ///
		chl_chk_inp fobt_chk_inp col_chk_inp did_flu_inp pap_chk_inp mam50_chk_inp psa_chk_inp ///
		usual_clinic_inp needmet_med_inp needmet_ment_inp needmet_rx_inp med_qual_inp ///
		obese smk_curr_bin_inp {
	       local `var'_cond = ""
	       local i =`i'+1
	       local rownames "`rownames' `var'"
	       }
	     
local cvd_risk_point_cond = "& age_inp>=30"
local fobt_chk_inp_cond = "& age_inp>=50"
local col_chk_inp_cond = "& age_inp>=50"
local did_flu_inp_cond = "& age_inp>=50"
local pap_chk_inp_cond = "& gender_inp==1"
local mam50_chk_inp_cond =  "& age_inp>=50 & gender_inp==1"
local psa_chk_inp_cond =  "& age_inp>=50 & gender_inp==0"
local med_qual_inp_cond = "& med_qual_inp!=0"
	      
matrix define A10_summary=J(`i', 2, .)
matrix rownames A10_summary=`rownames'
matrix colnames A10_summary=N_nomiss share_nomiss
	      
local i=0
foreach var in bp_sar_inp bp_dar_inp  chl_inp hdl_inp  a1c_inp phqtot_high ///
	        hbp_dx_post_lottery chl_dx_post_lottery dia_dx_post_lottery dep_dx_post_lottery ///
		hbp_diure_med_inp  antihyperlip_med diabetes_med antidep_med ///
		cvd_risk_point ///
		health_change_noworse mcs8_score pcs8_score pain_low_inp poshappiness_bin_inp ///
		any_oop_inp tr_tot_spend_inp catastrophic_exp_inp owe_inp borrow_inp ///
		rx_num_mod_inp doc_num_mod_inp surg_num_mod_inp ed_num_mod_inp hosp_num_mod_inp ///
		chl_chk_inp fobt_chk_inp col_chk_inp did_flu_inp pap_chk_inp mam50_chk_inp psa_chk_inp ///
		usual_clinic_inp needmet_med_inp needmet_ment_inp needmet_rx_inp med_qual_inp ///
		obese smk_curr_bin_inp {
	       local i =`i'+1
	       dis "`i' `var'"
	       quietly gen `var'_nomi = !missing(`var')
	       count if `var'_nomi==1
	       matrix A10_summary[`i', 1]= r(N)
	       sum `var'_nomi if 1==1 ``var'_cond'
	       matrix A10_summary[`i', 2]=r(mean)
	       count if `var'_nomi==1
	       matrix list A10_summary
	      }
	      
	
		} // End of TA10 switch

* ***************************************
* Table S11: Distribution of analytic variables (Table S11)
* ***************************************
if `TA11_alt_switch'==1 {
    clear
	use Data/data_for_analysis.dta
	keep if treatment==0 & $sample==1
	

	* Panel A:  Physiologic measures of health
	*bmi_inp  bp_sar_inp bp_dar_inp  chl_inp hdl_inp  a1c_inp 
	
	matrix summary_physiologic=J(6, 7,.)
	matrix rownames summary_physiologic=bmi_inp  bp_sar_inp bp_dar_inp  chl_inp hdl_inp  a1c_inp 
	matrix colnames summary_physiologic= mean sd p5 p25 med p75 p95
	
	local i=0
	foreach var in bmi_inp  bp_sar_inp bp_dar_inp  chl_inp hdl_inp  a1c_inp {
	local i=`i'+1
	count if missing(`var') & $sample==1
	*matrix summary_physiologic[`i', 8]=r(N)
	sum `var' [aw=weight_total_inp], detail
	local `var'_mean=round(r(mean), 10^(min(-2, int(log10(r(mean)))-2))) // rounding to at least 2 sig figs
	matrix summary_physiologic[`i', 1]=``var'_mean'
	local `var'_SD=round(r(sd), 10^(min(-2, int(log10(r(sd)))-2)))
	matrix summary_physiologic[`i', 2]=``var'_SD'
	local `var'_med=round(r(p5), 10^(min(-2, int(log10(r(p50)))-2)))
	matrix summary_physiologic[`i', 3]=``var'_med'
	local `var'_med=round(r(p25), 10^(min(-2, int(log10(r(p5)))-2)))
	matrix summary_physiologic[`i', 4]=``var'_med'
	local `var'_med=round(r(p50), 10^(min(-2, int(log10(r(p25)))-2)))
	matrix summary_physiologic[`i', 5]=``var'_med'
	local `var'_75=round(r(p75), 10^(min(-2, int(log10(r(p75)))-2)))
	matrix summary_physiologic[`i', 6]=``var'_75'
	local `var'_95=round(r(p95), 10^(min(-2, int(log10(r(p95)))-2)))
	matrix summary_physiologic[`i', 7]=``var'_95'
	}

	* Panel B: See end of the file
	
	* Panel C: Self-reported measures of health
	*health_last12_inp health_change_inp pcs8_score mcs8_score phqtot_inp phq_prob 
	
	matrix summary_self=J(4, 7,.)
	matrix rownames summary_self= pcs8_score mcs8_score phqtot_inp phq_prob 
	matrix colnames summary_self= mean sd p5 p25 med p75 p95
	
	foreach var in phqtot_inp phq_prob {
	replace `var'=100*`var'
	}
	
	local i=0
	foreach var in pcs8_score mcs8_score phqtot_inp phq_prob  {
	local i=`i'+1
	count if missing(`var') & $sample==1
	*matrix summary_self[`i', 8]=r(N)
	sum `var' [aw=weight_total_inp], detail
	local `var'_mean=round(r(mean), 10^(min(-2, int(log10(r(mean)))-2))) // rounding to at least 2 sig figs
	matrix summary_self[`i', 1]=``var'_mean'
	local `var'_SD=round(r(sd), 10^(min(-2, int(log10(r(sd)))-2)))
	matrix summary_self[`i', 2]=``var'_SD'
	local `var'_med=round(r(p5), 10^(min(-2, int(log10(r(p50)))-2)))
	matrix summary_self[`i', 3]=``var'_med'
	local `var'_med=round(r(p25), 10^(min(-2, int(log10(r(p5)))-2)))
	matrix summary_self[`i', 4]=``var'_med'
	local `var'_med=round(r(p50), 10^(min(-2, int(log10(r(p25)))-2)))
	matrix summary_self[`i', 5]=``var'_med'
	local `var'_75=round(r(p75), 10^(min(-2, int(log10(r(p75)))-2)))
	matrix summary_self[`i', 6]=``var'_75'
	local `var'_95=round(r(p90), 10^(min(-2, int(log10(r(p95)))-2)))
	matrix summary_self[`i', 7]=``var'_95'
	}
	
	*sf1_inp sf2_inp sf3_inp sf4_inp sf5_inp sf6_inp sf7_inp sf8_inp phq1_inp phq2_inp phq3_inp phq4_inp phq5_inp phq6_inp phq7_inp phq8_inp 
  
  * Panel C_1, C_2: see end of the the file

  * Panel D: Health care utilization
	*doc_num_mod_inp surg_num_mod_inp ed_num_mod_inp hosp_num_mod_inp
	*doc_num_incl_probe_inp surg_num_incl_probe_inp ed_num_incl_probe_inp hosp_num_incl_probe_inp
	
	local tablevars doc_num_mod_inp surg_num_mod_inp ed_num_mod_inp hosp_num_mod_inp rx_num_mod_inp
	local tablevars2 doc_num surg_num ed_num hosp_num	 
		
	matrix define summary_utilization = J(5, 9, .)
	matrix rownames summary_utilization = "doc" "surg" "ed" "hosp" "rx"
	matrix colnames summary_utilization = "% reporting any use" "Mean" "SD" "Median" "75%" "95%" "Censor cut point" "% Censored" "missing"


	local i = 0

	foreach var of local tablevars {
			local i = `i' + 1
			qui: count if `var'>0 & !missing(`var')
			local `var'_anyuse = r(N)  // anyuse
			qui: sum `var' [aw=weight_total_inp], d
			local `var'_Z_N = r(N) // total (including zeroes
			
			qui: sum `var' if `var'>0 [aw=weight_total_inp], d
			local `var'_N = r(N) // total  greater than 0
			local `var'_mean = r(mean)
			local `var'_sd = r(sd)
			local `var'_median = r(p50)
			local `var'_75 = r(p75)
			local `var'_95 = r(p95)
			
			
			count if missing(`var') & sample_inp_resp==1
			matrix summary_utilization[`i',9] = r(N)
			
		   
			matrix summary_utilization[`i',2] = round(``var'_mean',10^(min(-2, int(log10(``var'_mean'))-2)))
			matrix summary_utilization[`i',3] = round(``var'_sd',10^(min(-2, int(log10(``var'_sd'))-2)))
			matrix summary_utilization[`i',4] = round(``var'_median',10^(min(-2, int(log10(``var'_median'))-2)))
			matrix summary_utilization[`i',5] = round(``var'_75',10^(min(-2, int(log10(``var'_75'))-2)))
			matrix summary_utilization[`i',6] = round(``var'_95',10^(min(-2, int(log10(``var'_95'))-2)))
		}
		
		

	// Censoring stats
	local i = 0
	foreach var of local tablevars2 {
			local i = `i'+1
			
			gen `var'_anyuse = `var'_incl_probe_inp>0 & !missing(`var'_incl_probe_inp)
			sum `var'_anyuse if !missing(`var'_incl_probe_inp) [aw=weight_total_inp]
			local `var'_pctuse = r(mean)*100 // weighted percent reporting any
			drop `var'_anyuse
			
			
			matrix summary_utilization[`i',1] = round(``var'_pctuse',10^(min(-2, int(log10(``var'_pctuse'))-2)))
			
			qui: sum `var'_incl_probe_inp, d
			local `var'_N = r(N) // total (non censored)
			local `var'_cutoff = 2*r(p99) // Cutoff at 2* 99th percentile
			gen censored = `var'_incl_probe_inp != `var'_mod_inp // this checks if there was censoring
			count if censored==1
			local `var'_censored = r(N)
			matrix summary_utilization[`i',7] = ``var'_cutoff'
			sum censored if `var'_incl_probe_inp!=. [aw=weight_total_inp]
			matrix summary_utilization[`i',8] = round(r(mean),0.001) * 100
			drop censored
	}
	
	* add "percent reporting any"  for rx 
		sum rx_any_mod_inp [aw=weight_total_inp]
		local rx_num_mod_inp_pctuse = r(mean)*100 // weighted percent reporting any

		
		matrix summary_utilization[5,1] = round(`rx_num_mod_inp_pctuse',10^(min(-2, int(log10(`rx_num_mod_inp_pctuse'))-2)))
		
	* Panel E: Finances
	*any_oop_inp tr_tot_spend_inp catastrophic_exp_inp owe_inp borrow_inp
	
	local tablevars tr_tot_spend_inp 
	local tablevars2 tot_spend_inp 
		
	matrix define summary_finances = J(1, 9, .)
	matrix rownames summary_finances= "total" 
	matrix colnames summary_finances = "% reporting any use" "Mean" "SD" "Median" "75%" "95%" "Censor cut point" "% Censored" "missing"


	local i = 0

	foreach var of local tablevars {
			local i = `i' + 1
			qui: count if `var'>0 & !missing(`var')
			local `var'_anyuse = r(N)  // anyuse
			qui: sum `var' [aw=weight_total_inp], d
			local `var'_Z_N = r(N) // total (including zeroes
			
			qui: sum `var' if `var'>0 [aw=weight_total_inp], d
			local `var'_N = r(N) // total  greater than 0
			local `var'_mean = r(mean)
			local `var'_sd = r(sd)
			local `var'_median = r(p50)
			local `var'_75 = r(p75)
			local `var'_95 = r(p95)
			
			
			
			count if missing(`var') & sample_inp_resp==1
			matrix summary_finances[`i',9] = r(N)
			
		 
			matrix summary_finances[`i',2] = round(``var'_mean',10^(min(-2, int(log10(``var'_mean'))-2)))
			matrix summary_finances[`i',3] = round(``var'_sd',10^(min(-2, int(log10(``var'_sd'))-2)))
			matrix summary_finances[`i',4] = round(``var'_median',10^(min(-2, int(log10(``var'_median'))-2)))
			matrix summary_finances[`i',5] = round(``var'_75',10^(min(-2, int(log10(``var'_75'))-2)))
			matrix summary_finances[`i',6] = round(``var'_95',10^(min(-2, int(log10(``var'_95'))-2)))
		}

	// Censoring stats
	local i = 0
	foreach var of local tablevars2  {
			local i = `i'+1
			
			gen `var'_anyuse = `var'>0 & !missing(`var')
			sum `var'_anyuse if !missing(`var') [aw=weight_total_inp]
			local `var'_pctuse = r(mean)*100 // weighted percent reporting any
			drop `var'_anyuse
			
			matrix summary_finances[`i',1] = round(``var'_pctuse',10^(min(-2, int(log10(``var'_pctuse'))-2)))
			
			if `i'==1  { // on the the total spending variable is censored
			qui: sum `var', d
			local `var'_N = r(N) // total (non censored)
			local `var'_cutoff = 2*r(p99) // Cutoff at 2* 99th percentile
			gen censored = `var' != tr_`var' // this checks if there was censoring
			count if censored==1
			local `var'_censored = r(N)
			
			matrix summary_finances[`i',7] = ``var'_cutoff'
			
			sum censored if `var'!=. [aw=weight_total_inp]
			
			matrix summary_finances[`i',8] = round(r(mean),0.001) * 100
			
			drop censored
			}
	}

	
	
} // End of TA11 switch



* ***************************************
* Table S14: Versions of the main paper Table 2 (NEW - not analysis plan) for ages 50-64, pre-randomization "high-risk" and pre-randomization specific dx
* ***************************************
 if `TA14_alt_switch'==1 {
    
    	foreach var in  bp_sar_inp bp_dar_inp  bp_hyper hbp_dx_post_lottery hbp_diure_med_inp {
	local `var'_controls "gender_inp age_decile_dum*"
        }
	
	
        * Panel A
        local table TA14_A // ages 50-64
	* Locals for running the control distribution code
	local list_survey "bp_sar_inp bp_dar_inp bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_inp chl_h hdl_inp hdl_low chl_dx_post_lottery antihyperlip_med a1c_inp a1c_dia dia_dx_post_lottery diabetes_med phqtot_high dep_dx_post_lottery antidep_med"
	local condition_survey "& $sample==1 & age_inp>=50 & age_inp<=64"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum* age_inp"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_h hdl_low chl_dx_post_lottery antihyperlip_med a1c_dia dia_dx_post_lottery diabetes_med phqtot_high dep_dx_post_lottery antidep_med"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
   
    * Panel B
        local table TA14_B // pre-randomization high-risk
	* Locals for running the control distribution code 
	local list_survey "bp_sar_inp bp_dar_inp bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_inp chl_h hdl_inp hdl_low chl_dx_post_lottery antihyperlip_med a1c_inp a1c_dia dia_dx_post_lottery diabetes_med phqtot_high  dep_dx_post_lottery antidep_med"
	local condition_survey "& $sample==1 & any_dx_pre_lottery==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum* any_dx_pre_lottery"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_h hdl_low chl_dx_post_lottery antihyperlip_med a1c_dia dia_dx_post_lottery diabetes_med phqtot_high dep_dx_post_lottery antidep_med"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
    * Panel C
        local table TA14_C // BP, with pre-randomization hbp diagnosis
	* Locals for running the control distribution code
	local list_survey "bp_sar_inp bp_dar_inp bp_hyper hbp_diure_med_inp"
	local condition_survey "& $sample==1 & hbp_dx_pre_lottery==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum* hbp_dx_pre_lottery"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "bp_hyper hbp_diure_med_inp"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
    * Panel D
        local table TA14_D // HDL, with pre-randomization high cholesterol diagnosis
	* Locals for running the control distribution code
	local list_survey "chl_inp chl_h hdl_inp hdl_low antihyperlip_med"
	local condition_survey "& $sample==1 & chl_dx_pre_lottery==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum* chl_dx_pre_lottery"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "chl_h hdl_low antihyperlip_med"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
	* Panel E // A1C, with pre-randomization dia diagnosis
        local table TA14_E
	* Locals for running the control distribution code
	local list_survey "a1c_inp a1c_dia diabetes_med"
	local condition_survey "& $sample==1 & dia_dx_pre_lottery==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum* dia_dx_pre_lottery"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "a1c_dia diabetes_med"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
	* Panel F // Depression, with pre-randomization dep diagnosis
        local table TA14_F
	* Locals for running the control distribution code
	local list_survey "phqtot_high antidep_med"
	local condition_survey "& $sample==1 & dep_dx_pre_lottery==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum* dep_dx_pre_lottery"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "phqtot_high antidep_med"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do

	}	// End of TA14 switch

	
	
	
	
	
	
* ***************************************
* Table S15a: Logit versions:Physical Health Measures
* * ***************************************
if `TA15a_alt_switch' == 1 {
        dis "test2"

	foreach var in  bp_sar_inp bp_dar_inp  bp_hyper hbp_dx_inp hbp_diure_med_inp hbp_dx_post_lottery  {
	local `var'_controls "gender_inp age_decile_dum*"
        }
	
    * Panel A
        local table TA15a
	* Locals for running the control distribution code
	local list_survey "bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_h hdl_low chl_dx_post_lottery antihyperlip_med a1c_dia dia_dx_post_lottery diabetes_med phqtot_high dep_dx_post_lottery antidep_med"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_h hdl_low chl_dx_post_lottery antihyperlip_med a1c_dia dia_dx_post_lottery diabetes_med phqtot_high dep_dx_post_lottery antidep_med"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_logit.do
	
	foreach var in  bp_sar_inp bp_dar_inp  bp_hyper hbp_dx_inp hbp_diure_med_inp dia_dx_post_lottery  {
	local `var'_controls ""
        }
	
} // end of TA9a switch


* ****************************************
* Table S15b: Logit:Health Related Quality of Life
* *****************************************
if `TA15b_alt_switch' == 1 {
 
    * Panel A
        local table TA15b
	* Locals for running the control distribution code
	local list_survey "health_change_noworse pain_low_inp poshappiness_bin_inp"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample"
	local exclude_from_ste "poshappiness_bin_inp"
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "health_change_noworse pain_low_inp poshappiness_bin_inp"
	local ste=0
	
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_logit.do
	
	
} // End of TA9b switch

* ****************************************
* Table S15c:  Logit: Finances
* *****************************************
if `TA15c_alt_switch' ==1 {

        local table TA15c
	
	* Locals for running the control distribution code
	local list_survey "any_oop_inp owe_inp borrow_inp catastrophic_exp_inp"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "any_oop_inp owe_inp borrow_inp catastrophic_exp_inp"
	local ste=1
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_logit.do
	
}
* ****************************************
* Table S15d:  Logit: Utlization, Access and Quality, Prevention, and Behavior
* *****************************************
if `TA15d_alt_switch' ==1 {

        local table TA15d
	* Locals for running the control distribution code
	local list_survey "rx_any_mod_inp doc_any_incl_probe_inp surg_any_incl_probe_inp ed_any_incl_probe_inp hosp_any_incl_probe_inp  $prevention_list usual_clinic_inp needmet_med_inp med_qual_bin_inp smk_curr_bin_inp obese"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "rx_any_mod_inp doc_any_incl_probe_inp surg_any_incl_probe_inp ed_any_incl_probe_inp hosp_any_incl_probe_inp  $prevention_list usual_clinic_inp needmet_med_inp med_qual_bin_inp smk_curr_bin_inp obese"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_logit.do
	
} // End of A9d switch


* ****************************************
* Table S16a: Physical Health Measures, Alternative Specifications
* ****************************************
if `TA16a_alt_switch' == 1 {
	foreach var in  bp_sar_inp bp_dar_inp  bp_hyper hbp_dx_inp hbp_diure_med_inp hbp_dx_post_lottery  {
	local `var'_controls ""
        }
	
	local control_hh= "nnn*"
	local control_sa= "nnn* gender_inp age_decile_dum*"
	local control_all= "nnn* gender_inp age_decile_dum* race_black_inp race_nwother_inp hispanic_inp  english_list self_list first_day_list have_phone_list pobox_list"
	
    foreach control in hh sa all {
	
    * Panel 1
        local table TA16a_1_`control'
	* Locals for running the control distribution code
	local list_survey "bp_sar_inp bp_dar_inp bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_inp chl_h hdl_inp hdl_low chl_dx_post_lottery antihyperlip_med a1c_inp a1c_dia dia_dx_post_lottery diabetes_med phqtot_high  dep_dx_post_lottery antidep_med"
	local condition_survey "& $sample==1"
	local controls_survey "`control_`control''"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample `control_`control'' gender_inp age_decile_dum*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_h hdl_low chl_dx_post_lottery antihyperlip_med a1c_dia dia_dx_post_lottery diabetes_med phqtot_high dep_dx_post_lottery antidep_med"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_itt_inp.do
	
	local table TA16a_2_`control'
	* Locals for running the control distribution code
	local list_survey "cvd_risk_point"
	local condition_survey "& $sample==1 & age_inp>=30"
	local controls_survey "`control_`control''"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample `control_`control'' gender_inp age_decile_dum* age_inp"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "cvd_risk_point"
	local ste=0
	
	* Set up data
	clear
	use Data/data_for_analysis.dta
	

	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_itt_inp.do

	* Panel 3
	local table TA16a_3_`control'
	* Locals for running the control distribution code
	local list_survey "cvd_risk_point"
	local condition_survey "& $sample==1 & any_dx_pre_lottery==1 & age_inp>=30"
	local controls_survey "`control_`control''"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample `control_`control'' any_dx_pre_lottery gender_inp age_decile_dum* age_inp"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "cvd_risk_point"
	local ste=0
	
	* Set up data
	clear
	use Data/data_for_analysis.dta


	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_itt_inp.do
	
	* Panel 4
   local table TA16a_4_`control'
	* Locals for running the control distribution code
	local list_survey "cvd_risk_point"
	local condition_survey "& $sample==1 & age_inp>=50 & age_inp<=64"
	local controls_survey "`control_`control''"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample `control_`control'' age_inp gender_inp age_decile_dum*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "cvd_risk_point"
	local ste=0
	
	* Set up data
	clear
	use Data/data_for_analysis.dta

	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_itt_inp.do
	
     } // end of control loop
     
} // end of Table TA16a loop

* ****************************************
* Table S16b: Health related quality fo life, Alternative Specifications
* ****************************************
if `TA16b_alt_switch' == 1 {
 
	
	local control_hh= "nnn*"
	local control_sa= "nnn* gender_inp age_decile_dum*"
	local control_all= "nnn* gender_inp age_decile_dum* race_black_inp race_nwother_inp hispanic_inp english_list self_list first_day_list have_phone_list pobox_list"
	
    foreach control in hh sa all {
    
        local table TA16b_`control'
	* Locals for running the control distribution code
	local list_survey "health_change_noworse mcs8_score pcs8_score pain_low_inp poshappiness_bin_inp"
	local condition_survey "& $sample==1"
	local controls_survey "`control_`control''"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample `control_`control'' gender_inp age_decile_dum*"
	local exclude_from_ste "poshappiness_bin_inp"
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "health_change_noworse pain_low_inp poshappiness_bin_inp"
	local ste=0
	
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
	foreach var in  bp_sar_inp bp_dar_inp  bp_hyper {
	local `var'_controls ""
        }
    } // end of control loops
} // end of TA16b switch

* ****************************************
* Table S16c: Finances, Alternative Specifications
* ****************************************
if `TA16c_alt_switch' == 1 {
 
	local control_hh= "nnn*"
	local control_sa= "nnn* gender_inp age_decile_dum*"
	local control_all= "nnn* gender_inp age_decile_dum* race_black_inp race_nwother_inp hispanic_inp english_list self_list first_day_list have_phone_list pobox_list"
	
    foreach control in hh sa all {
	
        local table TA16c_`control'
	* Locals for running the control distribution code
	local list_survey "$oop_spending $debt_details"
	local condition_survey "& $sample==1"
	local controls_survey "`control_`control''"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample `control_`control'' gender_inp age_decile_dum*"
	local exclude_from_ste "any_oop_inp catastrophic_exp_inp"
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "any_oop_inp owe_inp borrow_inp catastrophic_exp_inp"
	local ste=0
	
	* Set up data
	clear
	use Data/data_for_analysis.dta


	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
     } // end of control loops	

} // end of TA16c switch

	
* ****************************************
* Table S16d: Utilization, Quality, and Prevention, Alternative Specifications
* ****************************************
if `TA16d_alt_switch' == 1 {
 
	local control_hh= "nnn*"
	local control_sa= "nnn* gender_inp age_decile_dum*"
	local control_all= "nnn* gender_inp age_decile_dum* race_black_inp race_nwother_inp hispanic_inp  english_list self_list first_day_list have_phone_list pobox_list"
	
    foreach control in hh sa all {
	
        local table TA16d_`control'
	* Locals for running the control distribution code
	local list_survey "rx_num_mod_inp doc_num_mod_inp surg_num_mod_inp ed_num_mod_inp hosp_num_mod_inp  $prevention_list usual_clinic_inp needmet_med_inp med_qual_bin_inp smk_curr_bin_inp obese"
	local condition_survey "& $sample==1"
	local controls_survey "`control_`control''"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample `control_`control'' gender_inp age_decile_dum*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 1
	local list_reverse "hdl_inp"
	local list_binary "usual_clinic_inp needmet_med_inp med_qual_bin_inp $prevention_list smk_curr_bin_inp obese"
	local ste=0
	local inp_12m_spend=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
     } // end of control loops
     
} // end of TA16d switch


* ****************************************
* Table S17: Other Binary Health Related Measures
* ****************************************
if `TA17_alt_switch' == 1 {

	foreach var in  bp_prehyper {
	local `var'_controls "gender_inp age_decile_dum*"
        }

        local table TA17
	* Locals for running the control distribution code
	local list_survey "bp_prehyper chl_high a1c_pre_dia"
	local condition_survey "& $sample==1"
	local controls_survey "$adjustments"
	local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample gender_inp age_decile_dum*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "hdl_inp"
	local list_binary "bp_prehyper chl_high a1c_pre_dia"
	local ste=0
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	
	* Run table
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	include Programs/tables_analysis_inp.do
	
} // End of TA17 switch


	
	if `TA18a_alt_switch' == 1 | `TA18b_alt_switch' == 1 | `TA18c_alt_switch' == 1 | `TA18d_alt_switch' == 1 | `TA18e_alt_switch' == 1{

local weight_1 = "weight_cross"
local weight_2 = "weight_cross"
local weight_3 = "weight_12m"
local weight_4 = "weight_total_inp"


local condition_1 = "sample_inp_resp==1 & sample_12m_resp==1"
local condition_2 = "sample_inp_resp==1 & sample_12m_resp==1"
local condition_3 = "sample_12m_resp==1"
local condition_4 = "sample_inp_resp==1"

local controls_1 = "ddd*"
local controls_2 = "nnn*"
local controls_3 = "ddd*"
local controls_4 = "nnn*"


local data_1 = "Data/survey12m_inperson_vars.dta"
local data_2 = "Data/data_for_analysis.dta"
local data_3 = "Data/survey12m_inperson_vars.dta"
local data_4 = "Data/data_for_analysis.dta"

}

*********************************************
* TABLE S18a: Physical Health Measures, sample comparisons
*********************************************	
if `TA18a_alt_switch' == 1 {
    
    local list_1 = "dep_screen_12m health_genflip_bin_12m health_notpoor_12m health_chgflip_bin_12m poshappiness_bin_12m"
    local list_2 = "phqtot_high health_last12_good health_last12_notbad health_change_noworse poshappiness_bin_inp"
    local list_3 = "dep_screen_12m health_genflip_bin_12m health_notpoor_12m health_chgflip_bin_12m poshappiness_bin_12m"
    local list_4 = "phqtot_high health_last12_good health_last12_notbad health_change_noworse poshappiness_bin_inp"

    
    local list_binary_1 = "dep_screen_12m health_genflip_bin_12m health_notpoor_12m health_chgflip_bin_12m poshappiness_bin_12m"
    local list_binary_2 = "phqtot_high health_last12_good health_last12_notbad health_change_noworse poshappiness_bin_inp"
    local list_binary_3 = "dep_screen_12m health_genflip_bin_12m health_notpoor_12m health_chgflip_bin_12m poshappiness_bin_12m"
    local list_binary_4 = "phqtot_high health_last12_good health_last12_notbad health_change_noworse poshappiness_bin_inp"
   
    foreach i of numlist 1/4 {
    
        local table TA18a_`i'
	* Locals for running the control distribution code
	local list_survey "`list_`i''"
	local condition_survey "& `condition_`i''"
	local controls_survey "`controls_`i''"
	local weight_survey "`weight_`i''"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "samp*"
	local exclude_from_ste "poshappiness_bin_inp poshappiness_bin_12m"
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse "dep_screen_12m phqtot_high"
	local list_binary "`list_binary_`i''"
	local ste=1
	local inp_12m_spend=0 // need this condition if spend_est is 1
		
	* Set up data
	clear
	use `data_`i''
	
	if inlist(`i', 1, 3)  {
	merge 1:1 person_id using Data/data_for_analysis.dta, keepusing(weight_total_inp sample_inp_resp)
	}
	
	gen weight_cross=weight_12m*weight_total_inp
	
	* Run table
	if `i'==2 | `i' == 4 {
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	}
	
	include Programs/tables_analysis_inp.do
    } // end of panel loop
	
} // End of S18a switch



*********************************************
* TABLE S18b: FINANCE ESTIMATES, sample comparisons
*********************************************	
if `TA18b_alt_switch'==1 {

     local list_1 = "cost_any_oop_12m cost_any_owe_12m cost_borrow_12m"
     local list_2 = "any_oop_inp owe_inp borrow_inp"
     local list_3 = "cost_any_oop_12m cost_any_owe_12m cost_borrow_12m"
     local list_4 = "any_oop_inp owe_inp borrow_inp"
   
     local list_binary_1 = "cost_any_oop_12m cost_any_owe_12m cost_borrow_12m"
     local list_binary_2 = "any_oop_inp owe_inp borrow_inp"
     local list_binary_3 = "cost_any_oop_12m cost_any_owe_12m cost_borrow_12m"
     local list_binary_4 =  "any_oop_inp owe_inp borrow_inp"
  

     foreach i of numlist 1/4 {
        local table TA18b_`i'
	* Locals for running the control distribution code
	local list_survey "`list_`i''"
	local condition_survey "& `condition_`i''"
	local controls_survey "`controls_`i''"
	local weight_survey "`weight_`i''"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "samp*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "`list_binary_`i''"
	local ste=1
		
	* Set up data
	clear
	use `data_`i''
	
	if inlist(`i', 1, 3)  {
	merge 1:1 person_id using Data/data_for_analysis.dta, keepusing(weight_total_inp sample_inp_resp)
	}
	
	gen weight_cross=weight_12m*weight_total_inp
	
	* Run table
	if `i'==2 | `i' == 4 {
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	}
	include Programs/tables_analysis_inp.do
	
    } // end of panel loop

} // end of TA18b switch

*********************************************
* TABLE S18c: Utlization Quality, sample comparisons
*********************************************	
if `TA18c_alt_switch' ==1 {

     local list_1 = "rx_num_mod_12m doc_num_mod_12m er_num_mod_12m hosp_num_mod_12m"
     local list_2 = "rx_num_mod_inp doc_num_mod_inp surg_num_mod_inp ed_num_mod_inp hosp_num_mod_inp"
     local list_3 = "rx_num_mod_12m doc_num_mod_12m er_num_mod_12m hosp_num_mod_12m"
     local list_4 = "rx_num_mod_inp doc_num_mod_inp surg_num_mod_inp ed_num_mod_inp hosp_num_mod_inp"
     
     local list_binary_1 = ""
     local list_binary_2 = ""
     local list_binary_3 = ""
     local list_binary_4 = ""
     
      local inp_12m_spend_1 = 1 // using 12m spending formula
     local inp_12m_spend_2 = 1 // using inp spending formula
     local inp_12m_spend_3 = 1
     local inp_12m_spend_4 = 1

         foreach i of numlist 1/4 {
        local table TA18c_`i'
	* Locals for running the control distribution code
	local list_survey "`list_`i''"
	local condition_survey "& `condition_`i''"
	local controls_survey "`controls_`i''"
	local weight_survey "`weight_`i''"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "samp*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 1
	local list_reverse ""
	local list_binary "`list_binary_`i''"
	local ste=0
	local inp_12m_spend=`inp_12m_spend_`i''
	
		
	* Set up data
	clear
	use `data_`i''
	
	if inlist(`i', 1, 3)  {
	merge 1:1 person_id using Data/data_for_analysis.dta, keepusing(weight_total_inp sample_inp_resp)
	
		foreach var in doc_num_mod_12m er_num_mod_12m hosp_num_mod_12m {
		replace `var'=2*`var'
		}
	
	}
	
	gen weight_cross=weight_12m*weight_total_inp
	
	* Run table
	if `i'==2 | `i' == 4 {
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	}
	include Programs/tables_analysis_inp.do
	
    } // end of panel loop
   local inp_12m_spend=0
    	
} // end of TA18c switch

*********************************************
* TABLE S18d: Prevention, sample comparisons
*********************************************	
if `TA18d_alt_switch' ==1 {

     local list_1 = "chl_chk_bin_12m pap_chk_bin_12m mam_chk_bin_12m"
     local list_2 = "chl_chk_inp pap_chk_inp mam50_chk_inp"
     local list_3 = "chl_chk_bin_12m pap_chk_bin_12m mam_chk_bin_12m"
     local list_4 = "chl_chk_inp pap_chk_inp mam50_chk_inp"
   
     local list_binary_1 = "chl_chk_bin_12m pap_chk_bin_12m mam_chk_bin_12m"
     local list_binary_2 = "chl_chk_inp pap_chk_inp mam50_chk_inp"
     local list_binary_3 = "chl_chk_bin_12m pap_chk_bin_12m mam_chk_bin_12m"
     local list_binary_4 = "chl_chk_inp pap_chk_inp mam50_chk_inp"
     

        foreach i of numlist 1/4 {
        local table TA18d_`i'
	* Locals for running the control distribution code
	local list_survey "`list_`i''"
	local condition_survey "& `condition_`i''"
	local controls_survey"`controls_`i''"
	local weight_survey "`weight_`i''"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "samp*"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "`list_binary_`i''"
	local ste=1
	local inp_12m_spend=0
	
		
	* Set up data
	clear
	use `data_`i''
	
	if inlist(`i', 1,  3)  {
	merge 1:1 person_id using Data/data_for_analysis.dta, keepusing(weight_total_inp sample_inp_resp)
	
	}
	
	gen weight_cross=weight_12m*weight_total_inp
	
	* Run table
	if `i'==2 | `i' == 4 {
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	}
	include Programs/tables_analysis_inp.do
	
    } // end of panel loop
   local inp_12m_spend=0
    	
} // end of TA18d switch

*********************************************
* TABLE S18e: Quality and behavior, sample comparisons
*********************************************	
if `TA18e_alt_switch' ==1 {

     local list_1 = "usual_clinic_12m needmet_med_12m med_qual_bin_12m smk_curr_bin_12m"
     local list_2 = "usual_clinic_inp needmet_med_inp med_qual_bin_inp smk_curr_bin_inp"
     local list_3 = "usual_clinic_12m needmet_med_12m med_qual_bin_12m smk_curr_bin_12m"
     local list_4 = "usual_clinic_inp needmet_med_inp med_qual_bin_inp smk_curr_bin_inp"

     
     local list_binary_1 = "usual_clinic_12m needmet_med_12m med_qual_bin_12m smk_curr_bin_12m"
     local list_binary_2 = "usual_clinic_inp needmet_med_inp med_qual_bin_inp smk_curr_bin_inp"
     local list_binary_3 = "usual_clinic_12m needmet_med_12m med_qual_bin_12m smk_curr_bin_12m"
     local list_binary_4 = "usual_clinic_inp needmet_med_inp med_qual_bin_inp smk_curr_bin_inp"

        foreach i of numlist 1/4 {
        local table TA18e_`i'
	* Locals for running the control distribution code
	local list_survey "`list_`i''"
	local condition_survey "& `condition_`i''"
	local controls_survey "`controls_`i''"
	local weight_survey "`weight_`i''"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "samp*"
	local exclude_from_ste "smk_curr_bin_inp smk_curr_bin_12m"
	local mult_inf "`mult_inf_switch'"
	local spend_est 0
	local list_reverse ""
	local list_binary "`list_binary_`i''"
	local ste=1
	local inp_12m_spend=0
	
		
	* Set up data
	clear
	use `data_`i''
	
	if inlist(`i', 1,  3)  {
	merge 1:1 person_id using Data/data_for_analysis.dta, keepusing(weight_total_inp sample_inp_resp)
	
	
	}
	
	gen weight_cross=weight_12m*weight_total_inp
	
	* Run table
	if `i'==2 | `i' == 4 {
	drop ohp_all_ever_survey
	rename ohp_all_ever_inperson ohp_all_ever_survey
	}
	include Programs/tables_analysis_inp.do
	
    } // end of panel loop
   local inp_12m_spend=0
    	
} // end of TA18e switch


*********************************************
* TABLE S19: Observational Estimates/OLS on Health Measures
*********************************************	
if `TA19_alt_switch'==1 {
* Locals for running the analysis table code


	foreach var in  bp_sar_inp bp_dar_inp bp_prehyper bp_hyper hbp_dx_post_lottery hbp_diure_med_inp {
	local `var'_controls "gender_inp age_decile_dum*"
        }
	
	local var_interest_survey1 "treatment"
	local condition1 "& $sample==1"
	local count_condition1 "& $sample==1"
	local insured_condition1 "& $sample==1 & ohp_all_ever_inp==1"
	
	local var_interest_survey2 "ohp_all_ever_inp"
	local condition2 "& $sample==1"
	local count_condition2 "& $sample==1"
	local insured_condition2 "& $sample==1 & ohp_all_ever_inp==1"
	
	local var_interest_survey3 "ohp_all_ever_inp"
	local condition3 "& $sample==1 & treatment==0"
	local count_condition3 "& $sample==1 & treatment==0"
	local insured_condition3 "& $sample==1 & ohp_all_ever_inp==1 & treatment==0"
	
	local var_interest_survey4 "ohp_std_ever_inp"
	local condition4 "& $sample==1 & treatment==1"
	local count_condition4 "& $sample==1 & ohp_std_ever!=. & treatment==1"
	local insured_condition4 "& $sample==1 & ohp_std_ever_inp==1 & treatment==1"

	* survey
		
	
	forv n=1/4 {
	local table "TA19_panel`n'"
	local list_survey "bp_sar_inp bp_dar_inp bp_hyper hbp_dx_post_lottery hbp_diure_med_inp chl_inp chl_h hdl_inp hdl_low chl_dx_post_lottery antihyperlip_med a1c_inp a1c_dia dia_dx_post_lottery diabetes_med phqtot_high dep_dx_post_lottery antidep_med"
	local var_interest "`var_interest_survey`n''"
	
	local condition_survey "`condition`n''"
	local controls_survey "$adjustments " // original specification doesn't have age and gender controln (age_decile_dum* gender_inp)
        local weight_survey "$weight"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""
	
	local add_vars "$sample ohp_std_ever* ohp_all_ever* gender_inp age_decile_dum*"
	
	* Set up data
	clear
	use Data/data_for_analysis.dta

	include Programs/tables_observational_inp.do
	}


	* clean up var controls
	foreach var in  bp_sar_inp bp_dar_inp bp_prehyper bp_hyper hbp_dx_post_lottery hbp_diure_med_inp {
		local `var'_controls ""
		}
		
	* Set up matrix
	matrix TableTA19_top = J(2,4,.)
	matrix rownames TableTA19_top = "SampleSize" "%insured" 
	matrix colnames TableTA19_top = "Random" "AnyvsNo" "AnyvsNoCONTROLS" "OHPStdvsNoTREATMENT"

	clear
		use Data/data_for_analysis.dta

	forv n=1/4 {
	count if 1==1 `count_condition`n''
	local samplesize`n'=r(N)
	count if 1==1 `insured_condition`n''
	local pctinsured`n'=r(N)/`samplesize`n''
	matrix TableTA19_top[1,`n'] = `samplesize`n''
	matrix TableTA19_top[2,`n'] = round(`pctinsured`n''*100)
	}


} // end of TA14 switch





	
log close

*******************
*Displaying Tables
*******************

quietly {
log using paper_tables.log, replace
 matrix define Filler_master=J(15,15,.)

	if `T1_alt_switch' == 1 {
	noi: di _newline
	noi: matrix list T1_alt, title(Table 1: Sample Characteristics) 
	noi: di _newline
	}

	if `T2_alt_switch' == 1 {
	noi: di _newline
	matrix T2_alt=T2_A_alt\T2_B_alt\T2_C_alt\T2_D_alt
	noi: matrix list T2_alt
	noi di "TableT2_A_N_survey is `TableT2_A_N_survey'" 
	}
	
	if `T3_alt_switch' == 1 {
	noi: di _newline
	noi: matrix list T3_alt, title(Table 3: Health related quality of life) 
	noi: di _newline
	}
	
	if `T4_alt_switch'==1 {
	noi: di _newline
	matrix TableT4 = T4_alt[1...,1...]
	noi: matrix list TableT4, title(Plan Table 4: Finances)
	noi: di "Sample is control survey responders with pre-lottery diagnosis of depression (N = `TableP4_N_survey')"
	}
	
	if `T5_alt_switch' ==1 {
	noi: di _newline
	noi: matrix list T5_alt, title(Alternative Table 5: Utilization, Access and Preventive Care, Behavior) 
	noi: di _newline
	noi: matrix list spend_est_T5
	noi: di _newline
	}
	
	if `TA5_alt_switch'==1 {
	noi: di _newline
	noi: matrix list TableTA5
	}
	
	if `TA6_alt_switch'==1 {
	noi: di _newline
	noi: matrix list sum_weights, title (Table A6: Summary of Weights)
	noi: di _newline
		}
	
	if `TA7_alt_switch'==1 {
		noi: di _newline
		matrix TableTA7 = TA7_1A_alt[1...,1...]\TA7_1_alt[2...,1...] \TA7_2_alt[51..., 1...]\TA7_3_alt[45..., 1...]\TA7_4_alt[45..., 1...]\TA7_5_alt[43..., 1...]
		noi: matrix list TableTA7, title(Table A2: Balance of treatment and controls)
		*noi: matrix list TableTA2_1
		*noi: matrix list TableTA2_1A
		}
	
	if `TA8_alt_switch'==1 {
	noi: di _newline
	matrix TableTA8_2_part=TableTA8_2[23..23, 2..2]\TableTA8_2[23..23, 4..4]
	matrix Filler_TA8_part= Filler_master[1..2, 1..5]
	matrix TableTA8_2_w_filler=(Filler_TA8_part, TableTA8_2_part)
	matrix rownames TableTA8_2_w_filler=ste p
	matrix TA8_alt=TA8_1_alt[1..11,1...]\TA8_1_alt[13..15,1...]\TableTA8_2_w_filler\TA8_1_alt[15..15,1...]\TA8_1_alt[12..12,1...]
	noi: matrix list TA8_alt, title(Table A8: Balance of pre_randomization diagnoses)

	noi: di "Sample is control survey responders (N = `TableTA8_1_N')"
		}

	if `TA9_alt_switch' == 1 {
	noi: di _newline
	matrix TA9_alt=TA9_A_alt
	noi: matrix list TA9_alt, title (Table A9: First Stage)
		}	

	if `TA10_alt_switch' == 1 {
	noi: matrix list A10_summary, title(Table A5: Summary of Variables)
		}
	

	if `TA11_alt_switch' == 1 {

	clear
	use Data/data_for_analysis.dta
	keep if $sample==1 & treatment==0
		noi: di _newline
		noi: matrix list summary_physiologic
		noi tab health_last12_inp [aw=weight_total_inp]
		noi tab health_change_inp [aw=weight_total_inp]
		noi: matrix list summary_self
		noi: matrix list summary_utilization
		noi: matrix list summary_finances
	}
	

	if `TA14_alt_switch' ==1 {
	
	noi matrix list TA14_A_alt, title (Table A8_a:Physiologic Measures, restricted to aged 50 and older)
	noi di "sample size is `TableTA8_A_N_survey'" 
	noi matrix list TA14_B_alt, title (Table A8_b:Physiologic Measures, pre-randomization high risk)
	noi di "sample size is `TableTA8_B_N_survey'"
	matrix TA14_3=TA14_C_alt\TA14_D_alt\TA14_E_alt\TA14_F_alt
	noi matrix list TA14_3, title (Table A14_c:Physiologic Measures, restricted to aged 50 and older)
	noi di "sample sizes are `TableTA14_C_N_survey', `TableTA14_D_N_survey', `TableTA14_E_N_survey', `TableTA14_F_N_survey'"
		
		}
	
	if `TA15a_alt_switch' == 1 {
	noi: di _newline
	matrix logit_TA15a_alt=logit_TA15a_alt
	noi: matrix list logit_TA15a_alt, title (Table TA15a: Logit version, Physical Health Measures)
	noi: di _newline
	}
	
	if `TA15b_alt_switch' == 1 {
	noi: di _newline
	noi: matrix list logit_TA15b_alt, title (Table TA15b: Logit, Health Related Quality of Life)
	noi: di _newline
	}
	
	if `TA15c_alt_switch' == 1 {
	noi: di _newline
	noi: matrix list logit_TA15c_alt, title (Table TA15c: Logit: Finances)
	noi: di _newline
	}
	
	if `TA15d_alt_switch' == 1 {
	noi: di _newline
	noi: matrix list logit_TA15d_alt, title (Table TA15d: Logit: Utlization, Access and Quality, Prevention, and Behavior)
	noi: di _newline
	}
	
	if `TA16a_alt_switch'== 1 {
	noi: di _newline
		foreach control in hh sa all {
		matrix TA16a_`control'_alt=TA16a_1_`control'_alt\TA16a_2_`control'_alt\TA16a_3_`control'_alt\TA16a_4_`control'_alt
		noi: matrix list TA16a_`control'_alt
		}
	}
	
	if `TA16b_alt_switch'== 1 {
	noi: di _newline
		foreach control in hh sa all {
		noi: matrix list TA16b_`control'_alt
		}
	}
	
	if `TA16c_alt_switch'== 1 {
	noi: di _newline
		foreach control in hh sa all {
		noi: matrix list TA16c_`control'_alt
		}
	}
	
	if `TA16d_alt_switch'== 1 {
	noi: di _newline
		foreach control in hh sa all {
		noi: matrix list TA16d_`control'_alt
		}
	}
	
	if `TA17_alt_switch' ==1 {
	noi: di _newline
	noi: matrix list TA17_alt, title(Alternative Table 11: Other Binary Measures) 
	noi: di _newline
	}

	if `TA18a_alt_switch' ==1 {
	noi: di _newline
	noi: di "Table S18: Panel A"
	noi: di "12m results for 12m respondents"
	noi: matrix list TA18a_3_alt
	noi: di "N=`TableTA18a_3_N_survey'"
	noi: matrix list TableTA18a_3
	noi: di _newline
	noi: di "Table S18: Panel A"
	noi: di "inperson results for inperson respondents"
	noi: matrix list TA18a_4_alt
	noi: di "N=`TableTA18a_4_N_survey'"
	noi: matrix list TableTA18a_4
	noi: di _newline
	noi: di "Table S18: Panel E"
	noi: di "12m results for inperson respondents"
	noi: matrix list TA18a_1_alt
	noi: di "N=`TableTA18a_1_N_survey'"
	noi: matrix list TableTA18a_1
	noi: di _newline
	noi: di "Table S18: Panel E"
	noi: di "inperson results for 12m respondents"
	noi: matrix list TA18a_2_alt
	noi: di "N=`TableTA18a_2_N_survey'"
	noi: matrix list TableTA18a_2
	
	}
	
	if `TA18b_alt_switch' ==1 {
	noi: di _newline
	
	noi: matrix list TableTA18b_2
	noi: di _newline
	noi: di "Table S18: Panel B"
	noi: di "12m results for 12m respondents"
	noi: matrix list TA18b_3_alt
	noi: di "N=`TableTA18b_3_N_survey'"
	noi: matrix list TableTA18b_3
	noi: di _newline
	noi: di "Table S18: Panel B"
	noi: di "inperson results for inperson respondents"
	noi: matrix list TA18b_4_alt
	noi: di "N=`TableTA18b_4_N_survey'"
	noi: matrix list TableTA18b_4
	noi: di _newline
	noi: di "Table S18: Panel F"
	noi: di "12m results for inperson respondents"
	noi: matrix list TA18b_1_alt
	noi: di "N=`TableTA18b_1_N_survey'"
	noi: matrix list TableTA18b_1
	noi: di _newline
	noi: di "Table S18: Panel F"
	noi: di "inperson results for 12m respondents"
	noi: matrix list TA18b_2_alt
	noi: di "N=`TableTA18b_2_N_survey'"
	noi: matrix list TableTA18b_2
	
	}
	
	
	if `TA18c_alt_switch' ==1 {
	noi: di _newline
	noi: di "Table S18: Panel C"
	noi: di "12m results for 12m respondents"
	noi: matrix list TA18c_3_alt
	noi: matrix list spend_est_TA18c_3
	noi: di "N=`TableTA18c_3_N_survey'"
	noi: matrix list TableTA18c_3
	noi: di _newline
	noi: di "Table S18: Panel C"
	noi: di "inperson results for inperson respondents"
	noi: matrix list TA18c_4_alt
	noi: matrix list spend_est_TA18c_4
	noi: di "N=`TableTA18c_4_N_survey'"
	noi: matrix list TableTA18c_4
	noi: di _newline
	noi: di "Table S18: Panel G"
	noi: di "12m results for inperson respondents"
	noi: matrix list TA18c_1_alt
	noi: matrix list spend_est_TA18c_1
	noi: di "N=`TableTA18c_1_N_survey'"
	noi: matrix list TableTA18c_1
	noi: di _newline
	noi: di "Table S18: Panel G"
	noi: di "inperson results for 12m respondents"
	noi: matrix list TA18c_2_alt
	noi: matrix list spend_est_TA18c_2
	noi: di "N=`TableTA18c_2_N_survey'"
	noi: matrix list TableTA18c_2
	noi: di _newline

	}
	
	if `TA18d_alt_switch' ==1 {
	noi: di _newline
	noi: di "Table S18: Panel D"
	noi: di "12m results for 12m respondents"
	noi: matrix list TA18d_3_alt
	noi: di "N=`TableTA18d_3_N_survey'"
	noi: matrix list TableTA18d_3
	noi: di _newline
	noi: di "Table S18: Panel D"
	noi: di "inperson results for inperson respondents"
	noi: matrix list TA18d_4_alt
	noi: di "N=`TableTA18d_4_N_survey'"
	noi: matrix list TableTA18d_4
	noi: di _newline
	noi: di _newline
	noi: di "Table S18: Panel H"
	noi: di "12m results for inperson respondents"
	noi: matrix list TA18d_1_alt
	noi: di "N=`TableTA18d_1_N_survey'"
	noi: matrix list TableTA18d_1
	noi: di _newline
	noi: di "Table S18: Panel H"
	noi: di "inperson results for 12m respondents"
	noi: matrix list TA18d_2_alt
	noi: di "N=`TableTA18d_2_N_survey'"
	noi: matrix list TableTA18d_2
	}
	
	if `TA18e_alt_switch' ==1 {
	noi: di _newline
	noi: di "Table S18: Panel D"
	noi: di "12m results for 12m respondents"
	noi: matrix list TA18e_3_alt
	noi: di "N=`TableTA18e_3_N_survey'"
	noi: matrix list TableTA18e_3
	noi: di _newline
	noi: di "Table S18: Panel D"
	noi: di "inperson results for inperson respondents"
	noi: matrix list TA18e_4_alt
	noi: di "N=`TableTA18e_4_N_survey'"
	noi: matrix list TableTA18e_4
	noi: di _newline
	noi: di _newline
	noi: di "Table S18: Panel H"
	noi: di "12m results for inperson respondents"
	noi: matrix list TA18e_1_alt
	noi: di "N=`TableTA18e_1_N_survey'"
	noi: matrix list TableTA18e_1
	noi: di _newline
	noi: di "Table S18: Panel H"
	noi: di "inperson results for 12m respondents"
	noi: matrix list TA18e_2_alt
	noi: di "N=`TableTA18e_2_N_survey'"
	noi: matrix list TableTA18e_2
	}
	
	
	if `TA19_alt_switch'==1 {
	noi: di _newline
	matrix TableTA19_bottom=[TableTA19_panel1[1...,2...], TableTA19_panel2[1...,2...], TableTA19_panel3[1...,2...], TableTA19_panel4[1...,2...]]
	matrix TableTA19=TableTA19_top\TableTA19_bottom
	noi: matrix list TableTA19
	
	}

	
log close
} // end of quietly loop

capture log close
