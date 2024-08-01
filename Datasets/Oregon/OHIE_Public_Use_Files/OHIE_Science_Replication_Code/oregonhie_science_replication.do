/**************************************
*This code replicates tables from Taubman et al (Science XX)
It runs on Stata versions 13 and earlier. 								

Tables replicated are:

Table 1	(F statistics do not replicate because not all variables are public (median household income in zip code is not publicly available)
Table 3 (percent with any visits only)
Table 4 
Table 5 (Panels B and C)

Table S1 (full table, except for variables pertaining to zip code)
Table S3 (percent with any visits only)
Table S4 (percent with any visits only)
Table S5 (Panels B and C)
Table S6 (percent with any visits only; F statistics do not replicate)
Table S7
Table S8 (percent with any visits only)
Table S11 (percent with any visits)
Table S12
Table S13 (percent with any visits only; number of visits and global test of sorting do NOT replicate)
Table S14 (percent with any visits except for "prior credit" subpopulation)
Table S15 (percent with any visits; Columns 3 and 6 do not replicate (median household income in zip code is not publicly available))
Table S16 (logit estimates only)
Table S17 (panels A and B)

The table-generating code is divided among several .do files, all of which are 
called by this master file.

*To run this file:*

Save this .do file to the folder where you wish to do your work.

Save the following .do file to a subfolder named "Programs"
	ed_prepare_data.do
	ed_first_stage.do
	ed_table_means.do
	ed_table2.do
	ed_tables_analysis.do
	ed_tables_balance.do
	ed_tables_logit.do
	global_sort_test.do
	ed_table_a14.do
	ed_proportional_check.do
	ed_tables_observational.do
	
Save the following datasets to a subfolder named "Data":
	oregonhie_descriptive_vars.dta
	oregonhie_inperson_vars.dta
	oregonhie_stateprograms_vars.dta
	oregonhie_survey12m_vars.dta
	oregonhie_ed_vars.dta

Edit the local "switches" below to choose which tables to run. For example, 
leaving the code as "local T1_switch=1" will replicate Table 1. Changing the
code to "local T1_switch=0" will cause Table 1 to not be replicated. 

The full log file outputs to oregon_hie_science_replication.log. In addition, a 
log file with just the generated tables outputs to all_tables.log. 

**************************************/


clear all
clear matrix 

set more off
set seed 1078654912
capture log close
log using oregon_hie_science_replication.log, text replace


**********************
*Prepare Data
***********************

*include Programs/ed_prepare_data.do

***********************
*Globals and Locals

*Regression settings
local weight_30sep2009 "1"
local weight_09mar2008 "1"		
global adjustments "nnn*"
global sample "sample_ed"

global ALT = 0

global baseline_list "female_list birthyear_list english_list self_list first_day_list have_phone_list pobox_list zip_hh_inc_list"
local spend_est=0

***************************
*Table-generating Switches
***************************

**Main Tables**
local T1_switch=1 // Table 1: Balance 
local T2_switch=0 // Table 2: ED Usage Timing and number of visits (Also Table A2) 
local T3_switch=0 // Table 3: Emergency Department Utilization by Hospital Admission and Timing (Also Table A3)
local T4_switch=0 // Table 4: Emergency Department Utilization by Type of Visit (also Table A4)
local T5_switch=0 // Table 5: Comparing Administrative Data and Survey Data Results (also Table A5)

**Appendix Tables**

local TA1_switch=0 // Table A1: ED vs Full Sample 
**Tables A2-A5 outputted by the T2-T5 switches**
**Table A6 outputted by T1 switch**
local TA7_switch=0 // Table A7: First Stage
local TA8_switch=0 // Table A8: Summary of Analytic Variables (Control Sample) 
*local TA9_switch=0 // Table A9: Comparison of ED Visits in Different Populations *DOES NOT RUN ON PUBLICLY AVAILABLE DATA
*local TA10_switch=0 // Table A10: Select Conditions (Control Sample) *DOES NOT RUN ON PUBLICLY AVAILABLE DATA
local TA11_switch=0 // Table A11: Emergency Department Use for Selected Conditions
local TA12_switch=0 // Table A12: List Charges 
local TA13_switch=0 // Table A13: ED use by Hospital Type
local TA14_switch=0 // Table A14: Heterogeneity
local TA15_switch=0 // Table A15: Sensitivity of Results to Choice of Covariates
local TA16_switch=0 // Table A16: Sensitivity of Results to Functional Form
local TA17_switch=0 // Table A17: Observational Estimates


local sensitivity=0 // Leave as 0 (gets set as 1 within the TableA15 loop)


foreach date in 30sep2009 {  

global binary "birthyear_list female_list english_list self_list first_day_list have_phone_list pobox_list ed_visit_`date' ed_hosp_`date' ed_out_`date' ed_off_`date' ed_on_`date' ed_ne_`date' ed_epct_`date' ed_edcnpa_`date' ed_edcnnpa_`date' ed_unclas_`date' ed_inj_`date' ed_skin_`date' ed_abdo_`date' ed_back_`date' ed_psy_`date' ed_heart_`date' ed_head_`date' ed_depres_`date' ed_sub_`date' ed_psysub_`date' ed_chron_`date' ed_hiun_`date' ed_loun_`date' ed_acsc_`date'"

global balance_binary "female_list english_list self_list first_day_list have_phone_list pobox_list ed_visit_09mar2008 ed_hosp_09mar2008 ed_out_09mar2008 ed_off_09mar2008 ed_on_09mar2008 ed_ne_09mar2008 ed_epct_09mar2008 ed_edcnpa_09mar2008 ed_edcnnpa_09mar2008 ed_unclas_09mar2008 ed_inj_09mar2008 ed_skin_09mar2008 ed_abdo_09mar2008 ed_back_09mar2008 ed_psy_09mar2008 ed_heart_09mar2008 ed_head_09mar2008 ed_depres_09mar2008 ed_sub_09mar2008 ed_psysub_09mar2008 ed_chron_09mar2008 ed_hiun_09mar2008 ed_loun_09mar2008 ed_acsc_09mar2008"

************************
*Table 1: Balance 
**************************
	if `T1_switch'==1 {
	
	clear
	use Data/balance_data.dta
	local balance_table "T1"
	local balance_matchvar "sample_ed"
	
	local T2_vars "ed_visit_09mar2008"
	local T3_vars "ed_hosp_09mar2008 ed_out_09mar2008 ed_on_09mar2008 ed_off_09mar2008"
	local T4_vars "tr_ed_edcnnp_09mar2008 tr_ed_edcnpa_09mar2008 tr_ed_epct_09mar2008 tr_ed_ne_09mar2008"
	local T5_vars "ed_acsc_09mar2008 ed_chron_09mar2008 ed_inj_09mar2008 ed_skin_09mar2008 ed_abdo_09mar2008 ed_back_09mar2008 ed_heart_09mar2008 ed_head_09mar2008 ed_depres_09mar2008 ed_psysub_09mar2008"
	local prelottery_vars "`T2_vars' `T3_vars' `T4_vars' `T5_vars'" 
	
	/*local T2_vars "ed_visit_09mar2008 tr_ed_visit_09mar2008"
	local T3_vars "ed_hosp_09mar2008 tr_ed_hosp_09mar2008 ed_out_09mar2008 tr_ed_out_09mar2008 ed_on_09mar2008 tr_ed_on_09mar2008 ed_off_09mar2008 tr_ed_off_09mar2008"
	local T4_vars "tr_ed_edcnnp_09mar2008 tr_ed_edcnpa_09mar2008 tr_ed_epct_09mar2008 tr_ed_ne_09mar2008 tr_ed_unclas_09mar2008"
	local T5_vars "ed_acsc_09mar2008 tr_ed_acsc_09mar2008 ed_chron_09mar2008 tr_ed_chron_09mar2008 ed_inj_09mar2008 tr_ed_inj_09mar2008 ed_skin_09mar2008 tr_ed_skin_09mar2008 ed_abdo_09mar2008 tr_ed_abdo_09mar2008 ed_back_09mar2008 tr_ed_back_09mar2008 ed_heart_09mar2008 tr_ed_heart_09mar2008 ed_head_09mar2008 tr_ed_head_09mar2008 ed_depres_09mar2008 tr_ed_depres_09mar2008 ed_psysub_09mar2008 tr_ed_psysub_09mar2008"
	local prelottery_vars "`T2_vars' `T3_vars' `T4_vars' `T5_vars'" */
	
	local balance_list "birthyear_list female_list english_list self_list first_day_list have_phone_list pobox_list `prelottery_vars'"
	
	local balance_controls "$adjustments"
	local balance_condition "& $sample==1"
	local balance_subset "$baseline_list"
	
	local inflate_list "$balance_binary"
	
	local weight "`weight_`date''"
	
	include Programs/ed_tables_balance.do 

	*Make Table 1
	*matrix define TableT1= TableT1_`date'[1..24, 1..2] \ TableT1_`date'[130..137, 1..2]
	
		} // End Balance Table (T1) Loop
		
	
*************************
*Table 2: Emergency Department Utilization (Timing/Urgency)
**************************
	if `T2_switch'==1 {
	
	
	local ed_visit_30sep2009_controls "ed_visit_09mar2008"
	*local tr_ed_visit_30sep2009_controls "tr_ed_visit_09mar2008 tr_ed_visit_09mar2008_m" // This variable is not public: it has been replaced with a censored version
	
		local table "T2"
		dis "`table'"
		local fs_var "ohp_all_ever_30sep2009"
		
	foreach subpop in all zero one twoplus twoplus_out fiveplus {
	
	clear
	use Data/data_for_analysis
	
		local condition "& $sample==1 & `subpop'==1"
		
		local sheet "2"
		
		local list "ed_visit_`date'" 
		*local list "ed_visit_`date' tr_ed_visit_`date'" // This variable is not public: it has been replaced with a censored version
		
		local source "`date'"
	
		local weight "`weight_`date''"
		
		include Programs/ed_table2.do

			}
	matrix define TableT2_`date'= TableT2_all_`date' \ TableT2_zero_`date' \ TableT2_one_`date' \ TableT2_twoplus_`date' \ TableT2_twoplus_out_`date' \ TableT2_fiveplus_`date'
	matrix list TableT2_30sep2009
	
		} // End of Table 2 Loop
		

**************************
*Table 3: Admission and Timing
**************************
	if `T3_switch'==1 {
	
	clear
	use Data/data_for_analysis
	
	/*Controls for Total Margin // These variables are not public: they have been replaced with censored versions.
	foreach var in tr_ed_hosp tr_ed_out tr_ed_on tr_ed_off {
			local `var'_`date'_controls "`var'_09mar2008 `var'_09mar2008_m"
			di "`var'_`date'_controls"
			di ``var'_`date'_controls' 
		} */
	*Controls for Extensive Margin
	foreach var in ed_hosp ed_out ed_on ed_off {
			local `var'_`date'_controls "`var'_09mar2008"
		}
	
		local table "T3"
		dis "`table'"
		local condition "& $sample==1"
		local controls "$adjustments"
		
		local sheet "T3 (admission and timing)"
		
		local list "ed_hosp_`date' ed_out_`date' ed_on_`date' ed_off_`date'"
		*local list "ed_hosp_`date' ed_out_`date' ed_on_`date' ed_off_`date' tr_ed_hosp_`date' tr_ed_out_`date' tr_ed_on_`date' tr_ed_off_`date'" 
		
		local source "`date'"
	
		local weight "`weight_`date''"
		
		include Programs/ed_tables_analysis.do
		
			} // End of T3 Switch
		
**************************
*Table 4: Type of Visit
**************************	
	if `T4_switch'==1 {
	
	clear
	use Data/data_for_analysis
	
	*Controls for Total Margin 
	foreach var in tr_ed_edcnnp tr_ed_edcnpa tr_ed_epct tr_ed_ne tr_ed_unclas {
			local `var'_`date'_controls "`var'_09mar2008 `var'_09mar2008_m"
			di "`var'_`date'_controls"
			di ``var'_`date'_controls'
		} 
	
		local table "T4"
		dis "`table'"
		local condition "& $sample==1"
		local controls "$adjustments"
		
		local list "tr_ed_edcnnp_`date' tr_ed_edcnpa_`date' tr_ed_epct_`date' tr_ed_ne_`date' tr_ed_unclas_`date'" 
		
		local source "`date'"
	
		local weight "`weight_`date''"
		
		include Programs/ed_tables_analysis.do
	
		} // End of T4 Switch
			
		
		


***************************
*Table 5: Comparing administrative and survey data
****************************
	if `T5_switch'==1 {

	clear
	use Data/data_for_analysis.dta
	
	local mail_controls "draw_survey_12m ddd*" 
	
	local r=1
	foreach survey in mail inp {

	local table "TA5_`survey'"
	dis "`table'"
	
	local source "`survey'"
	local weight "weight_`survey'"
	
	local condition "& $sample==1"
	local condition "& `survey'_survey==1"
	
	local list_binary "$binary"
	
	local survey_controls "``survey'_controls'" 
	local controls "$adjustments"
	
	*Total margin variables not publicly available - censored versions have been provided
	local list "er_any_`survey' er_num_`survey' survey_any_`survey' tr_survey_tot_`survey'" 

	foreach var in er_any_`survey' survey_any_`survey' {
		replace `var'=`var'*100
		}
		
		matrix define firstcol_`survey'=J(8,2,.)
	
	*sample counts of survey respondents (column 1)
	di "Count if `survey' survey respondants in ED sample"
	count if `survey'_survey==1 & weight_`survey'!=0
	matrix firstcol_`survey'[`r', 1]= r(N)
	matrix firstcol_`survey'[`r'+2, 1]= r(N)
	matrix firstcol_`survey'[`r'+4, 1]= r(N)
	matrix firstcol_`survey'[`r'+6, 1]= r(N)
	
	*first stage results (column 2)
	di "reg ohp_all_ever_`survey' treatment `controls' if `survey'_survey==1 [pw=weight_`survey'], cluster(household_id)"
	reg ohp_all_ever_`survey' treatment `controls' if `survey'_survey==1 [pw=weight_`survey'], cluster(household_id)
	local fs = round(_b[treatment], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		
	matrix firstcol_`survey'[`r',2] = `fs'
	matrix firstcol_`survey'[`r'+2,2] = `fs'
	matrix firstcol_`survey'[`r'+4,2] = `fs'
	matrix firstcol_`survey'[`r'+6,2] = `fs'
	
	include Programs/ed_tables_analysis.do
	
	matrix list TableTA5_`survey'__`date'

		} // End of survey loop

		
	matrix define firstcol= firstcol_mail \ firstcol_inp
	matrix list firstcol
	
	*Make table
	matrix define Table5= TableTA5_mail__`date' \ TableTA5_inp__`date'
	local rownames "er_any_mail" "" "er_num_mail" "" "survey_any_mail" "" "tr_survey_tot_mail" "" "er_any_inp" "" "er_num_inp" "" "survey_any_inp" "" "tr_survey_tot_inp" "" 
	matrix define Table5= firstcol, Table5
	matrix rownames Table5="`rownames'"
	
			} // End of T5 Loop		
			
************************
*Table A1: ED vs. Full Sample
**************************
	if `TA1_switch'==1 {

	local condition_A1_1 ""
	local condition_A1_2 "& $sample==1"

	foreach part in A1_1 A1_2 {
	
		clear
		use Data/balance_data.dta
	
		local table "`part'_`date'"
		local shape "long"
		local weight "`weight_`date''"
		
		local sheet "A1"
		
		local list "birthyear_list female_list english_list self_list first_day_list have_phone_list pobox_list"

	include Programs/ed_table_means.do 

}

matrix define TableA1_`date'= TableA1_1_`date', TableA1_2_`date'
matrix list TableA1_`date'
		
	} // End of Table A1 Loop


***********************************
*Table A7: First Stage Estimates
*************************************
	if `TA7_switch'==1 {
	
	clear
	use Data/data_for_analysis
	
		local table "A7"
		*dis `table'
		local list "ohp_all_ever_`date' ohp_std_ever_`date' ohp_all_mo_`date' ohp_all_end_`date'"
		local controls "$adjustments"
		local sample "$sample"
		
		local source "`date'"
		local weight "`weight_`date''"

	include Programs/ed_first_stage.do

	matrix colnames TableA7_`date'= "cmean" "edsample"
				
			} // End of Table A8 Loop



***********************
*Table A8: Summary of analytic variables (control sample only)
***************************
	if `TA8_switch'==1 {

	clear
	use Data/data_for_analysis	
	set linesize 225
	
	/*Turn these variables into indicators for people with ANY non-zero probability
	in the total margin variables of the NYU Algorithm */
	
	foreach var in ed_ne ed_epct ed_edcnpa ed_edcnnp ed_unclas {
		gen `var'_`date'= (!missing(tr_`var'_`date') & tr_`var'_`date'!=0)
		}
	
*truncated version
	local varlist "ed_visit ed_hosp ed_out ed_on ed_off ed_edcnnp ed_edcnpa ed_epct ed_ne ed_unclas"

	local rows: word count `varlist' 
	matrix define TableA8_`date'=J(`rows'+2, 9,.)
	local rownum=1

	foreach var in `varlist' {
		replace `var'_`date'=100*`var'_`date'
		
		sum `var'_`date' if treatment==0
		matrix TableA8_`date'[`rownum',1]=round(r(mean), 0.01)
		*sum tr_`var'_`date' if treatment==0 & `var'_`date'!=0, detail
		*matrix TableA8_`date'[`rownum',2]=round(r(mean), 0.01)
		*matrix TableA8_`date'[`rownum',3]=round(r(sd), 0.01)
		*matrix TableA8_`date'[`rownum',4]=round(r(p50), 0.01)
		*matrix TableA8_`date'[`rownum',5]=round(r(p75), 0.01)
		*matrix TableA8_`date'[`rownum',6]=round(r(p95), 0.01)
		*sum tot_`var'_`date'[aweight=`weight_`date''] if treatment==0 & `var'_`date'==100, detail
		*matrix TableA8_`date'[`rownum',7]=round(2*r(p99), 0.01)
		*count if missing(tr_`var'_`date') & !missing(tot_`var'_`date') & treatment==0 
		*matrix TableA8_`date'[`rownum',8]=r(N)
		*gen nm_tr_`var'_`date'= (!missing(tr_`var'_`date'))
		*sum nm_tr_`var'_`date' if treatment==0
		*matrix TableA8_`date'[`rownum',9]=round(r(mean), 0.01)
		
	local rownum=`rownum'+1	
}
	local rownames= "`varlist'"
	matrix rownames TableA8_`date'=`varlist'
	matrix colnames TableA8_`date'= ///
	"percentsample_withany" "Mean" "SD" "Median" "_75percentile" "_95percentile" "Cutpoint_for_Truncation" "Number_Truncations" "Percent_Non_Missing"

		matrix list TableA8_`date'
		
		*add the matched survey variables (survey_any_mail, survey_tot_mail, tr_survey_tot_mail)
	
	foreach survey in mail inp {
		replace survey_any_`survey'=100*survey_any_`survey'
		
		sum survey_any_`survey' if treatment==0 [aweight=weight_`survey']
		matrix TableA8_`date'[`rownum',1]=round(r(mean), 0.01)
		
		sum tr_survey_tot_`survey' if treatment==0 & survey_any_`survey'!=0, detail
		matrix TableA8_`date'[`rownum',2]=round(r(mean), 0.01)
		matrix TableA8_`date'[`rownum',3]=round(r(sd), 0.01)
		matrix TableA8_`date'[`rownum',4]=round(r(p50), 0.01)
		matrix TableA8_`date'[`rownum',5]=round(r(p75), 0.01)
		matrix TableA8_`date'[`rownum',6]=round(r(p95), 0.01)
		*sum survey_tot_`survey' if treatment==0 & survey_any_`survey'==100, detail
		*matrix TableA8_`date'[`rownum',7]=round(2*r(p99), 0.01)
		*count if missing(tr_survey_tot_`survey') & !missing(survey_tot_`survey') & `survey'_survey==1 
		*matrix TableA8_`date'[`rownum',8]=r(N)
			
		local rownum=`rownum'+1	
		
			}
		
		matrix list TableA8_`date'
	
			} // End of Table A9 Loop
	

				
**************************
*Table A11: Emergency Department Utilization for Selected Conditions
**************************
	if `TA11_switch'==1 {
	clear
	use Data/data_for_analysis
	
	*These variables are not publicly available - censored versions of variables have been made available
	/*foreach var in tr_ed_inj tr_ed_skin tr_ed_abdo tr_ed_back tr_ed_heart tr_ed_head tr_ed_depres tr_ed_psysub tr_ed_chron tr_ed_acsc {
			local `var'_`date'_controls "`var'_09mar2008 `var'_09mar2008_m"
			di "`var'_`date'_controls"
			di ``var'_`date'_controls'
		} */
	
	foreach var in ed_inj ed_skin ed_abdo ed_back ed_heart ed_head ed_depres ed_psysub ed_chron ed_acsc {
			local `var'_`date'_controls "`var'_09mar2008"
		}
		
		local table "T11"
		dis "`table'"
		local condition "& $sample==1"
		local controls "$adjustments"
		local source "`date'"
		
		local list "ed_acsc_`date' ed_chron_`date' ed_inj_`date' ed_skin_`date' ed_abdo_`date' ed_back_`date' ed_heart_`date' ed_head_`date' ed_depres_`date' ed_psysub_`date'" 							
		*local list "ed_acsc_`date' ed_chron_`date' ed_inj_`date' ed_skin_`date' ed_abdo_`date' ed_back_`date' ed_heart_`date' ed_head_`date' ed_depres_`date' ed_psysub_`date' tr_ed_acsc_`date' tr_ed_chron_`date' tr_ed_inj_`date' tr_ed_skin_`date' tr_ed_abdo_`date' tr_ed_back_`date' tr_ed_heart_`date' tr_ed_head_`date' tr_ed_depres_`date' tr_ed_psysub_`date'" 							
		
		local weight "`weight_`date''"
		
		local subsample " " 
		include Programs/ed_tables_analysis.do
		
	} // End of Table A11 Loop	
	
				
**************************
*Table A12: List Charges
**************************	
	if `TA12_switch'==1 {
	
	clear
	use Data/data_for_analysis
	
	*Controls for Total Margin
	local tr_charg_30sep2009_controls "tr_charg_09mar2008 tr_charg_09mar2008_m"
	local tr_ed_charg_30sep2009_controls "tr_ed_charg_09mar2008 tr_ed_charg_09mar2008_m"
	
		local table "A12"
		dis "`table'"
		local condition "& $sample==1"
		local controls "$adjustments"
		
		local list "tr_ed_charg_`date' tr_charg_`date'" 
		
		local source "`date'"
	
		local weight "`weight_`date''"
		
		include Programs/ed_tables_analysis.do
	
				} // End of A11 Switch
	
	
**************************
*Table A13: Emergency Department Utilization by Hospital Type
**************************
	if `TA13_switch'==1 {
	
	clear
	use Data/data_for_analysis.dta
	
	foreach var in ed_hiun ed_loun {
			local `var'_`date'_controls "`var'_09mar2008"
		}
	
	*These variables are not publicly available - censored versions of variables have been made available
	/*foreach var in tr_ed_hiun tr_ed_loun {
		local `var'_`date'_controls "`var'_09mar2008 `var'_09mar2008_m"
		} */
	
		local table "A13"
		dis "`table'"
		local condition "& $sample==1"
		local controls "$adjustments"
		local source "`date'"
		
		local list "ed_hiun_`date' ed_loun_`date'" 		
		*local list "ed_hiun_`date' ed_loun_`date' tr_ed_hiun_`date' tr_ed_loun_`date'" 							
		
		local weight "`weight_`date''"
		
		include Programs/ed_tables_analysis.do
		
		*include Programs/global_sort_test.do // Does not run on publicly available data

				
		} // End of Table A13 Loop
		

*******************************
*Table A14: Heterogeneity of Results
*******************************
	if `TA14_switch'==1 {
	
	clear
	use Data/data_for_analysis.dta
	
	preserve
		
	replace ed_visit_30sep2009=ed_visit_30sep2009*100
	
	local ed_visit_30sep2009_controls "ed_visit_09mar2008"
	*These variables are not publicly available - censored versions of variables have been made available
	*local tr_ed_visit_30sep2009_controls "tr_ed_visit_09mar2008 tr_ed_visit_09mar2008_m"
	
	local n=0

	local n=`n'+1
		local table "TA14a"
		dis "`table'"
		local condition "& $sample==1"
		local controls "$adjustments"
		local source "`date'"
		
		local sample "$sample"
		
		local list "ed_visit" 
		*local list "ed_visit tr_ed_visit" 
		
		local fs_var "ohp_all_ever_`date'"
	
		local list_names ""N" "FS" "c_mean1" "LATE1" "p1"
		*local list_names ""N" "FS" "c_mean1" "LATE1" "p1" "c_mean2" "LATE2" "p2""
		local subpops "all male old white morehs smoke first_day dx_prelot"
		local subpop_namesB ""all" "se" " " "male" "se" "female" "se" "p" "old" "se" "young" "se" "p" "white" "se" "nonwhite" "se" "p" "morehs" "se" "lesshs" "se" "p" "smoke" "se" "nosmoke" "se" "p" "firstday" "se" "notfirstday" "se" "p" "prelotterydxall" "se" "noprelotterydxall" "se" "p""
		
		local weight "`weight_`date''"
		local list_binary "$binary"
		
			include Programs/ed_table_a14.do 
			
		restore
	
			matrix define TableA14__30sep2009== TableTA14a_30sep2009 
				
		local subsample " " 
				
				} // End of TableA13 Heterogeneity Table Loop
	
	
*************************
*Table A15: Sensitivity of results to choice of covariates
*************************
	if `TA15_switch'==1 {
	
	*************************
	*First sensitivity check, Table 15
	*************************
	
	*Baseline
	
	*Control means
	clear 
	use Data/data_for_analysis
	local table "A15"
	local source "`date'"
	local condition "& $sample==1"
	local controls "$adjustments"
	
	*Set local list (variables split over multiple locals)
	local T2_vars "ed_visit_`date'"
	*local T2_vars2 "tr_ed_visit_`date'"
	local T3_vars "ed_hosp_`date' ed_out_`date' ed_on_`date' ed_off_`date'"
	*local T3_vars2 "tr_ed_hosp_`date' tr_ed_out_`date' tr_ed_on_`date' tr_ed_off_`date'"
	local T4_vars "tr_ed_edcnnp_`date' tr_ed_edcnpa_`date' tr_ed_epct_`date' tr_ed_ne_`date' tr_ed_unclas_`date'"
	
	local list "`T2_vars' `T3_vars' `T2_vars2' `T3_vars2' `T4_vars'" 
	
	
		foreach var in ed_visit ed_hosp ed_out ed_on ed_off {
			local `var'_`date'_controls "`var'_09mar2008"
			*local tr_`var'_`date'_controls "tr_`var'_09mar2008 tr_`var'_09mar2008_m"
				} // End of controls loop
				
		foreach var in ed_edcnnp ed_edcnpa ed_epct ed_ne ed_unclas	{ 
			local tr_`var'_`date'_controls "tr_`var'_09mar2008 tr_`var'_09mar2008_m"
				} // End of controls loop
		
		local weight "`weight_`date''"
	
	local sensitivity=1
	include Programs/ed_tables_analysis.do
	
	matrix define baselinea= `results'
	matrix list baselinea
	
	*no preperiod controls 
	clear
	use Data/data_for_analysis
	
		local table "A15anc"
		local lab "no_con"

		local controls "$adjustments"
		
		foreach var of local list {
		local `var'_controls " "
		}
		
		local weight "`weight_`date''"
		
	include Programs/ed_tables_analysis.do

	*added lottery list vars 
	clear
	use Data/data_for_analysis.dta
		
		local table "A15alistv"
		local lab "list"
		local condition "& $sample==1"
		local controls "$adjustments female_list birthyear_list english_list self_list first_day_list have_phone_list pobox_list"
		local source "`date'"
		
		foreach var in ed_visit ed_hosp ed_out ed_on ed_off {
			local `var'_`date'_controls "`var'_09mar2008"
			local tr_`var'_`date'_controls "tr_`var'_09mar2008 tr_`var'_09mar2008_m"
				} // End of controls loop
				
		foreach var in ed_edcnnp ed_edcnpa ed_epct ed_ne ed_unclas {	
			local tr_`var'_`date'_controls "tr_`var'_09mar2008 tr_`var'_09mar2008_m"
				} // End of controls loop

	include Programs/ed_tables_analysis.do

	*Make table A15
		matrix define TableA15=baselinea, TableA15anc__30sep2009, TableA15alistv__30sep2009
		matrix list TableA15
				
			} // End of sensitivity switch (Table A15)

		local sensitivity=0	
******************
*Table A16: Sensitivity of Results to Functional Form
******************
	if `TA16_switch'==1 {

		clear
		use Data/data_for_analysis.dta
		
		local table "A16"
		
		local list_vars "ed_visit ed_hosp ed_out ed_on ed_off"
		
		foreach var in `list_vars' {
			local `var'_`date'_controls "`var'_09mar2008"	
		}
		
		local condition "& $sample==1"
			local controls "$adjustments"
			local source "`date'"
			
		local weight_30sep2009 "1"

		include Programs/ed_tables_logit.do
		
		/*Total margin variables are not publicly available - censored variables are available in their place
		clear
		use Data/data_for_analysis.dta
		
		local table "proport"
		
		local visit_vars "tr_ed_visit tr_ed_hosp tr_ed_out"
		local timing_vars "tr_ed_on tr_ed_off"
		local visit_type "tr_ed_edcnnp tr_ed_edcnpa tr_ed_epct tr_ed_ne tr_ed_unclas"
		local list_vars "`visit_vars' `timing_vars' `visit_type' `severity_vars'"
		
		foreach var in `visit_vars' `timing_vars' `visit_type' {
			local `var'_`date'_controls "`var'_09mar2008 `var'_09mar2008_m"	
				}
		
		local condition "& $sample==1"
			local controls "$adjustments"
			local source "`date'"
			
		local weight_30sep2009 "1"

		include Programs/ed_proportional_check.do */
		
		
			} //End of Table A16 Logit Loop



***************************
*Table A17: Observational Estimates
****************************
	if `TA17_switch'==1 {
	
	local Table "A17"
	local source "`date'"
	local condition "& $sample==1"
	local controls "$adjustments"
	local weight "`weight_`date''"
	local sheet "observational"
	
	foreach var in ed_visit ed_hosp ed_out ed_off ed_on ed_acsc {
			local `var'_`date'_controls "`var'_09mar2008 `var'_09mar2008_m"
		}
	
	*Variables are not publicly available - censored versions are available
	*foreach var in tr_ed_visit tr_ed_hosp tr_ed_out tr_ed_on tr_ed_off tr_ed_edcnnp tr_ed_edcnpa tr_ed_preven tr_ed_epct tr_ed_ne tr_ed_unclas {
	*		local `var'_`date'_controls "`var'_09mar2008 `var'_09mar2008_m"
	*	}
		
	*local list "ed_visit_`date' ed_hosp_`date' ed_out_`date' ed_on_`date' ed_off_`date' tr_ed_visit_`date' tr_ed_hosp_`date' tr_ed_out_`date' tr_ed_on_`date' tr_ed_off_`date' tr_ed_edcnnp_`date' tr_ed_edcnpa_`date' tr_ed_epct_`date' tr_ed_ne_`date' tr_ed_unclas_`date'"
	local list "ed_visit_`date' ed_hosp_`date' ed_out_`date' ed_on_`date' ed_off_`date'"
	
	local var_interest_1 "treatment"
	local condition1 "& $sample==1"
	local count_condition1 "& $sample==1"
	local insured_condition1 "& $sample==1 & ohp_all_ever_30sep2009==1"
	
	local var_interest_2 "ohp_all_ever_30sep2009"
	local condition2 "& $sample==1"
	local count_condition2 "& $sample==1"
	local insured_condition2 "& $sample==1 & ohp_all_ever_30sep2009==1" 

	local var_interest_3 "ohp_all_ever_30sep2009"
	local condition3 "& $sample==1 & treatment==0"
	local count_condition3 "& $sample==1 & treatment==0"
	local insured_condition3 "& $sample==1 & ohp_all_ever_30sep2009==1 & treatment==0"
			
	local var_interest_4 "ohp_std_ever_30sep2009"
	local condition4 "& $sample==1 & treatment==1"
	local count_condition4 "& $sample==1 & ohp_std_ever_30sep2009!=. & treatment==1"
	local insured_condition4 "& $sample==1 & ohp_std_ever_30sep2009==1 & treatment==1"

	forval n=1/4 {
	
	clear 
	use Data/data_for_analysis.dta
	
		local table "TA17_panel`n'"
		include Programs/ed_tables_observational.do
		
		} // end of forvalues loop
		
		* Set up matrix
	matrix TableA17_top = J(2,4,.)
	matrix rownames TableA17_top = "SampleSize" "%insured" 
	matrix colnames TableA17_top = "Random" "AnyvsNo" "AnyvsNoCONTROLS" "OHPStdvsNoTREATMENT"

	clear
		use Data/data_for_analysis.dta

	forv n=1/4 {
		count if 1==1 `count_condition`n''
		local samplesize`n'=r(N)
		count if 1==1 `insured_condition`n''
		local pctinsured`n'=r(N)/`samplesize`n''
		matrix TableA17_top[1,`n'] = `samplesize`n''
		matrix TableA17_top[2,`n'] = round(`pctinsured`n''*100)
	}
		
		matrix TableA17_bottom=[TableTA17_panel1[1...,2...], TableTA17_panel2[1...,2...], TableTA17_panel3[1...,2...], TableTA17_panel4[1...,2...]]
		matrix TableA17=TableA17_top\TableA17_bottom 
	
		noi: matrix list TableA17
			
		} // End of Table A17 Observational Estimates Loop
	
*
	} // End of Appendix Table Date Loop


**************************************
* PRINTING RESULTS TO A SEPARATE FILE
**************************************
log close
log using all_tables.log, text replace
set linesize 255

qui {

noi: di _newline
noi: di "***********MAIN TABLES***********"
noi: di _newline

foreach date in 30sep2009 { // 
	
	if `T1_switch'==1 {
		noi: di _newline
		noi: matrix list TableT1_`date', title(Table 1: Balance (ED vs Full Sample))
		}
	if `T2_switch'==1 {
		noi: di _newline
		noi: matrix list TableT2_`date', title(Tables 2 and A2: Emergency Department Utilization)
		} 
	if `T3_switch'==1 {
		noi: di _newline
		noi: matrix list TableT3__`date', title(Tables 3 and A3: ED Use by Admission and Urgency)
		}
	if `T4_switch'==1 {
		noi: di _newline
		noi: matrix list TableT4__`date', title(Tables 4 and A4: ED Usage by Timing/Urgency)
		} 
	if `T5_switch'==1 {
		noi: di _newline
		noi: matrix list Table5, title(Tables 5 and A5: Comparing Administrative Data and Survey Responses)
		} 

	if `TA1_switch'==1 {
		noi: di _newline
		noi: matrix list TableA1_`date'
		}
	if `TA7_switch'==1 {
		noi: di _newline
		noi: matrix list TableA7_`date'
		}
	if `TA8_switch'==1 {
		noi: di _newline
		noi: matrix list TableA8_`date'
		}
	if `TA11_switch'==1 {
		noi: di _newline
		noi: matrix list TableT11__`date'
		}
	if `TA12_switch'==1 {
		noi: di _newline
		noi: matrix list TableA12__`date', title(Table A12: List Charges)
		}
	if `TA13_switch'==1 {
		noi: di _newline
		noi: matrix list TableA13__`date'
		}
	if `TA14_switch'==1 {
		noi: di _newline
		noi: matrix list TableA14__30sep2009
		}
	if `TA15_switch'==1 {
		noi: di _newline
		noi: matrix list TableA15
		}
	if `TA16_switch'==1 {
		noi: di _newline
		noi: matrix list logit_A16
		*noi: matrix list proport
		}
	if `TA17_switch'==1 {
		noi: di _newline
		noi: matrix list TableA17
		}
}
}

log close
*


