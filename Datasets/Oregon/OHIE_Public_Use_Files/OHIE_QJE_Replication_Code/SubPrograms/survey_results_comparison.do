

foreach date in 0m 6m 12m {
	local survey_usetot_list_`date' "rx_num_mod_`date' doc_num_mod_`date' er_num_mod_`date' hosp_num_mod_`date'"

	local survey_useext_list_`date' "rx_any_`date' doc_any_`date' er_any_`date' hosp_any_`date'"

	local survey_finance_list_`date' "cost_any_oop_`date' cost_any_owe_`date' cost_borrow_`date' cost_refused_`date'"

	local survey_access_list_`date' "usual_clinic_`date' needmet_med_`date' needmet_rx_`date' not_er_noner_`date'"

	local survey_health_list_`date' "health_genflip_bin_`date' health_notpoor_`date' health_chgflip_bin_`date' notbaddays_tot_`date' notbaddays_phys_`date' notbaddays_ment_`date'"
	
	}
		
****************
* Data
****************

use Data/data_for_analysis.dta, clear


*****************
* Table: First Stage 
*****************
foreach x in 0m 6m {
	matrix define FS_`x' = J(4,2,.)
	
	reg ohp_all_ever_survey`x' if treatment==0 & sample_`x'_resp == 1 [pw=weight_`x']
	local fs_1_1=round(_b[_cons],0.001)
		if substr("`fs_1_1'",1,3)==".00" | `fs_1_1'==0 {
		local fs_1_1=_b[_cons]
		}
	matrix FS_`x'[1,1] = `fs_1_1'
	
	reg ohp_all_ever_survey`x' treatment eee* if sample_`x'_resp == 1 [pw=weight_`x'], cluster(household_id)
	local fs_1_2=round(_b[treatment],0.001)
	local fs_2_2=round(_se[treatment],0.001)
		if substr("`fs_1_2'",1,3)==".00" | `fs_1_2'==0 {
		local fs_1_2=_b[treatment]
		}
		if substr("`fs_2_2'",1,3)==".00" | `fs_2_2'==0 {
		local fs_2_2=_se[treatment]
		}
	matrix FS_`x'[1,2] = `fs_1_2'
	matrix FS_`x'[2,2] = `fs_2_2'
	
	reg ohp_all_mo_survey`x' if treatment==0 & sample_`x'_resp == 1 [pw=weight_`x']
	local fs_3_1 = round(_b[_cons],0.001)
		if substr("`fs_3_1'",1,3)==".00" | `fs_3_1'==0 {
		local fs_3_1=_b[_cons]
		}	
	matrix FS_`x'[3,1] = `fs_3_1'

	reg ohp_all_mo_survey`x' treatment eee* if sample_`x'_resp == 1 [pw=weight_`x'], cluster(household_id)
	local fs_3_2 = round(_b[treatment],0.001)
	local fs_4_2 = round(_se[treatment],0.001)
		if substr("`fs_3_2'",1,3)==".00" | `fs_3_2'==0 {
		local fs_3_2=_b[treatment]
		}
		if substr("`fs_4_2'",1,3)==".00" | `fs_4_2'==0 {
		local fs_4_2=_se[treatment]
		}	
	matrix FS_`x'[3,2] = `fs_3_2'
	matrix FS_`x'[4,2] = `fs_4_2'

   matrix list FS_`x'
}
*****************
* Table: Standard Treatment Effect
*****************

matrix define STE = J(20,6,.) 
matrix colnames STE = "RF-0m" "RF-6m" "RF-12m" "p-diff_0_6" "p-diff_6_12" "p-diff_0_12"

local temp ""
local count = 1

foreach domain in useext usetot finance health access {

local outcomes "`survey_`domain'_list_0m' `survey_`domain'_list_6m' `survey_`domain'_list_12m'"
local numvar : word count `outcomes'

local numvar_0m : word count `survey_`domain'_list_0m'
local numvar_6m : word count `survey_`domain'_list_6m'
local numvar_12m : word count `survey_`domain'_list_12m'

local start_0m = 1
local stop_0m = `numvar_0m'

local start_6m = `numvar_0m'+1
local stop_6m = `numvar_0m'+`numvar_6m'

local start_12m = `numvar_0m'+`numvar_6m'+1
local stop_12m = `numvar'

clear
use Data/data_for_analysis.dta

* Calculating and saving standard deviations 
local i = 0
foreach var of local outcomes {
	local ++i

	foreach word of local survey_`domain'_list_0m{
		if "`var'"=="`word'"{
			local controls "eee*"
			local condition "& sample_0m_resp==1"
			local weight "weight_0m"
			}
		}
		
	foreach word of local survey_`domain'_list_6m{
		if "`var'"=="`word'"{
			local controls "eee*"
			local condition "& sample_6m_resp==1"
			local weight "weight_6m"
			}
		}	
	
	foreach word of local survey_`domain'_list_12m{
		if "`var'"=="`word'"{
			local controls "ddd*"
			local condition "& sample_12m_resp==1"
			local weight "weight_12m"
			}
		}

	reg `var' if treatment==0 `condition' [pw=`weight']
	local `i'_sd = round(e(rmse)*1000)/1000
	}

* Expanding data for stacked regressions
expand `numvar'
bys person_id: gen order = _n
		
* Creating variables for stacked regressions

gen outcome = .				
gen stack_weight = .

local i = 0
		
foreach var of local outcomes {
	
	local ++i

	foreach word of local survey_`domain'_list_0m{
		if "`var'"=="`word'"{
			local controls "eee*"
			local condition "& sample_0m_resp==1"
			local weight "weight_0m"
			}
		}
		
	foreach word of local survey_`domain'_list_6m{
		if "`var'"=="`word'"{
			local controls "eee*"
			local condition "& sample_6m_resp==1"
			local weight "weight_6m"
			}
		}		

	foreach word of local survey_`domain'_list_12m{
		if "`var'"=="`word'"{
			local controls "ddd*"
			local condition "& sample_12m_resp==1"
			local weight "weight_12m"
			}
		}
	
	di "`i' `var'"
	
	replace outcome = `var' if (order == `i') `condition'
			
	sum `weight' if (order == `i') `condition'
	replace stack_weight = `weight'/r(mean) if (order == `i') `condition'
						
	gen treatment_X`i' = treatment*(order == `i')
				
	foreach control of varlist constant `controls'{
		gen X`i'_`control' = `control' *(order == `i')
		replace X`i'_`control' = 0 if X`i'_`control' == .
		}

	} // end of outcome variable loop


*Running stacked regression and calculating STE
 
reg outcome treatment_X* X* [pw=stack_weight], cluster(household_id) noconstant

foreach date in 0m 6m 12m { 
	local lincom_string_`date' ""

	forval i = `start_`date''/`stop_`date''{

		di "`i'"
		
		if "`lincom_string_`date''" == "" local lincom_string_`date'  "1/`numvar_`date''*_b[treatment_X`i']/``i'_sd'"
		else local lincom_string_`date'  "`lincom_string_`date'' + 1/`numvar_`date''*_b[treatment_X`i']/``i'_sd'"
		}
		
	di "`lincom_string_`date''"
	lincom "`lincom_string_`date''"

	local itt_effect_`date' = round(r(estimate),0.001)
	local itt_se_`date' = round(r(se),0.001)
	local itt_p_`date' = round(2*ttail(r(df), abs(r(estimate)/r(se))),0.001)
	
	if `itt_effect_`date''==0 | substr("`itt_effect_`date''",1,3)==".00" {
		local itt_effect_`date' = r(estimate)
		}
	
	if `itt_se_`date''==0 | substr("`itt_se_`date''",1,3)==".00" {
		local itt_se_`date' = r(se)
	}
	
	if `itt_p_`date''==0 {
		local itt_p_`date' = 2*ttail(r(df), abs(r(estimate)/r(se)))
		}

	}
	
local lincom_string "`lincom_string_6m' - (`lincom_string_0m')"
lincom "`lincom_string'"
local itt_p_diff_0_6 = round(2*ttail(r(df), abs(r(estimate)/r(se))),0.001)
if `itt_p_diff_0_6'==0 {
		local itt_p_diff_0_6 = 2*ttail(r(df), abs(r(estimate)/r(se)))
		}	

local lincom_string "`lincom_string_12m' - (`lincom_string_6m')"
lincom "`lincom_string'"
local itt_p_diff_6_12 = round(2*ttail(r(df), abs(r(estimate)/r(se))),0.001)
if `itt_p_diff_6_12'==0 {
		local itt_p_diff_6_12 = 2*ttail(r(df), abs(r(estimate)/r(se)))
		}
		
local lincom_string "`lincom_string_12m' - (`lincom_string_0m')"
lincom "`lincom_string'"
local itt_p_diff_0_12 = round(2*ttail(r(df), abs(r(estimate)/r(se))),0.001)
if `itt_p_diff_0_12'==0 {
		local itt_p_diff_0_12 = 2*ttail(r(df), abs(r(estimate)/r(se)))
		}		

matrix STE[`count',1] = `itt_effect_0m'
matrix STE[`count'+1,1] = `itt_se_0m'
matrix STE[`count'+2,1] = `itt_p_0m'

matrix STE[`count',2] = `itt_effect_6m'
matrix STE[`count'+1,2] = `itt_se_6m'
matrix STE[`count'+2,2] = `itt_p_6m'

matrix STE[`count',3] = `itt_effect_12m'
matrix STE[`count'+1,3] = `itt_se_12m'
matrix STE[`count'+2,3] = `itt_p_12m'

matrix STE[`count',4] = `itt_p_diff_0_6'
matrix STE[`count',5] = `itt_p_diff_6_12'
matrix STE[`count',6] = `itt_p_diff_0_12'

	local count = `count'+4
	local temp "`temp' `domain' se p skip"
	}

matrix rownames STE = `temp'
matrix list STE 

