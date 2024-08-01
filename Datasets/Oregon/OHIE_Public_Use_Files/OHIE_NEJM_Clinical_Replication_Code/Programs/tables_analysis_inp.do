

* ***********************************************
* Analysis comparing treatment and controls
* ***********************************************

dis "`condition_survey'"
***************************
*Set up matrix for results
keep household_id person_id treatment constant numhh_list ohp_all_ever_survey ohp_all_ever_admin ///
`list_survey' `list_admin' `controls_survey' `controls_admin' `weight_survey' `weight_admin' `add_vars' 

 * inflate 0/1 variables by 100
 local inflate_list "`list_binary' phq_prob"


local list "`list_admin' `list_survey'"
dis "Running list `list'"
local source ""

local results = "Table`table'"

local numvar : word count `list'

di "`numvar'"

local rows = 2*(`numvar')+6


dis "matrix define `results' = J(`rows', 4, .)"
matrix define `results' = J(`rows', 4, .)
matrix colnames `results' = "c-mean" "RF" "2SLS" "p-vals" 

matrix define pvalues = J(`numvar',5,.)
matrix colnames pvalues = "var_index" "actual_pvalue" "sim_pvalue" "counter" "adj-p"

if `spend_est' == 1 {
matrix define spend_est_`table' = J(2,4,.)
matrix rownames spend_est_`table' = "spending estimate" "se"
matrix colnames spend_est_`table' = "c-mean" "RF" "2SLS" "p-vals" 
			}
	
if $ALT==1 {
matrix `table'_alt = J(`numvar',9,.)
matrix colnames `table'_alt = "c-mean" "se" "RF" "l95" "h95" "2SLS" "l95" "h95" "pvalue"

}
	
*******************************
* Sample size for table notes

count if 1==1 `condition_survey'
local Table`table'_N_survey = r(N)

count if 1==1 `condition_admin'
local Table`table'_N_admin = r(N)


***********************************
* Control means and comparison of treaments and controls 
			
local temp  ""
local count = 0
local i = 0


if $ALT==1 {
local count_alt = 0
local temp_alt ""
}


foreach datatype in admin survey inperson{

	local source "`datatype'"
	local controls "`controls_`datatype''"
	local condition "`condition_`datatype''"
	local weight "`weight_`datatype''"
	
	foreach var of local list_`datatype' {
		
		local ++i
		di "`i' `var' `source'"
		dis "reg `var' if treatment==0 `condition' [pw=`weight']"
		
		* Binary variables (and probabilities) gets inflated by 100
		local inflate_`i'=0
		 local inflate_var: list var & inflate_list
		 if "`var'"=="`inflate_var'" {
			di  "`i' `var' gets inflated by 100"
			sum `var'
		        replace `var'=100*`var'
			sum `var'
			local inflate_`i'=1
		 }
		 


		*Binary variables don't need SDs displayed
		local binary_`i'=0
		local binary_var: list var & list_binary
		if "`var'"=="`binary_var'" {
			di "`i' `var' does not need SD"
			local binary_`i'=1
		}
		
		*Control means
		dis "reg `var' if treatment==0 `condition' [pw=`weight']"
		reg `var' if treatment==0 `condition' [pw=`weight']
		
		local `i'_mean = round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
		local `i'_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-1)))
		
		local `i'_mean_complete=_b[_cons]
		
		
		*matrix `results'[`count'+1, 1] = ``i'_mean'
		
		if `binary_`i''!=1 | "`var'" == "cvd_risk_point" {
			matrix `results'[`count'+2, 1] = ``i'_sd'
		}
		
		if $ALT==1 {
		matrix `table'_alt[`count_alt'+1, 1] = ``i'_mean'
			if `binary_`i''!=1 | "`var'" == "cvd_risk_point" {
			matrix `table'_alt[`count_alt'+1, 2] = ``i'_sd'
			}
		
		} 
		
		*ITT comparison
		reg `var' treatment `controls' ``var'_controls' if 1==1 `condition' [pw=`weight'], cluster(household_id)
		matrix postest=r(table)
		*matrix list postest
		
		local `i'_itt_beta = round(_b[treatment],10^(min(-2, int(log10(abs(_b[treatment])))-2)))
		local `i'_itt_se = round(_se[treatment], 10^(min(-2, int(log10(abs(_se[treatment])))-2)))
		 
		local `i'_itt_p = round(postest[4,1], 0.01)
		
		local `i'_itt95_l=round(postest[5,1],  10^(min(-2, int(log10(abs(postest[5,1])))-2)))
		local `i'_itt95_h=round(postest[6,1],  10^(min(-2, int(log10(abs(postest[6,1])))-2)))
		
		if postest[4,1]<0.1 {
			local `i'_itt_p=round(postest[4,1], 0.001)
			
		}
		
		if postest[4,1]<0.001 {
			local `i'_itt_p=postest[4,1]
			
		}
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		matrix `results'[`count'+1, 2] = ``i'_itt_beta'
		matrix `results'[`count'+2, 2] = ``i'_itt_se'
		

		matrix pvalues[`i',1] = `i'
		matrix pvalues[`i',2] = ``i'_itt_p'
		
		if $ALT==1 {
		matrix `table'_alt[`count_alt'+1, 3] = ``i'_itt_beta'
		matrix `table'_alt[`count_alt'+1, 4] = ``i'_itt95_l'
		matrix `table'_alt[`count_alt'+1, 5] = ``i'_itt95_h'
		
		}
		
		*ToT comparison
		di "ivregress 2sls `var' (ohp_all_ever_`source'= treatment) `controls' ``var'_controls' if 1==1 `condition' [pw = `weight'], cluster(household_id)"  
		ivregress 2sls `var' (ohp_all_ever_`source'= treatment) `controls' ``var'_controls' ///
		  if 1==1 `condition' [pw = `weight'], cluster(household_id)  
		matrix postest=r(table)
		*matrix list postest
		  	
		local `i'_tot_beta = round(_b[ohp_all_ever_`source'],10^(min(-2, int(log10(abs(_b[ohp_all_ever_`source'])))-2)))
		local `i'_tot_se = round(_se[ohp_all_ever_`source'], 10^(min(-2, int(log10(abs(_se[ohp_all_ever_`source'])))-2)))
		
		
		local `i'_tot_p = round(postest[4,1], 0.01)
		
		local `i'_tot95_l=round(postest[5,1],  10^(min(-2, int(log10(abs(postest[5,1])))-2)))
		local `i'_tot95_h=round(postest[6,1],  10^(min(-2, int(log10(abs(postest[6,1])))-2)))
		
		
		if postest[4,1]<0.1 {
			local `i'_tot_p=round(postest[4,1], 0.001)
			
		}
		
		if postest[4,1]<0.001 {
			local `i'_tot_p=postest[4,1]
			
		}
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		matrix `results'[`count'+1, 3] = ``i'_tot_beta'
		matrix `results'[`count'+2, 3] = ``i'_tot_se'
		matrix `results'[`count'+1, 4] = ``i'_tot_p'
		
		if $ALT==1 {
		matrix `table'_alt[`count_alt'+1, 6] = ``i'_tot_beta'
		matrix `table'_alt[`count_alt'+1, 7] = ``i'_tot95_l'
		matrix `table'_alt[`count_alt'+1, 8] = ``i'_tot95_h'
		matrix `table'_alt[`count_alt'+1, 9] = ``i'_tot_p'
                }
		
		local count_alt = `count_alt'+1
		if "`var'"!="" {
		local temp_alt "`temp_alt' `var'"
		}

		local count = `count' + 2
		di `count'
		
		local temp "`temp' `var' se"

		
		} //end of var loop
	
	local count = `count'+2
	local temp "`temp' ste se"
	
	if $ALT==1 {
	local count_alt = 0 
	}
	
	} //end of datatype loop

local source ""

* calculate control mean for spending estimates

if `spend_est' == 1 {
	
		if `inp_12m_spend'==1 {
		local spend_mean = 1*(312*`1_mean_complete'+150*`2_mean_complete'+435*`3_mean_complete'+7523*`4_mean_complete')
		}
		
		else {
		local spend_mean = 312*`1_mean_complete'+150*(`2_mean_complete'+`3_mean_complete')+435*`4_mean_complete'+7523*`5_mean_complete'
		}
		
}

*****************************************************
*Multiple inference adjustment

if "`mult_inf'" != "no"{
	
	/* *** Multiple inference adjustment *** */
	* 1 - sort p-values
	* 2 - randomize treatment (and draws) hh_size keeping the empirical distribution
	* 3 - repeat the estimation procedure above
	* 4 - compute the stepped down p-value 
	* 5 - keep counter with the numbers --

// Assign an index person for each household for simulating treatment
bys household_id (person_id): gen hh_id = (_n==1)

// Get treatment probabilities by hh size
forvalues num = 1/3{
	sum treatment if  hh_id==1 & numhh_list == `num'
	local pr_treat_`num' = r(mean)
	}	

// Set up pvalue counter matrix
matrix list pvalues
mata: st_matrix("pvalues_counter_sorted", sort(st_matrix("pvalues"),2))
matrix list pvalues_counter_sorted 
forvalues v = 1/`numvar' {
	matrix pvalues_counter_sorted[`v',4] = 0
	}

forvalues iter = 1/$iterations{
	if $iterations < 10{
		di "Running iteration: `iter' "
		}
	else{
		if mod(`iter',100) == 0 {
			di "Running iteration: `iter' "
			}
		}

	quietly{
	// Randomly assign treatment by household size
	gen temp_treatment = .
	gen tmpuni = uniform() if hh_id == 1
	forvalues num = 1/3{
		replace temp_treatment = (tmpuni <= `pr_treat_`num'') if hh_id ==1 & numhh_list == `num'  
		}
	tab temp_treatment numhh_list if hh_id==1
	bys household_id: egen sim_treat = max(temp_treatment) /* assign same treatment to entire household */
	tab numhh_list sim_treat, row col
	// Run estimation with simulated treatment
	local i = 0
	foreach var of local list{
		local ++i
		
		foreach word of local list_survey{
		if "`var'"=="`word'"{
			local source "survey"
			local controls "`controls_survey'"
			local condition "`condition_survey'"
			local weight "`weight_survey'"
			}
		}
	
		foreach word of local list_admin{
		if "`var'"=="`word'"{
			local source "admin"
			local controls "`controls_admin'"
			local condition "`condition_admin'"
			local weight "`weight_admin'"
			}
		}

		matrix pvalues[`i',3] = .
		qui: reg `var' sim_treat `controls' ``var'_controls' if 1==1 `condition' [pw=`weight'], cluster(household_id)
		
		matrix postest=r(table)
		matrix list postest
		
		local `i'_sim_p = postest[4,1]
		matrix pvalues[`i',3] = ``i'_sim_p'
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		} // end of var loop

	// Sort by original p-values

	matrix list pvalues
	mata: st_matrix("pvalues_sorted", sort(st_matrix("pvalues"),2))
	matrix list pvalues_sorted
	
	// Enforce monotonicity on simulated p-values

	forvalues v = 2/`numvar' {
		matrix pvalues_sorted[`numvar' - `v' + 1, 3] = min(pvalues_sorted[`numvar' - `v' + 1, 3], pvalues_sorted[`numvar' - `v' + 2, 3])
		}
	matrix list pvalues_sorted

	// Compare actual p-values to simulated p-values and keep counter of number smaller
	
	forvalues v = 1/`numvar' {
		if (pvalues_sorted[`v',3] < pvalues_sorted[`v',2]) {
			matrix pvalues_counter_sorted[`v',4] = pvalues_counter_sorted[`v',4] + 1
			}
		}
	matrix list pvalues_counter_sorted

	// Cleaning up for next loop
	drop tmpuni temp_treatment sim_treat

	} // end of quiet loop
	
	} // end of iter loop

// Calculate adjusted p-value

matrix list pvalues_counter_sorted
forvalues v = 1/`numvar' {
	matrix pvalues_counter_sorted[`v',5] = pvalues_counter_sorted[`v',4] / $iterations
	}
matrix list pvalues_counter_sorted

// Enforce monotonicity one more time

forvalues v = 2/`numvar' {
	matrix pvalues_counter_sorted[`numvar' - `v' + 1, 5] = min(pvalues_counter_sorted[`numvar' - `v' + 1, 5], pvalues_counter_sorted[`numvar' - `v' + 2, 5])
	}
matrix list pvalues_counter_sorted

// Restore original variable order

mata: st_matrix("pvalues_adjusted", sort(st_matrix("pvalues_counter_sorted"),1))
matrix list pvalues_adjusted


// Adding adjusted p-values to the table matrix
local i = 0
local count = 0
foreach datatype in admin survey{
	foreach var of local list_`datatype' {

		local ++i
		
		matrix `results'[`count'+2, 4] =round(pvalues_adjusted[`i',5]*1000)/1000
		
		if round(pvalues_adjusted[`i',5],0.0001)==0{
		matrix `results'[`count'+2, 4] =pvalues_adjusted[`i',5]
		}
		local count = `count'+2

		} // end of var loop

	local count = `count'+2

	} // end of datatype loop

} // end of no multiple inference 

******************************************************
*Standardized treatment effects

if `ste'==1 {

local list_all : list list - exclude_from_ste

local numvar_admin : word count `list_admin'
local numvar_survey : word count `list_survey'
local numvar_all : word count `list_all'

local start_admin = 1
local stop_admin = `numvar_admin'
local ste_row_admin = `numvar_admin'*2

local start_survey = `numvar_admin'+1
local stop_survey = `numvar'
local ste_row_survey = `numvar'*2+2

local start_all = 1
local stop_all = `numvar'
local ste_row_all = `numvar'*2+4


compress


* Expanding dataset for stacked regressions
expand `numvar'
bys person_id: gen order = _n
	
* Creating variables for stacked regressions

gen outcome = .				
gen stack_weight = .
gen include_all = 0
gen include_admin = 0
gen include_survey = 0 

local i = 0
local dropvars_all "temp"
local dropvars_admin "temp"
local dropvars_survey "temp"

foreach var of local list {

	local ++i

	foreach word of local list_survey{
		if "`var'"=="`word'"{
			local source "survey"
			local controls "`controls_survey'"
			local condition "`condition_survey'"
			local weight "`weight_survey'"
			}
		}
	
	foreach word of local list_admin{
		if "`var'"=="`word'"{
			local source "admin"
			local controls "`controls_admin'"
			local condition "`condition_admin'"
			local weight "`weight_admin'"
			}
		}
	
	di "`i' `var' `source'"
	
	replace outcome = `var' if (order == `i') `condition'
	
	sum `weight' if (order == `i') `condition'
	replace stack_weight = `weight'/r(mean) if (order == `i') `condition'
		
	gen treatment_X`i' = treatment*(order == `i')
	gen ohp_all_X`i' = ohp_all_ever_`source'*(order == `i')
	replace ohp_all_X`i' = 0 if ohp_all_X`i' == .
	
	foreach control of varlist constant `controls' ``var'_controls'{
		gen X`i'_`control' = `control' *(order == `i')
		replace X`i'_`control' = 0 if X`i'_`control' == .
		}

	* Choosing which variables to include in stacked regression
	foreach datatype in admin survey all{
		local skip_`i'_`datatype' = 0
		
		local include_var: list var & list_`datatype'

		if "`var'"=="`include_var'"{
			di "Include `i' `var' in STE for `datatype'"
			replace include_`datatype' = 1 if order==`i' 
			}

		else{
			local skip_`i'_`datatype' = 1
			local dropvars_`datatype' "`dropvars_`datatype'' treatment_X`i' ohp_all_X`i' X`i'_* "
			}

		} // end of datatype loop
		
	* Choosing which variables to reverse signs for in STE
	local reverse_`i' = 0
		
	local reverse_var: list var & list_reverse
		
	if "`var'" == "`reverse_var'" {
		di "`i' `var' should have reverse sign in STE"
		local reverse_`i' = 1
	}	
		
	} // end of var loop  
	


desc outcome stack_weight *X*


* Running stacked regressions and calculating standardized treatment

foreach datatype in admin survey all{
   dis "`list_`datatype''"
	if "`list_`datatype''" == "" {
		continue
		}

	di _newline "Running stacked regression for `datatype' data"
	
	preserve
	gen temp=.
	di "Dropping `dropvars_`datatype''"
	drop `dropvars_`datatype''

	desc treatment*
	
	* ITT
	local lincom_string  ""
   
	reg outcome treatment_X* X* if include_`datatype'==1 [pw=stack_weight], cluster(household_id) noconstant
	
	matrix postest=r(table)
	matrix list postest
	 
	forval i = 1/`numvar'{

		if "`skip_`i'_`datatype''" == "1"{
			di "Skipping `i'"
			continue
			}
		
		di "`i'"
		
		local `i'_itt_beta_pool = round(_b[treatment_X`i'],10^(min(-2, int(log10(abs(_b[treatment_X`i'])))-2)))
		local `i'_itt_se_pool = _se[treatment_X`i']
		
		local t = _b[treatment_X`i']/_se[treatment_X`i']
		local `i'_itt_p_pool = round(postest[4,1],0.001)
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		di ``i'_itt_beta_pool'
		di ``i'_itt_beta'
		*assert round(``i'_itt_beta_pool',10^(min(-2, int(log10(abs(``i'_itt_beta_pool')))-2))) == round(``i'_itt_beta',10^(min(-2, int(log10(abs(``i'_itt_beta')))-2)))
		
		if `reverse_`i''==1{
			if "`lincom_string'" == "" local lincom_string  "- 1/`numvar_`datatype''*_b[treatment_X`i']/``i'_sd'"
			else local lincom_string  "`lincom_string' - 1/`numvar_`datatype''*_b[treatment_X`i']/``i'_sd'"
		}
		else {
			if "`lincom_string'" == "" local lincom_string  "1/`numvar_`datatype''*_b[treatment_X`i']/``i'_sd'"
			else local lincom_string  "`lincom_string' + 1/`numvar_`datatype''*_b[treatment_X`i']/``i'_sd'"
		}
		
		}
	
	di "`lincom_string'"
	lincom "`lincom_string'"
	local itt_effect = round(r(estimate),10^(min(-2, int(log10(abs(r(estimate))))-2)))
	local itt_se = round(r(se),10^(min(-2, int(log10(abs(r(se))))-2)))
	local itt_p = round(2*normal(-abs(r(estimate)/r(se))),0.01)

	if 2*normal(-abs(r(estimate)/r(se)))<0.1 {
		local itt_p = round(2*normal(-abs(r(estimate)/r(se))),0.001)
	}
	
	if 2*normal(-abs(r(estimate)/r(se)))<0.001 {
		local itt_p = 2*normal(-abs(r(estimate)/r(se)))
	}
	

	
	matrix `results'[`ste_row_`datatype''+1, 2] = `itt_effect'
	matrix `results'[`ste_row_`datatype''+2, 2] = `itt_se'
	
	
	if `spend_est' == 1 {
	        if `inp_12m_spend'==1 {
		lincom 1*(312*_b[treatment_X1]+150*_b[treatment_X2]+435*_b[treatment_X3]+7523*_b[treatment_X4])
		}
		
		else {
		lincom 312*_b[treatment_X1]+150*(_b[treatment_X2]+_b[treatment_X3])+435*_b[treatment_X4]+7523*_b[treatment_X5]
		}
		
		local spend_itt_effect = round(r(estimate),10^(min(-2, int(log10(abs(r(estimate))))-2)))
		local spend_itt_se = round(r(se),10^(min(-2, int(log10(abs(r(se))))-2)))
		local spend_itt_p = round(2*normal(-abs(r(estimate)/r(se))),0.001)
		local spend_itt_p_complete = 2*normal(-abs(r(estimate)/r(se)))
	
	
		if `spend_itt_p'==0 {
		local spend_itt_p = 2*normal(-abs(r(estimate)/r(se)))
		}
	
		matrix spend_est_`table'[1,2] = `spend_itt_effect'
		matrix spend_est_`table'[2,2] = `spend_itt_se'
		
	
	} // end of spend_est loop
		
* TOT
	
	local lincom_string  ""
	
	ivregress 2sls outcome (ohp_all_X* = treatment_X*) X* if 1==1 [pw = stack_weight], cluster(household_id) noconstant
	
	matrix postest=r(table)
	*matrix list postest
	
	forval i = 1/`numvar'{

			if "`skip_`i'_`datatype''" == "1"{
			di "Skipping `i'"
			continue
			}

		di "`i'"
		
		local `i'_tot_beta_pool = round(_b[ohp_all_X`i'], 10^(min(-2, int(log10(abs(_b[ohp_all_X`i'])))-2)))
		local `i'_tot_se_pool = round(_se[ohp_all_X`i'], 10^(min(-2, int(log10(abs(_se[ohp_all_X`i'])))-2)))
		
		local `i'_tot_p_pool = round(postest[4,1], 0.001)
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		di ``i'_tot_beta_pool'
		di ``i'_tot_beta'
		*assert round(``i'_tot_beta_pool',10^(min(-2, int(log10(abs(``i'_tot_beta_pool')))-2))) == round(``i'_tot_beta',10^(min(-2, int(log10(abs(``i'_tot_beta')))-2))) 
		
		if `reverse_`i''==1 {
			if "`lincom_string'" == "" local lincom_string  "- 1/`numvar_`datatype''*_b[ohp_all_X`i']/``i'_sd'"
			else local lincom_string  "`lincom_string' - 1/`numvar_`datatype''*_b[ohp_all_X`i']/``i'_sd'"
		}
		else {
			if "`lincom_string'" == "" local lincom_string  "1/`numvar_`datatype''*_b[ohp_all_X`i']/``i'_sd'"
			else local lincom_string  "`lincom_string' + 1/`numvar_`datatype''*_b[ohp_all_X`i']/``i'_sd'"
		}
		}

	di "`lincom_string'"
	lincom "`lincom_string'"
	local tot_effect = round(r(estimate),10^(min(-2, int(log10(abs(r(estimate))))-2)))
	local tot_se = round(r(se),10^(min(-2, int(log10(abs(r(se))))-2)))
	local tot_p = round(2*normal(-abs(r(estimate)/r(se))),0.01)
	
	
	if 2*normal(-abs(r(estimate)/r(se)))<0.1 {
		local tot_p = round(2*normal(-abs(r(estimate)/r(se))),0.001)
	}
	
	if 2*normal(-abs(r(estimate)/r(se)))<0.001 {
		local tot_p = 2*normal(-abs(r(estimate)/r(se)))
	}
	
	
	local `table'_ste_tot_effect=`tot_effect'
	local `table'_ste_tot_se=`tot_se'
	local `table'_ste_tot_p=`tot_p'

	matrix `results'[`ste_row_`datatype''+1, 3] = `tot_effect'
	matrix `results'[`ste_row_`datatype''+2, 3] = `tot_se'
	matrix `results'[`ste_row_`datatype''+1, 4] = `tot_p'
	
	if `spend_est' == 1 {
	
		if `inp_12m_spend'==1 {
		lincom 1*(312*_b[ohp_all_X1]+150*_b[ohp_all_X2]+435*_b[ohp_all_X3]+7523*_b[ohp_all_X4])
		}
		
		else {
		lincom 312*_b[ohp_all_X1]+150*(_b[ohp_all_X2]+_b[ohp_all_X3])+435*_b[ohp_all_X4]+7523*_b[ohp_all_X5]
		}
		
		local spend_tot_effect = round(r(estimate),10^(min(-2, int(log10(abs(r(estimate))))-2)))
		local spend_tot_se = round(r(se),10^(min(-2, int(log10(abs(r(se))))-2)))
		local spend_tot_p = round(2*normal(-abs(r(estimate)/r(se))),0.001)
		local spend_tot_p_complete = 2*normal(-abs(r(estimate)/r(se)))
	
		if `spend_tot_p'==0 {
		local spend_tot_p = 2*normal(-abs(r(estimate)/r(se)))
		}
	
		matrix spend_est_`table'[1,3] = `spend_tot_effect'
		matrix spend_est_`table'[2,3] = `spend_tot_se'
		matrix spend_est_`table'[1,4] = `spend_tot_p_complete'
		di "tot `spend_tot_p_complete'"
		di "itt `spend_itt_p_complete'"
		
		matrix spend_est_`table'[1,1] =`spend_mean'
	
	} // end of spend_est loop

	restore
	
	} //end of datatype loop

local source ""
local temp "`temp'"
	
} // End of STE switch

******************************************************
******************************************************

* Displaying table
matrix list `results'
di "`temp'"

matrix rownames `results' = `temp'


noi: matrix list `results', title("Table `table': Comparison of treatment and controls")
noi: display "Sample size for survey data analysis is `Table`table'_N_survey'"
noi: display "Sample size for administrative data analysis is `Table`table'_N_admin'"

if `spend_est'==1 {
noi: matrix list spend_est_`table', title(estimated spending effect for table `table')
		} // end of spend_est

if $ALT==1 {
local temp_alt "`temp_alt'"
di "`temp_alt'"
matrix rownames `table'_alt = `temp_alt'

noi: matrix list `table'_alt
}

assert 1==1
dis "test"







