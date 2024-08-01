

* ***********************************************
* Analysis comparing treatment and controls
* ***********************************************

***************************
*Set up matrix for results

local results = "Table`table'_`subpop'_`date'"

local numvar : word count `list'
di "`numvar'"


if `sensitivity'!=1 {
local rows = 2*(`numvar') // used to be +6 for standardized treatment effects, then was +2 (not sure why)
di "`rows'"
matrix define `results' = J(`rows', 4, .)
matrix colnames `results' = "c_mean" "RF" "LATE" "p_vals" 

}


if `sensitivity'==1 {
local rows = 3*(`numvar') 
di "`rows'"
matrix define `results' = J(`rows', 1, .)
matrix colnames `results' = "RF`lab'" 
}

if `spend_est' == 1 {
matrix define spend_est_`table' = J(2,4,.)
matrix rownames spend_est_`table' = "spending estimate" "se"
matrix colnames spend_est_`table' = "c-mean" "RF" "2SLS" "p_vals" 
			}

*******************************
* Sample size for table notes

count if 1==1 `condition_survey'
local Table`table'_N_survey = r(N)

count if 1==1 `condition_admin'
local Table`table'_N_admin = r(N)

* inflate 0/1 variables by 100
local inflate_list "$binary"
di "`inflate_list'"

***********************************
* Control means and comparison of treaments and controls 
			
local temp  ""
local count = 0
local i = 0


	foreach var of local list {
		
		local ++i
		di "`i' `var' `source'"
		dis "reg `var' if treatment==0 `condition' `subsample' [pw=`weight']"
		
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
		
		*Control means
		di "reg `var' if treatment==0 `condition' `subsample' [pw=`weight']"
		reg `var' if treatment==0 `condition' `subsample' [pw=`weight']
		local `i'_mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		local `i'_sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
	
		if `sensitivity'!=1 {
		matrix `results'[`count'+1, 1] = ``i'_mean'
		matrix `results'[`count'+2, 1] = ``i'_sd'
		
			} // End sensitivity switch
		
	
		*ITT comparison
		dis "reg `var' treatment `controls' ``var'_controls' ``survey'_controls' `hdd_controls' if 1==1 `condition' `subsample' [pw=`weight'], cluster(household_id)"
		reg `var' treatment `controls' ``var'_controls' ``survey'_controls' `hdd_controls' if 1==1 `condition' `subsample' [pw=`weight'], cluster(household_id)
			
		local `i'_itt_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
		local `i'_itt_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
		local `i'_itt_p = round(2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))*1000)/1000 
		
		local `i'_itt95_l=round(_b[treatment]-invttail(e(df_r), 0.025)*_se[treatment],  10^(min(-2, int(log10(abs(_b[treatment]-invttail(e(df_r), 0.025)*_se[treatment])))-2)))
		local `i'_itt95_h=round(_b[treatment]+invttail(e(df_r), 0.025)*_se[treatment],  10^(min(-2, int(log10(abs(_b[treatment]+invttail(e(df_r), 0.025)*_se[treatment])))-2)))
		
		matrix postest=r(table)
		local `i'_itt_p= round(postest[4,1],10^(min(-3, int(log10(abs(postest[4,1])))-2)))
		
		if `sensitivity'!=1 {
		matrix `results'[`count'+1, 2] = ``i'_itt_beta'
		matrix `results'[`count'+2, 2] = ``i'_itt_se'
		
		}
		
		if `sensitivity'==1 {
		
		matrix `results'[`count'+1, 1] = ``i'_itt_beta'
		matrix `results'[`count'+2, 1] = ``i'_itt_se'
		matrix `results'[`count'+3, 1] = ``i'_itt_p'
		
		local count = `count' + 3
		di `count'
		
		local temp "`temp' `var' se p"
		local count_alt = `count_alt'+1
		if "`var'"!="" {
		local temp_alt "`temp_alt' `var'"
		}
		
		} 
		
		
		*ToT comparison
		dis "ivregress 2sls `var' (ohp_all_ever_`source'=treatment) `controls' ``var'_controls' ``survey'_controls' `hdd_controls' if 1==1 `subsample' `condition' [pw = `weight'], cluster(household_id)"  	
		ivregress 2sls `var' (ohp_all_ever_`source'=treatment) `controls' ``var'_controls' ``survey'_controls' `hdd_controls' ///
		  if 1==1 `subsample' `condition' [pw = `weight'], cluster(household_id)  	
		matrix postest=r(table)
		
		local `i'_tot_beta = round(_b[ohp_all_ever_`source'],10^(min(-3, int(log10(abs(_b[ohp_all_ever_`source'])))-2)))
		local `i'_tot_se = round(_se[ohp_all_ever_`source'], 10^(min(-3, int(log10(abs(_se[ohp_all_ever_`source'])))-2)))
		local `i'_tot_p = round(postest[4,1],10^(min(-3, int(log10(abs(postest[4,1])))-2)))
		
		local `i'_tot95_l=round(_b[ohp_all_ever_`source']-invnormal(0.975)*_se[ohp_all_ever_`source'],  10^(min(-2, int(log10(abs(_b[ohp_all_ever_`source']-invnormal(0.975)*_se[ohp_all_ever_`source'])))-2)))
		local `i'_tot95_h=round(_b[ohp_all_ever_`source']+invnormal(0.975)*_se[ohp_all_ever_`source'],  10^(min(-2, int(log10(abs(_b[ohp_all_ever_`source']+invnormal(0.975)*_se[ohp_all_ever_`source'])))-2)))
		
		
	if `sensitivity'!=1 {
		matrix `results'[`count'+1, 3] = ``i'_tot_beta'
		matrix `results'[`count'+2, 3] = ``i'_tot_se'
		matrix `results'[`count'+1, 4] = ``i'_tot_p'
				
		local temp "`temp' `var' se"
		local count_alt = `count_alt'+1
		if "`var'"!="" {
		local temp_alt "`temp_alt' `var'"
		}
		
		local count = `count' + 2
		di `count'
		
		} // End sensitivity switch

	

		
		} //end of var loop
	
	
	local count = `count'+2
	*local temp "`temp' ste se"
	

if `spend_est'==1 {

local control_mean_spend= 435*(`1_mean'/1.5) + 7523*(`2_mean'/1.5)
local control_mean_spend_round= round(`control_mean_spend',10^(min(-3, int(log10(abs(r(estimate))))-2)))
matrix spend_est_`table'[1,1] = `control_mean_spend_round'
	
matrix list spend_est_`table'


*Standardized treatment effects

local list_all : list list - exclude_from_ste

local numvar_all : word count `list_all'

local start_all = 1
local stop_all = `numvar'
local ste_row_all = `numvar'*2+4

compress

* Expanding dataset for stacked regressions
expand `numvar'
bys res_person_id: gen order = _n
	
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
	
	di "`i' `var' `source'"
	
	replace outcome = `var' if (order == `i') `condition'
		
	gen treatment_X`i' = treatment*(order == `i')
	gen ohp_all_X`i' = ohp_all_ever_`source'*(order == `i')
	replace ohp_all_X`i' = 0 if ohp_all_X`i' == .
	
	foreach control of varlist constant `controls' ``var'_controls'{
		gen X`i'_`control' = `control' *(order == `i')
		replace X`i'_`control' = 0 if X`i'_`control' == .
		}

	* Choosing which variables to include in stacked regression
	foreach datatype in all{
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

foreach datatype in all {
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
   
	reg outcome treatment_X* X*, cluster(household_id) noconstant // if include_`datatype'==1 [pw=stack_weight]
	 
	forval i = 1/`numvar'{

		if "`skip_`i'_`datatype''" == "1"{
			di "Skipping `i'"
			continue
			}
		
		di "`i'"
		
		local `i'_itt_beta_pool = round(_b[treatment_X`i'],10^(min(-3, int(log10(abs(_b[treatment_X`i'])))-2)))
		local `i'_itt_se_pool = _se[treatment_X`i']
		
		local t = _b[treatment_X`i']/_se[treatment_X`i']
		local `i'_itt_p_pool = 2*ttail(e(df_r), abs(`t'))
		
		di ``i'_itt_beta_pool'
		di ``i'_itt_beta'
		assert round(``i'_itt_beta_pool',10^(min(-3, int(log10(abs(``i'_itt_beta_pool')))-2))) == round(``i'_itt_beta',10^(min(-3, int(log10(abs(``i'_itt_beta')))-2)))
		
		if `reverse_`i''==1{
			if "`lincom_string'" == "" local lincom_string  "- 1/`numvar_`datatype''*_b[treatment_X`i']/``i'_sd'"
			else local lincom_string  "`lincom_string' - 1/`numvar_`datatype''*_b[treatment_X`i']/``i'_sd'"
		}
		else {
			if "`lincom_string'" == "" local lincom_string  "1/`numvar_`datatype''*_b[treatment_X`i']/``i'_sd'"
			else local lincom_string  "`lincom_string' + 1/`numvar_`datatype''*_b[treatment_X`i']/``i'_sd'"
		}
		
		
		lincom 435*_b[treatment_X1]/1.5 + 7523*(_b[treatment_X2])/1.5 
		local spend_itt_effect = round(r(estimate),10^(min(-3, int(log10(abs(r(estimate))))-2)))
		local spend_itt_se = round(r(se),10^(min(-3, int(log10(abs(r(se))))-2)))
		local spend_itt_p = round(2*ttail(r(df), abs(r(estimate)/r(se))),0.001)
		local spend_itt_p_complete = 2*ttail(r(df), abs(r(estimate)/r(se)))
	
	
		if `spend_itt_p'==0 {
		local spend_itt_p = 2*ttail(r(df), abs(r(estimate)/r(se)))
		}
	
		matrix spend_est_`table'[1,2] = `spend_itt_effect'
		matrix spend_est_`table'[2,2] = `spend_itt_se'
		*matrix spend_est_`table'[1,4] = `spend_itt_p'
	
	} // end of spend_est loop
		
* TOT
	
	local lincom_string  ""
	
	ivregress 2sls outcome (ohp_all_X* = treatment_X*) X* if 1==1, cluster(household_id) noconstant // [pw = stack_weight]
	
	forval i = 1/`numvar'{

			if "`skip_`i'_`datatype''" == "1"{
			di "Skipping `i'"
			continue
			}

		di "`i'"
		
		local `i'_tot_beta_pool = round(_b[ohp_all_X`i'], 10^(min(-3, int(log10(abs(_b[ohp_all_X`i'])))-2)))
		local `i'_tot_se_pool = round(_se[ohp_all_X`i'], 10^(min(-3, int(log10(abs(_se[ohp_all_X`i'])))-2)))
		
		local `i'_tot_p_pool = round(2*normal(-abs(_b[ohp_all_X`i']/_se[ohp_all_X`i']))*1000)/1000
		
		di ``i'_tot_beta_pool'
		di ``i'_tot_beta'
		assert round(``i'_tot_beta_pool',10^(min(-3, int(log10(abs(``i'_tot_beta_pool')))-2))) == round(``i'_tot_beta',10^(min(-3, int(log10(abs(``i'_tot_beta')))-2))) 
		
		if `reverse_`i''==1 {
			if "`lincom_string'" == "" local lincom_string  "- 1/`numvar_`datatype''*_b[ohp_all_X`i']/``i'_sd'"
			else local lincom_string  "`lincom_string' - 1/`numvar_`datatype''*_b[ohp_all_X`i']/``i'_sd'"
		}
		else {
			if "`lincom_string'" == "" local lincom_string  "1/`numvar_`datatype''*_b[ohp_all_X`i']/``i'_sd'"
			else local lincom_string  "`lincom_string' + 1/`numvar_`datatype''*_b[ohp_all_X`i']/``i'_sd'"
		}
		}
	
	if `spend_est' == 1 {
		lincom 435*_b[ohp_all_X1]/1.5 + 7523*(_b[ohp_all_X2])/1.5
		local spend_tot_effect = round(r(estimate),10^(min(-3, int(log10(abs(r(estimate))))-2)))
		local spend_tot_se = round(r(se),10^(min(-3, int(log10(abs(r(se))))-2)))
		local spend_tot_p = round(2*normal(-abs(r(estimate)/r(se))),0.001)
		local spend_tot_p_complete = 2*normal(-abs(r(estimate)/r(se)))
	
		*if `spend_tot_p'==0 {
		local spend_tot_p = 2*normal(-abs(r(estimate)/r(se)))
		*}
	
		matrix spend_est_`table'[1,3] = `spend_tot_effect'
		matrix spend_est_`table'[2,3] = `spend_tot_se'
		matrix spend_est_`table'[1,4] = `spend_tot_p'
		di "tot `spend_tot_p_complete'"
		di "itt `spend_itt_p_complete'"
	
	} // end of spend_est loop

	restore
	
	} //end of datatype loop

local source ""
*local temp "`temp' joint-ste se"
	
}

******************************************************
******************************************************

* Displaying table

matrix list `results'

*di `temp'

matrix rownames `results' = `temp'

noi: matrix list `results', title("Table `table': Comparison of treatment and controls")
noi: display "Sample size for ED analysis is `Table`table'_N_survey'"


if $ALT==1 {
local temp_alt "`temp_alt'"
di "`temp_alt'"
matrix rownames `table'_alt = `temp_alt'

noi: matrix list `table'_alt
}

*



