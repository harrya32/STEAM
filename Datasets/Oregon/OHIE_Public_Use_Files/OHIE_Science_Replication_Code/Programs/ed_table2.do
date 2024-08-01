

* ***********************************************
* Analysis comparing treatment and controls
* ***********************************************

***************************
*Set up matrix for results

local results = "Table`table'_`subpop'_`date'"

local numvar : word count `list'
di "`numvar'"

local rows = 2*(`numvar') // used to be +6 for standardized treatment effects, then was +2 (not sure why)
di "`rows'"

matrix define `results' = J(`rows', 6, .)
matrix colnames `results' = "N" "FS" "c_mean" "RF" "LATE" "p_vals"
matrix rownames `results'= "`subpop'"


* inflate 0/1 variables by 100
local inflate_list "$binary"
di "`inflate_list'"

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


	local row=1
	
	foreach var of local list {
		local count=1
		
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
		
		
		local ++i
		di "`i' `var' `source'"
		dis "reg `var' if treatment==0 `condition' `subsample' [pw=`weight']"
		
		*Controls
			if regexm("`subpop'", "all") ==1 {
				local controls "$adjustments ``var'_controls'"
					} // End regexm loop
					
			if regexm("`subpop'", "all")!=1 {
				local controls "$adjustments"
					} // End regexm loop
	
		
		*Binary variables don't need SDs displayed
		local binary_`i'=0
		local binary_var: list var & list_binary
		if "`var'"=="`binary_var'" {
			di "`i' `var' does not need SD"
			local binary_`i'=1
		}
						
		*sample counts
		count if `subpop'==1
		matrix `results'[`row', 1]= r(N)
		
		*first stage
		di "reg `fs_var' treatment `controls' if `subpop'==1 [pw=`weight'], cluster(household_id)"
		reg `fs_var' treatment `controls' if `subpop'==1 [pw=`weight'], cluster(household_id)
		local fs_`subpop' = round(_b[treatment], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		
		matrix `results'[`row',2] = `fs_`subpop''
		
		*Control means
		reg `var' if treatment==0 `condition' `subsample' [pw=`weight']
		local `i'_mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		local `i'_sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
	
		matrix `results'[`row', 3] = ``i'_mean'
		matrix `results'[`row'+1, 3] = ``i'_sd'
	
	matrix list `results' 
		
		/*if `binary_`i''!=1 {
			matrix `results'[`count'+2, 1] = ``i'_sd'
		}*/
		
		if $ALT==1 {
		matrix `table'_alt[`count_alt'+1, 1] = ``i'_mean'
			if `binary_`i''!=1 | "`var'" == "cvd_risk_point" {
			matrix `table'_alt[`count_alt'+1, 2] = ``i'_sd'
			}
			
		} 
			
		*ITT comparison
		dis "reg `var' treatment `controls' if 1==1 `condition' `subsample' [pw=`weight'], cluster(household_id)"
		di "`controls'"
		reg `var' treatment `controls' if 1==1 `condition' `subsample' [pw=`weight'], cluster(household_id)
		local `i'_itt_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
		local `i'_itt_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
		*local `i'_itt_p = round(2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))*1000)/1000 
		
		local `i'_itt95_l=round(_b[treatment]-invttail(e(df_r), 0.025)*_se[treatment],  10^(min(-2, int(log10(abs(_b[treatment]-invttail(e(df_r), 0.025)*_se[treatment])))-2)))
		local `i'_itt95_h=round(_b[treatment]+invttail(e(df_r), 0.025)*_se[treatment],  10^(min(-2, int(log10(abs(_b[treatment]+invttail(e(df_r), 0.025)*_se[treatment])))-2)))
		
		matrix postest=r(table)
		local `i'_itt_p= round(postest[4,1], 10^(min(-3, int(log10(abs(postest[4,1])))-3)))
		
		matrix `results'[`row', 4] = ``i'_itt_beta'
		matrix `results'[`row'+1, 4] = ``i'_itt_se'
		
		*ToT comparison
		dis "ivregress 2sls `var' (ohp_all_ever_`source'=treatment) `controls' if 1==1 `subsample' `condition' [pw = `weight'], cluster(household_id)"  
		di "`controls'"
		ivregress 2sls `var' (ohp_all_ever_`source'=treatment) `controls' ///
		  if 1==1 `subsample' `condition' [pw = `weight'], cluster(household_id)  	
		matrix postest=r(table)
		*matrix list postest
		
		local `i'_tot_beta = round(_b[ohp_all_ever_`source'],10^(min(-3, int(log10(abs(_b[ohp_all_ever_`source'])))-2)))
		local `i'_tot_se = round(_se[ohp_all_ever_`source'], 10^(min(-3, int(log10(abs(_se[ohp_all_ever_`source'])))-2)))
		local `i'_tot_p = round(postest[4,1],10^(min(-3, int(log10(abs(postest[4,1])))-2)))
		
		local `i'_tot95_l=round(_b[ohp_all_ever_`source']-invnormal(0.975)*_se[ohp_all_ever_`source'],  10^(min(-2, int(log10(abs(_b[ohp_all_ever_`source']-invnormal(0.975)*_se[ohp_all_ever_`source'])))-2)))
		local `i'_tot95_h=round(_b[ohp_all_ever_`source']+invnormal(0.975)*_se[ohp_all_ever_`source'],  10^(min(-2, int(log10(abs(_b[ohp_all_ever_`source']+invnormal(0.975)*_se[ohp_all_ever_`source'])))-2)))
		
		if ``i'_tot_p'==0 {
		local ``i'_tot_p'= round(postest[4,1], 10^(min(-3, int(log10(abs(postest[4,1])))-3)))
		*local `i'_tot_p = 2*normal(-abs(_b[ohp_all_ever_`source']/_se[ohp_all_ever_`source']))
		}
		
		
		matrix `results'[`row', 5] = ``i'_tot_beta'
		matrix `results'[`row'+1, 5] = ``i'_tot_se'
		matrix `results'[`row', 6] = ``i'_tot_p'
		
	if $ALT==1 {
	matrix `table'_alt[`count_alt'+1, 6] = ``i'_tot_beta'
	matrix `table'_alt[`count_alt'+1, 7] = ``i'_tot95_l'
	matrix `table'_alt[`count_alt'+1, 8] = ``i'_tot95_h'
               }
		
		local row = `row' + 2
		di `row'
		
		} //end of var loop
	


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
