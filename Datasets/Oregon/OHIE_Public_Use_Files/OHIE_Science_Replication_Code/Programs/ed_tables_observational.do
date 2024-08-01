
* ***********************************************
* Analysis comparing treatment and controls
* ***********************************************


***************************
*Set up matrix for results
*keep household_id res_person_id treatment ohp_all_ever_30sep2009 constant numhh_list ///
  *`var_interest' `list_survey' `list_admin' `controls_survey' `controls_admin' `weight_survey' `weight_admin' `add_vars' 

local results = "Table`table'"

local numvar : word count `list'
di "`numvar'"

local rows = 3*(`numvar')

matrix define `results' = J(`rows', 2, .)

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
		di "`i' `n' `var' `source'" 
		
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
		
		
		/*Control means
		di "reg `var' if `var_interest_`n''==0 `condition`n'' [pw=`weight']"
		reg `var' if `var_interest_`n''==0 `condition`i'' [pw=`weight']
		local `i'_mean = round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
		local `i'_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-1)))
		matrix `results'[`count'+1, 1] = ``i'_mean'
		matrix `results'[`count'+2, 1] = ``i'_sd' */
		
		
		*ITT comparison (TOT comparison for column 1)
	if "`var_interest_`n''"!= "treatment" {	
	di "reg `var' `var_interest_`n'' `controls' ``var'_controls' if 1==1 `condition`n'' [pw=`weight'], cluster(household_id)"
		reg `var' `var_interest_`n'' `controls' ``var'_controls' if 1==1 `condition`n'' [pw=`weight'], cluster(household_id)
		
		matrix postest=r(table)
		
		local `i'_itt_beta = round(_b[`var_interest_`n''],10^(min(-2, int(log10(abs(_b[`var_interest_`n''])))-2)))
		local `i'_itt_se = round(_se[`var_interest_`n''], 10^(min(-2, int(log10(abs(_se[`var_interest_`n''])))-2)))
		local `i'_itt_p = round(postest[4, 1], 10^(min(-3, int(log10(abs(postest[4,1])))-2)))
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		matrix `results'[`count'+1, 2] = ``i'_itt_beta'
		matrix `results'[`count'+2, 2] = ``i'_itt_se'
		matrix `results'[`count'+3, 2] = ``i'_itt_p'
      }
      else {
	  di "ivregress 2sls `var' (ohp_all_ever_30sep2009= treatment) `controls' ``var'_controls' if 1==1 `condition`n'' [pw = `weight'], cluster(household_id)"
		ivregress 2sls `var' (ohp_all_ever_30sep2009= treatment) `controls' ``var'_controls' ///
		  if 1==1 `condition' `condition`n'' [pw = `weight'], cluster(household_id)  
		
		matrix postest=r(table)
			
		local `i'_tot_beta = round(_b[ohp_all_ever_30sep2009],10^(min(-2, int(log10(abs(_b[ohp_all_ever_30sep2009])))-2)))
		local `i'_tot_se = round(_se[ohp_all_ever_30sep2009], 10^(min(-2, int(log10(abs(_se[ohp_all_ever_30sep2009])))-2)))
		local `i'_tot_p = round(postest[4, 1], 10^(min(-3, int(log10(abs(postest[4,1])))-2)))
		
		matrix `results'[`count'+1, 2] = ``i'_tot_beta'
		matrix `results'[`count'+2, 2] = ``i'_tot_se'
		matrix `results'[`count'+3, 2] = ``i'_tot_p'
                
     }
		local count = `count' + 3
		di `count'
		
		local temp "`temp' `var' se p"

		
		} //end of var loop
	
	

local source ""
	

******************************************************
******************************************************

* Displaying table

matrix rownames `results' = `temp'

noi: matrix list `results', title("Table `table'")
noi: display "Sample size for survey data analysis is `Table`table'_N_survey'"
noi: display "Sample size for administrative data analysis is `Table`table'_N_admin'"






