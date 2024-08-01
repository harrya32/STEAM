

* ***********************************************
* Analysis comparing treatment and controls
* ***********************************************


***************************
*Set up matrix for results
keep household_id person_id treatment ohp_all_ever_survey ohp_all_ever_admin ohp_all_ever_inperson constant numhh_list ///
  `var_interest' `list_survey' `list_admin' `controls_survey' `controls_admin' `weight_survey' `weight_admin' `add_vars' 


local list "`list_admin' `list_survey'"
local source ""

local results = "Table`table'"

local numvar : word count `list'
di "`numvar'"

local rows = 3*(`numvar')

matrix define `results' = J(`rows', 2, .)


***********************************
* Control means and comparison of treaments and controls 
			
local temp  ""
local count = 0
local i = 0

foreach datatype in admin survey {

	local source "`datatype'"
	local controls "`controls_`datatype''"
	local condition " `condition_`datatype''"
	local weight "`weight_`datatype''"
	
	foreach var of local list_`datatype' {
		
		local ++i
		di "`i' `var' `source'"
		
		*Control means
		reg `var' if `var_interest'==0 `condition' [pw=`weight']
		local `i'_mean = round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
		local `i'_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-1)))
		matrix `results'[`count'+1, 1] = ``i'_mean'
		matrix `results'[`count'+2, 1] = ``i'_sd'
		
		
		*ITT comparison (TOT comparison for column 1
	if "`var_interest'"!= "treatment" {	
		reg `var' `var_interest' `controls' ``var'_controls' if 1==1 `condition' [pw=`weight'], cluster(household_id)
		
		matrix postest=r(table)
		matrix list postest
		
		local `i'_itt_beta = round(_b[`var_interest'],10^(min(-2, int(log10(abs(_b[`var_interest'])))-2)))
		local `i'_itt_se = round(_se[`var_interest'], 10^(min(-2, int(log10(abs(_se[`var_interest'])))-2)))
		local `i'_itt_p = round(postest[4,1], 0.01)
		
		if postest[4,1]<0.1 {
			local `i'_itt_p=round(postest[4,1], 0.001)
			
		}
		
		if postest[4,1]<0.001 {
			local `i'_itt_p=postest[4,1]
			
		}
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		matrix `results'[`count'+1, 2] = ``i'_itt_beta'
		matrix `results'[`count'+2, 2] = ``i'_itt_se'
		matrix `results'[`count'+3, 2] = ``i'_itt_p'
      }
      else {
		ivregress 2sls `var' (ohp_all_ever_inperson= treatment) `controls' ``var'_controls' ///
		  if 1==1 `condition' [pw = `weight'], cluster(household_id)  
		
		matrix postest=r(table)
		matrix list postest
			
		local `i'_tot_beta = round(_b[ohp_all_ever_inperson],10^(min(-2, int(log10(abs(_b[ohp_all_ever_inperson])))-2)))
		local `i'_tot_se = round(_se[ohp_all_ever_inperson], 10^(min(-2, int(log10(abs(_se[ohp_all_ever_inperson])))-2)))
		local `i'_tot_p = round(postest[4,1], 0.01)
		
		
		if postest[4,1]<0.1 {
			local `i'_tot_p=round(postest[4,1], 0.001)
			
		}
		
		if postest[4,1]<0.001 {
			local `i'_tot_p=postest[4,1]
			
		}
		
		matrix `results'[`count'+1, 2] = ``i'_tot_beta'
		matrix `results'[`count'+2, 2] = ``i'_tot_se'
		matrix `results'[`count'+3, 2] = ``i'_tot_p'
                
     }
		local count = `count' + 3
		di `count'
		
		local temp "`temp' `var' se p"

		
		} //end of var loop
	
	} //end of datatype loop

local source ""
	

******************************************************
******************************************************

* Displaying table

matrix rownames `results' = `temp'

*noi: matrix list `results', title("Table `table'")
noi: display "Sample size for survey data analysis is `Table`table'_N_survey'"
noi: display "Sample size for administrative data analysis is `Table`table'_N_admin'"






