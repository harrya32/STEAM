

* ***********************************************
* Analysis comparing treatment and controls
* ***********************************************


***************************
*Set up matrix for results
keep household_id person_id treatment ohp_all_ever_survey ohp_all_ever_admin constant numhh_list ///
  `list_survey' `list_admin' `controls_survey' `controls_admin' `weight_survey' `weight_admin' `add_vars' 

 * inflate 0/1 variables by 100
 local inflate_list "`list_binary' phq_prob"

local list "`list_admin' `list_survey'"
local source ""

local numvar : word count `list'
di "`numvar'"

matrix logit_`table' = J(`numvar'*3,3,.)
matrix colnames logit_`table' = "c-mean" "RF" "RF/Logit ME" 

if $ALT==1 {
matrix logit_`table'_alt = J(`numvar',10,.)
matrix colnames logit_`table'_alt = "c-mean" "se" "RF" "l95" "h95" "pvalue" "Logit" "l95" "h95" "pvalue"
}

		
*******************************
* Sample size for table notes

count if 1==1 `condition_survey'
local Table`table'_N_survey = r(N)

count if 1==1 `condition_admin'
local Table`table'_N_admin = r(N)


***********************************
* Control means and comparison of treaments and controls 


local logit_count = 0
local logit_temp ""

if $ALT==1 {
local logit_count_alt = 0
local logit_temp_alt ""
}


local i = 0

foreach datatype in admin survey{

	local source "`datatype'"
	local controls "`controls_`datatype''"
	local condition "`condition_`datatype''"
	local weight "`weight_`datatype''"
	
	foreach var of local list_`datatype' {
		
		local ++i
		di "`i' `var' `source'"
		
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
		reg `var' if treatment==0 `condition' [pw=`weight']
		local `i'_mean = round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
		local `i'_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-1)))
		

		matrix logit_`table'[`logit_count'+1, 1] = ``i'_mean'
		matrix logit_`table'[`logit_count'+2, 1] = ``i'_sd'
		
		if $ALT==1 {
		matrix logit_`table'_alt[`logit_count_alt'+1, 1] = ``i'_mean'
		matrix logit_`table'_alt[`logit_count_alt'+1, 2] = ``i'_sd'
		}	
		
		*RF - OLS
		reg `var' treatment `controls' ``var'_controls' if 1==1 `condition' [pw=`weight'], cluster(household_id)
		
		matrix postest=r(table)
		matrix list postest
			
		local `i'_itt_beta = round(_b[treatment],10^(min(-2, int(log10(abs(_b[treatment])))-2)))
		local `i'_itt_se = round(_se[treatment], 10^(min(-2, int(log10(abs(_se[treatment])))-2)))
		local `i'_itt_p = round(postest[4,1], 0.01)
		
		local `i'_itt95_l =round(postest[5,1],  10^(min(-2, int(log10(abs(postest[5,1])))-2)))
		local `i'_itt95_h =round(postest[6,1],  10^(min(-2, int(log10(abs(postest[6,1])))-2)))
		
		
		if postest[4,1]<0.1 {
			local `i'_itt_p=round(postest[4,1], 0.001)
			
		}
		
		if postest[4,1]<0.001 {
			local `i'_itt_p=postest[4,1]
			
		}
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		
		matrix logit_`table'[`logit_count'+1, 2] = ``i'_itt_beta'
		matrix logit_`table'[`logit_count'+2, 2] = ``i'_itt_se'
		matrix logit_`table'[`logit_count'+3, 2] = ``i'_itt_p'

		if $ALT==1 {
		matrix logit_`table'_alt[`logit_count_alt'+1, 3] = ``i'_itt_beta'
		matrix logit_`table'_alt[`logit_count_alt'+1, 4] = ``i'_itt95_l'
		matrix logit_`table'_alt[`logit_count_alt'+1, 5] = ``i'_itt95_h'
		matrix logit_`table'_alt[`logit_count_alt'+1, 6] = ``i'_itt_p'
		}			
		
		*RF - LOGIT
		logit `var' treatment `controls' ``var'_controls' if 1==1 `condition' [pw=`weight'], cluster(household_id)
		margins, dydx(*) atmeans post // then we are going to inflate everything by 100
		
		matrix postest=r(table)
		matrix list postest
		
		local `i'_itt_lgt_mfx =  round(_b[treatment]*100,10^(min(-2, int(log10(abs(_b[treatment]*100)))-2)))
		local `i'_itt_lgt_se = round(_se[treatment]*100, 10^(min(-2, int(log10(abs(_se[treatment]*100)))-2)))
		local `i'_itt_lgt_p = round(postest[4,1], 0.01)
		
		local `i'_itt_lgt_95_l=round(postest[5,1]*100,  10^(min(-2, int(log10(abs(postest[5,1]*100)))-2)))
		local `i'_itt_lgt_95_h=round(postest[6,1]*100,  10^(min(-2, int(log10(abs(postest[6,1]*100)))-2)))
		
		
		if postest[4,1]<0.1 {
			local `i'_itt_lgt_p=round(postest[4,1], 0.001)
			
		}
		
		if postest[4,1]<0.001 {
			local `i'_itt_lgt_p=postest[4,1]
			
		}
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		matrix logit_`table'[`logit_count'+1, 3] = ``i'_itt_lgt_mfx'
		matrix logit_`table'[`logit_count'+2, 3] = ``i'_itt_lgt_se'
		matrix logit_`table'[`logit_count'+3, 3] = ``i'_itt_lgt_p'	
		
		if $ALT==1 {
		matrix logit_`table'_alt[`logit_count_alt'+1, 7] = ``i'_itt_lgt_mfx'
		matrix logit_`table'_alt[`logit_count_alt'+1, 8] = ``i'_itt_lgt_95_l'
		matrix logit_`table'_alt[`logit_count_alt'+1, 9] = ``i'_itt_lgt_95_h'
		matrix logit_`table'_alt[`logit_count_alt'+1, 10] = ``i'_itt_lgt_p'
		}
		
		local logit_count_alt = `logit_count_alt'+1
		if "`var'"!="" {
		local logit_temp_alt "`logit_temp_alt' `var'"
		}


		local logit_count = `logit_count'+3
		if "`var'"!="" {
		local logit_temp "`logit_temp' `var' se p"
		}
		
		} //end of var loop

	local logit_count = 0 

	local logit_count_alt = 0
	} // end of datatype loop

local source ""
	

******************************************************
******************************************************

* Displaying table

matrix rownames logit_`table' = `logit_temp'

matrix rownames logit_`table'_alt = `logit_temp_alt'

noi: matrix list logit_`table', title("Table logit_`table': Logit, Probit, IVProbit MFX")
noi: display "Sample size for survey data analysis is `Table`table'_N_survey'"
noi: display "Sample size for administrative data analysis is `Table`table'_N_admin'"

******************************************************
******************************************************




