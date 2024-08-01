

* ***********************************************
* Analysis comparing treatment and controls
* ***********************************************


***************************
*Set up matrix for results
*keep household_id res_person_id treatment ohp_all_ever_30sep2009 `list' nnn*

 * inflate 0/1 variables by 100
 local inflate_list "`list_binary' phq_prob"

local list "`list"
local source ""

local numvar : word count `list_vars'
di "`numvar'"

matrix define logit_`table' = J(`numvar'*3,2,.)
matrix colnames logit_`table' = "RF" "LogitME" 

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


	foreach var of local list_vars {
		
		local ++i
		di "`i' `var' `source'"
		
		* Binary variables (and probabilities) gets inflated by 100
			di  "`i' `var' gets inflated by 100"
			sum `var'_`date'
		        replace `var'_`date'=100*`var'_`date'
			sum `var'_`date'
			local inflate_`i'=1
	
		*Control means
		*di "reg `var'_`date' if treatment==0 `condition' [pw=`weight_`date'']"
		*reg `var'_`date' if treatment==0 `condition' [pw=`weight_`date'']
		*local `i'_mean = round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-2)))
		*local `i'_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-2)))
		
		*matrix logit_`table'[`logit_count'+1, 1] = ``i'_mean'
		*matrix logit_`table'[`logit_count'+2, 1] = ``i'_sd'
		
		*if $ALT==1 {
		*matrix logit_`table'_alt[`logit_count_alt'+1, 1] = ``i'_mean'
		*matrix logit_`table'_alt[`logit_count_alt'+1, 2] = ``i'_sd'
		*}	
		
		*RF - OLS
		di "reg `var'_`date' treatment `controls' ``var'_`date'_controls' if 1==1 `condition' [pw=`weight_`date''], cluster(household_id)"
		reg `var'_`date' treatment `controls' ``var'_`date'_controls' if 1==1 `condition' [pw=`weight_`date''], cluster(household_id)
		
		matrix postest=r(table)
		*matrix list postest
			
		local `i'_itt_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
		local `i'_itt_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
		local `i'_itt_p = round(postest[4, 1], 10^(min(-3, int(log10(abs(postest[4,1])))-2)))
		
		local `i'_itt95_l =round(postest[5,1],  10^(min(-2, int(log10(abs(postest[5,1])))-2)))
		local `i'_itt95_h =round(postest[6,1],  10^(min(-2, int(log10(abs(postest[6,1])))-2)))
		
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		
		matrix logit_`table'[`logit_count'+1, 1] = ``i'_itt_beta'
		matrix logit_`table'[`logit_count'+2, 1] = ``i'_itt_se'
		matrix logit_`table'[`logit_count'+3, 1] = ``i'_itt_p'

		if $ALT==1 {
		matrix logit_`table'_alt[`logit_count_alt'+1, 3] = ``i'_itt_beta'
		matrix logit_`table'_alt[`logit_count_alt'+1, 4] = ``i'_itt95_l'
		matrix logit_`table'_alt[`logit_count_alt'+1, 5] = ``i'_itt95_h'
		matrix logit_`table'_alt[`logit_count_alt'+1, 6] = ``i'_itt_p'
		}			
		
		*RF - LOGIT
		di "logit `var'_`date' treatment `controls' ``var'_`date'_controls' if 1==1 `condition' [pw=`weight_`date''], cluster(household_id)"
		logit `var'_`date' treatment `controls' ``var'_`date'_controls' if 1==1 `condition' [pw=`weight_`date''], cluster(household_id)
		margins, dydx(*) atmeans post // then we are going to inflate everything by 100
		
		matrix postest=r(table)
		*matrix list postest
		
		local `i'_itt_lgt_mfx =  round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
		local `i'_itt_lgt_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
		local `i'_itt_lgt_p = round(postest[4,1], 0.01)
		
		local `i'_itt_lgt_95_l=round(postest[5,1],  10^(min(-2, int(log10(abs(postest[5,1])))-2)))
		local `i'_itt_lgt_95_h=round(postest[6,1],  10^(min(-2, int(log10(abs(postest[6,1])))-2)))
		
		
		if postest[4,1]<0.1 {
			local `i'_itt_lgt_p=round(postest[4,1], 0.001)
			
		}
		
		if postest[4,1]<0.001 {
			local `i'_itt_lgt_p=postest[4,1]
			
		}
		
		matrix define postest=J(1,1,.) // clear postest matrix
		
		matrix logit_`table'[`logit_count'+1, 2] = ``i'_itt_lgt_mfx'*100
		matrix logit_`table'[`logit_count'+2, 2] = ``i'_itt_lgt_se'*100
		matrix logit_`table'[`logit_count'+3, 2] = ``i'_itt_lgt_p'	
		
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


local source ""
	

******************************************************
******************************************************

* Displaying table

matrix rownames logit_`table' = `logit_temp'

if $ALT==1 {
matrix rownames logit_`table'_alt = `logit_temp_alt'
}

noi: matrix list logit_`table', title("Table logit_`table': Logit, Probit, IVProbit MFX")
noi: display "Sample size for survey data analysis is `Table`table'_N_survey'"
noi: display "Sample size for administrative data analysis is `Table`table'_N_admin'"

******************************************************
******************************************************




