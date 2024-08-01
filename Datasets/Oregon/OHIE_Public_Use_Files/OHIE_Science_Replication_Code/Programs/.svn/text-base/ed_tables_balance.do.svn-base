
assert _N==74922
	
* ***********************************************
* Balance of treatment and controls
* ***********************************************
		 
     ***************************
      *Set up matrix for results
      keep household_id person_id treatment constant `balance_list' `balance_matchvar' `balance_controls' `balance_weight' `add_vars'
      *local add_vars ""

      ***************************
      *Set up matrix for results
      
      local results = "Table`balance_table'_`date'"
      di "`results'"
      local numvar : word count `balance_matchvar' `balance_list'
      di "`numvar'"

      local rows = 3*(`numvar')+4

      if "`balance_subset'" != "`balance_list'"{
	      local rows = `rows'+4
	      }

		matrix define `results' = J(`rows', 2, .)
		matrix colnames `results' = "c_means" "treat_control_diff"
		
		if $ALT==1 {
		local alt_rows= `numvar'+3 
		if "`balance_subset'" != "`balance_list'"{
		local alt_rows = `alt_rows'+3
		}
		matrix `balance_table'_alt = J(`alt_rows',6,.)
		matrix colnames `balance_table'_alt = "c-mean" "sd" "treat-control-diff" "l95" "h95" "pvalue"

}

      *******************************
      * Sample size for table notes

      count if `weight'!=0 `balance_condition'
      local Table`balance_table'_N = r(N)

      ***********************************
      * Control means and balance of treaments and controls 
			
		local temp  ""
		local count = 0
		
		if $ALT==1 {
		local count_alt = 0
		local temp_alt ""
		}


		* Matching variable (response, match, etc);  not conditional
		foreach var of local balance_matchvar{
				* inflate binary variable by 100
				replace `var'=100*`var'
				
				reg `var' if treatment==0 [pw=`weight']
				local `var'_mean =round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
				local `var'_sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
				
				
				reg `var' treatment `balance_controls' ``var'_controls' [pw=`weight'], cluster(household_id) 
				local `var'_d_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
				local `var'_d_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
				
				matrix postest=r(table)
				local `var'_d_p= round(postest[4,1],10^(min(-3, int(log10(abs(postest[4,1])))-2)))
				
				local `var'_d95_l=round(_b[treatment]-invttail(e(df_r), 0.025)*_se[treatment],  10^(min(-3, int(log10(abs(_b[treatment]-invttail(e(df_r), 0.025)*_se[treatment])))-2)))
				local `var'_d95_h=round(_b[treatment]+invttail(e(df_r), 0.025)*_se[treatment],  10^(min(-3, int(log10(abs(_b[treatment]+invttail(e(df_r), 0.025)*_se[treatment])))-2)))
				
				
				if ``var'_d_p' == 0 {
				local `var'_d_p = 2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))
				}
				
				matrix `results'[`count'+1, 1] = ``var'_mean'
				matrix `results'[`count'+2, 1] = ``var'_sd'
				
				matrix `results'[`count'+1, 2] = ``var'_d_beta'
				matrix `results'[`count'+2, 2] = ``var'_d_se'
				matrix `results'[`count'+3, 2] = ``var'_d_p'
				
				if $ALT==1 {
				matrix `balance_table'_alt[`count_alt'+1, 1] = ``var'_mean'
				matrix `balance_table'_alt[`count_alt'+1, 2] = ``var'_sd'
				matrix `balance_table'_alt[`count_alt'+1, 3] = ``var'_d_beta'
				matrix `balance_table'_alt[`count_alt'+1, 4] = ``var'_d95_l'
				matrix `balance_table'_alt[`count_alt'+1, 5] = ``var'_d95_h'
				matrix `balance_table'_alt[`count_alt'+1, 6] = ``var'_d_p'
				}
		
				local count = `count' + 3
				di `count'
				
				local count_alt = `count_alt'+1
				if "`var'"!="" {
				local temp_alt "`temp_alt' `var'"
				}
		
				local temp "`temp' `var' se p"
				replace `var'=`var'/100
				}

*
assert _N==74922

		* Covariates; conditional on being in sample
		foreach var of local balance_list {	
		
			* Binary variables (and probabilities) gets inflated by 100
			
			 local inflate_var: list var & inflate_list
			 if "`var'"=="`inflate_var'" {
				di  "`i' `var' gets inflated by 100"
				sum `var'
					replace `var'=100*`var'
				sum `var'
				
			 }
					 														
			reg `var' if treatment==0 `balance_condition' [pw=`weight'] //  
			local `var'_mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
			local `var'_sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
			

			
			reg `var' treatment `balance_controls' if 1==1 `balance_condition' [pw=`weight'], cluster(household_id) // took out ``var'_controls' 
			local `var'_d_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
			local `var'_d_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
			*local `var'_d_p = round(2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))*1000)/1000
			
			matrix postest=r(table)
			local `var'_d_p= round(postest[4,1],10^(min(-3, int(log10(abs(postest[4,1])))-2)))	
			
			local `var'_d95_l=round(_b[treatment]-invttail(e(df_r), 0.025)*_se[treatment],  10^(min(-3, int(log10(abs(_b[treatment]-invttail(e(df_r), 0.025)*_se[treatment])))-2)))
			local `var'_d95_h=round(_b[treatment]+invttail(e(df_r), 0.025)*_se[treatment],  10^(min(-3, int(log10(abs(_b[treatment]+invttail(e(df_r), 0.025)*_se[treatment])))-2)))
				
				if ``var'_d_p' == 0 {
				local `var'_d_p = 2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))
				}				
				
			matrix `results'[`count'+1, 1] = ``var'_mean'
			matrix `results'[`count'+2, 1] = ``var'_sd'
				
			matrix `results'[`count'+1, 2] = ``var'_d_beta'
			matrix `results'[`count'+2, 2] = ``var'_d_se'
			matrix `results'[`count'+3, 2] = ``var'_d_p'
			
			if $ALT==1 {
			        matrix `balance_table'_alt[`count_alt'+1, 1] = ``var'_mean'
				matrix `balance_table'_alt[`count_alt'+1, 2] = ``var'_sd'
				matrix `balance_table'_alt[`count_alt'+1, 3] = ``var'_d_beta'
				matrix `balance_table'_alt[`count_alt'+1, 4] = ``var'_d95_l'
				matrix `balance_table'_alt[`count_alt'+1, 5] = ``var'_d95_h'
				matrix `balance_table'_alt[`count_alt'+1, 6] = ``var'_d_p'
				}
		
			
			local count = `count' + 3
			di `count'
			
			local count_alt = `count_alt'+1
				if "`var'"!="" {
				local temp_alt "`temp_alt' `var'"
				}
		
			
			local temp "`temp' `var' se p"
			}
			

		*************************
		*F-stats for joint test 
			
		local numtest : word count `balance_list' 
		di "`numtest'"

		 *Expanding dataset for stacked regressions
      compress
      keep if 1==1 `balance_condition'
		expand `numtest'
		bys person_id: gen order = _n
		
		gen outcome = .				

		* Creating variables for stacked regressions

      local i = 0
      local ftest_string ""
      local prelottery_string ""

		foreach var of local balance_list {
			local i = `i'+1
			di "`i' `var'"
			replace outcome = `var' if (order == `i')
			gen treatment_`i' = treatment *(order == `i')
			
			foreach control of varlist `balance_controls' constant { // took out ``var'_controls'
				gen X`i'`control' = `control' *(order == `i')
				}
		
			foreach word of local balance_subset {
				if "`var'"=="`word'"{
					local ftest_string "`ftest_string' treatment_`i'"
					di "`ftest_string'"
					}
				}
			foreach word of local prelottery_vars {
				if "`word'"=="`var'"{
					local prelottery_string "`prelottery_string' treatment_`i'"
					di "`prelottery_string'"
					}		
				}


				}
			


* Running stacked regressions and f-tests
reg outcome treatment_* X* if 1==1 `balance_condition' [pw=`weight'], cluster(household_id) nocons

local i = 0
foreach var of local balance_list {
	local i = `i'+1
	di "`i' `var'"
	local `var'_s_beta = round(_b[treatment_`i'],10^(min(-3, int(log10(abs(_b[treatment_`i'])))-2)))
	di "``var'_d_beta'"
	di "``var'_s_beta'"
	assert round(``var'_d_beta',10^(min(-3, int(log10(abs(``var'_d_beta')))-2))) == round(``var'_s_beta',10^(min(-3, int(log10(abs(``var'_s_beta')))-2)))
	}

testparm treatment_* 

local joint_fstat = round(r(F),.001) 
local joint_pval = round(r(p),.001)

	if `joint_fstat' == 0 {
		local joint_fstat = r(F)
		}

	if `joint_pval' == 0 {
		local joint_pval = r(p)
		}

di "`ftest_string'"
	testparm `ftest_string'
	local joint_fstat_subset = round(r(F),.001)
	local joint_pval_subset = round(r(p),.001)
	


* test
assert round(Ftail(r(df), r(df_r), r(F)), .001)==round(r(p),.001)

di "`prelottery_string'"
testparm `prelottery_string'
local prelottery_fstat=round(r(F),.001)
local prelottery_pval=round(r(p),.001)


	if `joint_fstat_subset' == 0 {
		local joint_fstat_subset = r(F)
		}

	if `joint_pval_subset' == 0 {
		local joint_pval_subset = r(p)
		}
*/
if "`balance_subset'" != "`balance_list'"{
	matrix `results'[`count'+1, 2] = `joint_fstat_subset'
	matrix `results'[`count'+2, 2] = `joint_pval_subset'
	matrix `results'[`count'+3, 2] = `Table`balance_table'_N'
	local count = `count'+3
	local temp "`temp' joint_F_sub joint_p_sub N_sub"
	}

matrix `results'[`count'+1, 2] = `joint_fstat'
matrix `results'[`count'+2, 2] = `joint_pval'

matrix `results'[`count'+3, 2] = `prelottery_fstat'				
matrix `results'[`count'+4, 2] = `prelottery_pval'

matrix `results'[`count'+5, 2] = `Table`balance_table'_N'



local temp "`temp' joint_F joint_p prelottery_F prelottery_p N"
matrix rownames `results' = `temp'
	
	
		if $ALT==1 {
		
			if "`balance_subset'" != "`balance_list'"{
				matrix `balance_table'_alt[`count_alt'+1, 6] = `joint_fstat_subset'
				matrix `balance_table'_alt[`count_alt'+2, 6] = `joint_pval_subset'
				matrix `balance_table'_alt[`count_alt'+3, 6] = `Table`balance_table'_N'
				
				local count_alt=`count_alt'+3
				local temp_alt "`temp_alt' joint_F_sub joint_p_sub N_sub"
			}
			
			matrix `balance_table'_alt[`count_alt'+1, 6] = `joint_fstat'
			matrix `balance_table'_alt[`count_alt'+2, 6] = `joint_pval'
			matrix `balance_table'_alt[`count_alt'+3, 6] = `Table`balance_table'_N'
			
			local temp_alt "`temp_alt' joint_F joint_p N"
		
		matrix rowname `balance_table'_alt =`temp_alt' // this was outside the loop before and causing the code to break
		}


		


     /* foreach var of local balance_list {
	local `var'_controls ""
	}*/

		*****************************
		* Displaying table
      
      noi: matrix list `results', title("Table `balance_table': Assessing potential non response bias: Balance of treatment and controls")
      noi: display "Sample size for analysis is `Table`balance_table'_N'"





