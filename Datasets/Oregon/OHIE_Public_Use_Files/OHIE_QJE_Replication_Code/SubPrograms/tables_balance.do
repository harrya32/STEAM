

* ***********************************************
* Balance of treatment and controls
* ***********************************************

     ***************************
      *Set up matrix for results
      keep household_id person_id treatment constant `balance_list' `balance_matchvar' `balance_controls' `balance_weight' `add_vars'
      local add_vars ""

      ***************************
      *Set up matrix for results

      local results = "Table`balance_table'"

      local numvar : word count `balance_matchvar' `balance_list'
      di "`numvar'"

      local rows = 3*(`numvar')+3

      if "`balance_subset'" != "`balance_list'"{
	      local rows = `rows'+3
	      }

		matrix define `results' = J(`rows', 2, .)
		matrix colnames `results' = "control-means" "treat-control-diff"

		
      *******************************
      * Sample size for table notes

      count if `balance_weight'!=0 `balance_condition'
      local Table`balance_table'_N = r(N)


      ***********************************
      * Control means and balance of treaments and controls 
			
		local temp  ""
		local count = 0

		* Matching variable (response, match, etc);  not conditional
		foreach var of local balance_matchvar{
				reg `var' if treatment==0 [pw=`balance_weight']
				local `var'_mean =round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
				local `var'_sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
				
				
				reg `var' treatment `balance_controls' ``var'_controls' [pw=`balance_weight'], cluster(household_id)
				local `var'_d_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
				local `var'_d_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
				local `var'_d_p = round(2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))*1000)/1000
				
				
				if ``var'_d_p' == 0 {
				local `var'_d_p = 2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))
				}
				
				matrix `results'[`count'+1, 1] = ``var'_mean'
				matrix `results'[`count'+2, 1] = ``var'_sd'
				
				matrix `results'[`count'+1, 2] = ``var'_d_beta'
				matrix `results'[`count'+2, 2] = ``var'_d_se'
				matrix `results'[`count'+3, 2] = ``var'_d_p'
				
				local count = `count' + 3
				di `count'
				
				local temp "`temp' `var' se p"
				}

		* Covariates; conditional on being in sample
		foreach var of local balance_list {															
			reg `var' if treatment==0 `balance_condition' [pw=`balance_weight']
			local `var'_mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
			local `var'_sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
			

			
			reg `var' treatment `balance_controls' ``var'_controls' if 1==1 `balance_condition' [pw=`balance_weight'], cluster(household_id)
			local `var'_d_beta = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
			local `var'_d_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
			local `var'_d_p = round(2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))*1000)/1000
				

				if ``var'_d_p' == 0 {
				local `var'_d_p = 2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))
				}				
				
			matrix `results'[`count'+1, 1] = ``var'_mean'
			matrix `results'[`count'+2, 1] = ``var'_sd'
				
			matrix `results'[`count'+1, 2] = ``var'_d_beta'
			matrix `results'[`count'+2, 2] = ``var'_d_se'
			matrix `results'[`count'+3, 2] = ``var'_d_p'
			
			local count = `count' + 3
			di `count'
			
			local temp "`temp' `var' se p"
			}
			
	
		*************************
		*F-stats for joint test 
			
		local numtest : word count `balance_list' 
		di "`numtest'"

		* Expanding dataset for stacked regressions
      compress
      keep if 1==1 `balance_condition'
		expand `numtest'
		bys person_id: gen order = _n
		
		gen outcome = .				

		* Creating variables for stacked regressions

      local i = 0
      local ftest_string ""

		foreach var of local balance_list {
			local i = `i'+1
			di "`i' `var'"
			replace outcome = `var' if (order == `i')
			gen treatment_`i' = treatment *(order == `i')
			
			foreach control of varlist constant `balance_controls' ``var'_controls'{
				gen X`i'`control' = `control' *(order == `i')
				}
			
			foreach word of local balance_subset{
				if "`var'"=="`word'"{
					local ftest_string "`ftest_string' treatment_`i'"
					di "`ftest_string'"
					}
				}
			}

* Running stacked regressions and f-tests
reg outcome treatment_* X* if 1==1 `balance_condition' [pw=`balance_weight'], cluster(household_id) nocons

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

testparm `ftest_string'
local joint_fstat_subset = round(r(F),.001)
local joint_pval_subset = round(r(p),.001)

	if `joint_fstat_subset' == 0 {
		local joint_fstat_subset = r(F)
		}

	if `joint_pval_subset' == 0 {
		local joint_pval_subset = r(p)
		}

if "`balance_subset'" != "`balance_list'"{
	matrix `results'[`count'+1, 2] = `joint_fstat_subset'
	matrix `results'[`count'+2, 2] = `joint_pval_subset'
	matrix `results'[`count'+3, 2] = `Table`balance_table'_N'
	local count = `count'+3
	local temp "`temp' joint_F_sub joint_p_sub N_sub"
	}

matrix `results'[`count'+1, 2] = `joint_fstat'
matrix `results'[`count'+2, 2] = `joint_pval'
matrix `results'[`count'+3, 2] = `Table`balance_table'_N'


		local temp "`temp' joint_F joint_p N"
		matrix rownames `results' = `temp'

      foreach var of local balance_list {
	local `var'_controls ""
	}

		*****************************
		* Displaying table
		
      noi: matrix list `results', title("Table `balance_table': Assessing potential non response bias: Balance of treatment and controls")
      noi: display "Sample size for analysis is `Table`balance_table'_N'"





