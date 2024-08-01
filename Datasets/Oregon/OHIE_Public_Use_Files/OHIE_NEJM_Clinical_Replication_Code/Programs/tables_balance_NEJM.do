

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

		matrix define `results' = J(`rows', 3, .)
		matrix colnames `results' = "control-means" "treat-means" "treat-control-diff"
		
		if $ALT==1 {
		local alt_rows= `numvar'+3 
		if "`balance_subset'" != "`balance_list'"{
		local alt_rows = `alt_rows'+3
		}
		matrix `balance_table'_alt = J(`alt_rows',3,.)
		matrix colnames `balance_table'_alt = "c-mean" "t-mean" "pvalue"

}

		
      *******************************
      * Sample size for table notes

      count if `balance_weight'!=0 `balance_condition'
      local Table`balance_table'_N = r(N)

      * inflate 0/1 variables by 100
	local inflate_list "`list_binary' phq_prob"
	
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
				
				reg `var' if treatment==0 [pw=`balance_weight']
				local `var'_C_mean =round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
				local `var'_C_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-1)))
				
				reg `var' if treatment==1 [pw=`balance_weight']
				local `var'_T_mean =round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
				local `var'_T_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-1)))
	
				reg `var' treatment `balance_controls' ``var'_controls' [pw=`balance_weight'], cluster(household_id)
				
				matrix postest=r(table)
				matrix list postest
			
				local `var'_d_beta = round(_b[treatment],10^(min(-2, int(log10(abs(_b[treatment])))-2)))
				local `var'_d_se = round(_se[treatment], 10^(min(-2, int(log10(abs(_se[treatment])))-2)))
				local `var'_d_p = round(postest[4,1], 0.01)
				
				local `var'_d95_l=round(postest[5,1],  10^(min(-2, int(log10(abs(postest[5,1])))-2)))
				local `var'_d95_h=round(postest[6,1],  10^(min(-2, int(log10(abs(postest[6,1])))-2)))
				
				
				if postest[4,1]<0.1 {
					local `var'_d_p = round(postest[4,1], 0.001)
					}
			
				if postest[4,1]<0.001 {
					local `var'_d_p = postest[4,1]
					}
				
				matrix define postest=J(1,1,.) // clear postest matrix	
				
				matrix `results'[`count'+1, 1] = ``var'_C_mean'
				matrix `results'[`count'+2, 1] = ``var'_C_sd'
			
				matrix `results'[`count'+1, 2] = ``var'_T_mean'
				matrix `results'[`count'+2, 2] = ``var'_T_sd'
	
				matrix `results'[`count'+1, 3] = ``var'_d_beta'
				matrix `results'[`count'+2, 3] = ``var'_d_se'
				matrix `results'[`count'+3, 3] = ``var'_d_p'
				
				if $ALT==1 {
				matrix `balance_table'_alt[`count_alt'+1, 1] = ``var'_C_mean'
				matrix `balance_table'_alt[`count_alt'+1, 2] = ``var'_T_mean'
				matrix `balance_table'_alt[`count_alt'+1, 3] = ``var'_d_p'
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

		* Covariates; conditional on being in sample
		foreach var of local balance_list {	
		
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
		 														
			reg `var' if treatment==0 `balance_condition' [pw=`balance_weight']
			local `var'_C_mean = round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
			local `var'_C_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-1)))
			
			reg `var' if treatment==1 `balance_condition' [pw=`balance_weight']
			local `var'_T_mean = round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
			local `var'_T_sd = round(e(rmse), 10^(min(-1, int(log10(abs(e(rmse))))-1)))

			
			reg `var' treatment `balance_controls' ``var'_controls' if 1==1 `balance_condition' [pw=`balance_weight'], cluster(household_id)
			
			matrix postest=r(table)
			matrix list postest
			
			local `var'_T_mean = round(_b[treatment] + ``var'_C_mean',10^(-1))
			
			local `var'_d_beta = round(_b[treatment],10^(min(-2, int(log10(abs(_b[treatment])))-2)))
			local `var'_d_se = round(_se[treatment], 10^(min(-2, int(log10(abs(_se[treatment])))-2)))
			local `var'_d_p = round(postest[4,1], 0.01)
			
			local `var'_d95_l=round(postest[5,1],  10^(min(-2, int(log10(abs(postest[5,1])))-2)))
			local `var'_d95_h=round(postest[6,1],  10^(min(-2, int(log10(abs(postest[6,1])))-2)))
				

				if postest[4,1]<0.1 {
					local `var'_d_p = round(postest[4,1], 0.001)
					}
			
				if postest[4,1]<0.001 {
					local `var'_d_p = postest[4,1]
					}	
				
			matrix define postest=J(1,1,.) // clear postest matrix				
				
			matrix `results'[`count'+1, 1] = ``var'_C_mean'
			matrix `results'[`count'+2, 1] = ``var'_C_sd'

			matrix `results'[`count'+1, 2] = ``var'_T_mean'
			matrix `results'[`count'+2, 2] = ``var'_T_sd'
				
			matrix `results'[`count'+1, 3] = ``var'_d_beta'
			matrix `results'[`count'+2, 3] = ``var'_d_se'
			matrix `results'[`count'+3, 3] = ``var'_d_p'
			
			if $ALT==1 {
			    matrix `balance_table'_alt[`count_alt'+1, 1] = ``var'_C_mean'
				matrix `balance_table'_alt[`count_alt'+1, 2] = ``var'_T_mean'
				matrix `balance_table'_alt[`count_alt'+1, 3] = ``var'_d_p'
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
	local `var'_s_beta = round(_b[treatment_`i'],10^(min(-2, int(log10(abs(_b[treatment_`i'])))-2)))
	di "``var'_d_beta'"
	di "``var'_s_beta'"
	assert round(``var'_d_beta',10^(min(-2, int(log10(abs(``var'_d_beta')))-2))) == round(``var'_s_beta',10^(min(-2, int(log10(abs(``var'_s_beta')))-2)))
	}

testparm treatment_* 

local joint_fstat = round(r(F),.01) 
local joint_pval = round(r(p),.01)

	if `joint_fstat' == 0 {
		local joint_fstat = r(F)
		}

	if r(p)<0.1 {
		local joint_pval = round(r(p),.001)
		}
	if r(p)<0.001 {
		local joint_pval = r(p)
		}

testparm `ftest_string'
local joint_fstat_subset = round(r(F),.01)
local joint_pval_subset = round(r(p),.01)

* test
assert round(Ftail(r(df), r(df_r), r(F)), .001)==round(r(p),.001)


	if `joint_fstat_subset' == 0 {
		local joint_fstat_subset = r(F)
		}

	if r(p)<0.1 {
		local joint_pval_subset = round(r(p),.001)
		}
	if r(p)<0.001 {
		local joint_pval_subset = r(p)
		}

if "`balance_subset'" != "`balance_list'"{
	matrix `results'[`count'+1, 3] = `joint_fstat_subset'
	matrix `results'[`count'+2, 3] = `joint_pval_subset'
	matrix `results'[`count'+3, 3] = `Table`balance_table'_N'
	local count = `count'+3
	local temp "`temp' joint_F_sub joint_p_sub N_sub"
	}

matrix `results'[`count'+1, 3] = `joint_fstat'
matrix `results'[`count'+2, 3] = `joint_pval'
matrix `results'[`count'+3, 3] = `Table`balance_table'_N'

local temp "`temp' joint_F joint_p N"
matrix rownames `results' = `temp'
		
		if $ALT==1 {
		
			if "`balance_subset'" != "`balance_list'"{
				matrix `balance_table'_alt[`count_alt'+1, 3] = `joint_fstat_subset'
				matrix `balance_table'_alt[`count_alt'+2, 3] = `joint_pval_subset'
				matrix `balance_table'_alt[`count_alt'+3, 3] = `Table`balance_table'_N'
				
				local count_alt=`count_alt'+3
				local temp_alt "`temp_alt' joint_F_sub joint_p_sub N_sub"
			}
			
			matrix `balance_table'_alt[`count_alt'+1, 3] = `joint_fstat'
			matrix `balance_table'_alt[`count_alt'+2, 3] = `joint_pval'
			matrix `balance_table'_alt[`count_alt'+3, 3] = `Table`balance_table'_N'
			
			local temp_alt "`temp_alt' joint_F joint_p N"
			
		}


		matrix rowname `balance_table'_alt =`temp_alt'


      foreach var of local balance_list {
	local `var'_controls ""
	}

		*****************************
		* Displaying table
		
      noi: matrix list `results', title("Table `balance_table': Assessing potential non response bias: Balance of treatment and controls")
      noi: display "Sample size for analysis is `Table`balance_table'_N'"










