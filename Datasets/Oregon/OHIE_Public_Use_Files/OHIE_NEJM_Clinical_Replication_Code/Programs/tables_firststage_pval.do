

* ***********************************************
* First stage estimates
* ***********************************************
	
      ***************************
      *Set up matrix for results

      local results = "Table`table'"

      local numvar : word count `list' 
      local rows = 3*(`numvar')+1

      matrix define `results' = J(`rows', 2, .)
		matrix colnames `results' = "c-mean" "`sample'" 


if $ALT==1 {
local alt_rows= `numvar'
matrix `table'_alt = J(`alt_rows',5,.)
matrix colnames `table'_alt = "c-mean" "sample" "l95" "h95" "pvalue"

}
	
	
	 * inflate 0/1 variables by 100
	local inflate_list "`list_binary' phq_prob"
	
	**********************************
	* Run first stage estimates

  local temp ""
  local i = 1
  
if $ALT==1 {
		local count_alt = 0
		local temp_alt ""
		}


	foreach yvar of local list{
	
		* Binary variables (and probabilities) gets inflated by 100
		local inflate_`i'=0
		 local inflate_var: list yvar & inflate_list
		 if "`yvar'"=="`inflate_var'" {
			di  "`i' `yvar' gets inflated by 100"
			sum `yvar'
		        replace `yvar'=100*`yvar'
			sum `yvar'
			local inflate_`i'=1
		 }
		 
		
		reg `yvar' if `sample'==1 & treatment == 0 [pw=`weight']
		local `sample'_`i'_mean = round(_b[_cons], 10^(min(-1, int(log10(abs(_b[_cons])))-1)))
		
		
		matrix `results'[3*`i'-2,1] = ``sample'_`i'_mean'
		
		reg `yvar' treatment `controls' if `sample' == 1 [pw=`weight'], cluster(household_id)
		matrix postest=r(table)
		matrix list postest
			
		local `sample'_`i' = round(_b[treatment],10^(min(-2, int(log10(abs(_b[treatment])))-2)))
		local `sample'_`i'_se = round(_se[treatment], 10^(min(-2, int(log10(abs(_se[treatment])))-2)))
		
		local `sample'_`i'_95l=round(postest[5,1],  10^(min(-2, int(log10(abs(postest[5,1])))-2)))
		local `sample'_`i'_95h=round(postest[6,1],  10^(min(-2, int(log10(abs(postest[6,1])))-2)))
		
		matrix define postest=J(1,1,.) // clear postest matrix	
		
		
		matrix `results'[3*`i'-2,2] = ``sample'_`i''
		matrix `results'[3*`i'-1,2] = ``sample'_`i'_se'

	   testparm treatment
	   local `sample'_`i'_pval = round(r(p),0.01)
	   
	   if r(p)<0.1 {
		local `sample'_`i'_pval = round(r(p),0.001)
	   
	   }
	   
	   if r(p)<0.001 {
		local `sample'_`i'_pval = r(p)
	   }
	 
	   
   	matrix `results'[3*`i',2] = ``sample'_`i'_pval'
	
		if $ALT==1 {
			matrix `table'_alt[`count_alt'+1, 1] = ``sample'_`i'_mean'
			matrix `table'_alt[`count_alt'+1, 2] = ``sample'_`i''
			matrix `table'_alt[`count_alt'+1, 3] = ``sample'_`i'_95l'
			matrix `table'_alt[`count_alt'+1, 4] = ``sample'_`i'_95h'
			matrix `table'_alt[`count_alt'+1, 5] = ``sample'_`i'_pval'
		}
		
   	local temp "`temp' `yvar' se pval"
	   local i = `i'+1
	   
	   local count_alt = `count_alt'+1
		if "`yvar'"!="" {
		local temp_alt "`temp_alt' `yvar'"
		}

			
	}
	
	count if `sample' == 1 & `weight'!=0
	local `sample'_N = r(N)
	matrix `results'[3*`i'-2,2] = ``sample'_N'
   local temp "`temp' N"

matrix rownames `results' = `temp'
if $ALT==1 {
		matrix rowname `table'_alt =`temp_alt'
		}

noi: matrix  list `results', title("Table `table': First stage estimates") format(%9.3g)





	



