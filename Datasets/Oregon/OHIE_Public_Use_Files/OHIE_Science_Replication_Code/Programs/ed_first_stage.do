

* ***********************************************
* First stage estimates
* ***********************************************
	
      ***************************
      *Set up matrix for results

      local results = "Table`table'_`date'"

      local numvar : word count `list' 
      local rows = 3*(`numvar')+1

      matrix define `results' = J(`rows', 2, .)
		matrix colnames `results' = "c-mean" "`sample'" 


if $ALT==1 {
local alt_rows= `numvar'
matrix `table'_alt = J(`alt_rows',5,.)
matrix colnames `table'_alt = "c_mean" "sample" "l95" "h95" "pvalue"

}
	
	**********************************
	* Run first stage estimates

  local temp ""
  local i = 1
  
if $ALT==1 {
		local count_alt = 0
		local temp_alt ""
		}


	foreach yvar of local list{
	
	if regexm("`yvar'", "ohp_all_mo_30sep2009")!=1 {
		replace `yvar'=100*`yvar'
			}
		
		reg `yvar' if `sample'==1 & treatment == 0 [pw=`weight']
		local `sample'_`i'_mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		
		
		matrix `results'[3*`i'-2,1] = ``sample'_`i'_mean'
		
		reg `yvar' treatment `controls' if `sample' == 1 [pw=`weight'], cluster(household_id)
		local `sample'_`i' = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
		local `sample'_`i'_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
		
		matrix `results'[3*`i'-2,2] = ``sample'_`i''
		matrix `results'[3*`i'-1,2] = ``sample'_`i'_se'

	   testparm treatment
	   local `sample'_`i'_pval = round(r(p),10^(min(-3, int(log10(abs(r(p))))-2)))
		
	   if ``sample'_`i'_pval' == 0 {
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


