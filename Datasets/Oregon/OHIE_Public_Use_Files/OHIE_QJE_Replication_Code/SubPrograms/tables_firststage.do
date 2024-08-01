

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


		preserve
		drop ohp_all_end_survey0m ohp_all_end_survey6m

	**********************************
	* Run first stage estimates

  local temp ""
  local i = 1


	foreach yvar of local list{
		
		reg `yvar' if `sample'==1 & treatment == 0 [pw=`weight']
		local `sample'_`i'_mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		
		
		matrix `results'[3*`i'-2,1] = ``sample'_`i'_mean'
		
		reg `yvar' treatment `controls' if `sample' == 1 [pw=`weight'], cluster(household_id)
		local `sample'_`i' = round(_b[treatment],10^(min(-3, int(log10(abs(_b[treatment])))-2)))
		local `sample'_`i'_se = round(_se[treatment], 10^(min(-3, int(log10(abs(_se[treatment])))-2)))
		
		matrix `results'[3*`i'-2,2] = ``sample'_`i''
		matrix `results'[3*`i'-1,2] = ``sample'_`i'_se'

	   testparm treatment
	   local `sample'_`i'_F = round(r(F),0.001)
	   
	   if ``sample'_`i'_F' == 0 {
	   local `sample'_`i'_F = r(F)
	   } 
	   
   	matrix `results'[3*`i',2] = ``sample'_`i'_F'
	
   	local temp "`temp' `yvar' se F"
	   local i = `i'+1
			
		}
	
	count if `sample' == 1 & `weight'!=0
	local `sample'_N = r(N)
	matrix `results'[3*`i'-2,2] = ``sample'_N'
   local temp "`temp' N"

matrix rownames `results' = `temp'

noi: matrix  list `results', title("Table `table': First stage estimates") format(%9.3g)


	restore


	



