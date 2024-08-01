

* ***********************************************
* Control means and standard errors
* ***********************************************

      local add_vars ""

      ***************************
      *Set up matrix for results

      local results = "Table`table'"

      local numvar : word count  `list'
      di "`numvar'"

if "`shape'" == "wide"{
	local rows = `numvar'
	matrix define `results' = J(`rows', 2, .)
	matrix colnames `results' = "control-means" "std dev"
	}

if "`shape'" == "long"{
	local rows = 2*`numvar'+1
	matrix define `results' = J(`rows', 1, .)
	matrix colnames `results' = "control-means" 
	}
	
      *******************************
      * Sample size for table notes

      count if 1==1 `condition'
      local Table`table'_N = r(N)


      ***********************************
      * Control means and balance of treaments and controls 
			
		local temp  ""
		local count = 1

		foreach var of local list {															
			reg `var' if treatment==0 `condition' [pw=`weight']
			local `var'_mean = round(_b[_cons]*1000)/1000
			local `var'_sd = round(e(rmse)*1000)/1000
			
			if substr("``var'_mean'",1,3)==".00" | substr("``var'_mean'",1,4)=="-.00" {
			local `var'_mean = _b[_cons]
			}
			
			if substr("``var'_sd'",1,3)==".00" {
			local `var'_sd = e(rmse)
			}

	   if "`shape'" == "wide"{
			matrix `results'[`count', 1] = ``var'_mean'
			matrix `results'[`count', 2] = ``var'_sd'
	
			local count = `count' + 1
			di `count'
			
		local temp "`temp' `var'"
		}

	if "`shape'" == "long"{
			matrix `results'[`count', 1] = ``var'_mean'
			matrix `results'[`count'+1, 1] = ``var'_sd'
	
			local count = `count' + 2
			di `count'
			
		local temp "`temp' `var' sd"
		}

	}

if "`shape'" == "long"{
	matrix `results'[`count', 1] = `Table`table'_N' 
	local temp "`temp' N"
	}
	
		*****************************
		* Displaying table
		
      matrix rownames `results' = `temp'
      noi: matrix list `results', title("Table `table': Control means and standard deviations")
      noi: display "Sample size for analysis is `Table`table'_N'"










