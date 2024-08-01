
* ***********************************************
* Characteristics of compliers
* ***********************************************

         ***************************
      	*Set up matrix for results

      	local results = "Table`table'"

      	local numvar : word count `list'
      	di "`numvar'"

		local rows = (`numvar')
		
		matrix define Table`table' = J(`rows', 2, .)
		matrix colnames Table`table' = "sample-means" "complier-means" 
		matrix rownames Table`table' = `list'
	
	
	  	***********************************
      	* First stage in full sample
      
    	reg `insurance' treatment `controls' if `sample' == 1  [pw=`weight'], cluster(household_id)
		local overall_fs = round(_b[treatment], 10^(min(-2, int(log10(abs(_b[treatment])))-2)))
  
	 	* First stage by Xs
	 	
	 	local count = 1
		foreach group of local list{
	
			reg `group' if `sample' == 1  [pw=`weight']
			local `group'_mean = _b[_cons]
			
			reg `insurance' treatment `controls' if `sample' == 1 & `group' == 1 [pw=`weight'], cluster(household_id)
			local `group'_fs = _b[treatment]
		
			local `group'_ratio = (``group'_fs' / `overall_fs')
			
			local `group'_complier_mean = round((``group'_ratio' * ``group'_mean')*100,.1)
			local `group'_mean = round(``group'_mean'*100,.1)
			
			matrix Table`table'[`count', 1] = ``group'_mean'
			matrix Table`table'[`count', 2] = ``group'_complier_mean'
		
			local count = `count' + 1
			di `count'
			}


		*****************************
		* Displaying table
		
      	noi: matrix list Table`table', title("Table `table': Characteristics of compliers")

