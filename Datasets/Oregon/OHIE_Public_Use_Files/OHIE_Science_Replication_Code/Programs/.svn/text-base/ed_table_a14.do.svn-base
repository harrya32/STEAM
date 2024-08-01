

***********************************
*LATE for different subpopulations
		
	local resultsB = "Table`table'_`date'"

local numvar : word count `list'
local columns = `numvar'*2+4
di "`columns'"

local numpop : word count `subpops'
local rows= `numpop'*5
di "`rows'"

matrix define `resultsB' = J(`rows', `columns', .)

matrix colnames `resultsB' = `list_names'

matrix list `resultsB'
di `subpop_namesB'
matrix rownames `resultsB'= `subpop_namesB'

local temp  ""
local i = 0

local row=1

*weights
local weight=1
		

foreach subpop in `subpops' { 
	local count = 1
	
	local weight=1
	local subsample " " 
	
	if regexm("`subpop'", "white")==1 | regexm("`subpop'", "morehs")==1 | regexm("`subpop'", "smoke")==1 | regexm("`subpop'", "dx_prelot") {
		local weight "weight_inp"
		local subsample "& inp_survey==1"
		} // End regexm weights loop
	
	*sample counts
	di "Sample Count"
		count if `subpop'==1 & `weight'!=0
		matrix `resultsB'[`row',`count']= r(N)
		
	*first stage (subpop)
	
		di "reg `fs_var' treatment `controls' if `subpop'==1 `subsample' [pw=`weight'], cluster(household_id)"
		reg `fs_var' treatment `controls' if `subpop'==1 `subsample' [pw=`weight'], cluster(household_id)
		local fs_`subpop' = round(_b[treatment], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		
		matrix `resultsB'[`row',`count'+1] = `fs_`subpop''
		
		*local subsample "& `subpop'==1"

	if regexm("`subpop'", "all")!=1 {
		
		*sample counts (complement)
		count if `subpop'!=1 & !missing(`subpop') & `weight'!=0
		matrix `resultsB'[`row'+2,`count']= r(N)
		
		*first stage (complement)
		di "reg `fs_var' treatment `controls' if `subpop'==0 `subsample' [pw=`weight'], cluster(household_id)"
		reg `fs_var' treatment `controls' if `subpop'==0 `subsample' [pw=`weight'], cluster(household_id)
		local fs_`subpop' = round(_b[treatment], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		
		matrix `resultsB'[`row'+2,`count'+1] = `fs_`subpop''

				} // End regexm loop (sample count and first stage)
		
	*generate the interaction variables		
		foreach var in `list' {
		gen con_`var'_`subpop'=(`var'_09mar2008*`subpop')
		gen con_`var'_`subpop'_m=(`var'_09mar2008_m*`subpop')
		local `var'_pop_controls "con_`var'_`subpop' con_`var'_`subpop'_m"
				}
				
	*generate hh size interactions for controls
		gen num2_`subpop'=(nnnnumhh_li_2*`subpop')
		gen num3_`subpop'=(nnnnumhh_li_3*`subpop')
		local hhinteract_`subpop'= "num2_`subpop' num3_`subpop'"
		
	*generate the treatment and insurance interactions
		gen treatment_`subpop'= treatment*`subpop'
		gen ohp_all_ever_`subpop'=ohp_all_ever_30sep2009*`subpop'
		
	local count=1
	
	foreach var of local list {
		
		local ++i
			
		*Control means*
		di "CONTROL MEAN"
		di "reg `var'_`date' if treatment==0 `condition' & `subpop'==1 `subsample' [pw=`weight']"
		reg `var'_`date' if treatment==0 `condition' & `subpop'==1 `subsample' [pw=`weight']
		local `i'_mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		local `i'_sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
	
		matrix `resultsB'[`row', `count'+2] = ``i'_mean'
		matrix `resultsB'[`row'+1, `count'+2] = ``i'_sd'
				
		*LATE (Coefficients for all subpops, se and p-val for "all" subgroup)
		di "ivregress 2sls `var'_`date' (ohp_all_ever_30sep2009=treatment) nnn* ``var'_`date'_controls' local if `subpop'==1 `subsample' [pw=`weight'], cluster(household_id)"
		ivregress 2sls `var'_`date' (ohp_all_ever_30sep2009=treatment) nnn* ``var'_`date'_controls' if `subpop'==1 `subsample' [pw=`weight'], cluster(household_id)
	
		local `i'_late_beta= round(_b[ohp_all_ever_30sep2009], .001)
		local `i'_late_se = round(_se[ohp_all_ever_`source'], 10^(min(-3, int(log10(abs(_se[ohp_all_ever_`source'])))-2)))
		matrix postest=r(table)
		local `i'_late_p = round(postest[4,1], 10^(min(-3, int(log10(abs(postest[4,1])))-2)))
		
		di ``i'_late_p'
		
		matrix `resultsB'[`row',`count'+3]= ``i'_late_beta'
		matrix `resultsB'[`row'+1, `count'+3]=``i'_late_se'
		matrix `resultsB'[`row',`count'+4]= ``i'_late_p'
	
		if regexm("`subpop'", "all")!=1 {
		
		
		*mean, coefficient, se, and p for subpop complement
		
		*Control means
		di "reg `var'_`date' if treatment==0 `condition' & `subpop'==0 `subsample' [pw=`weight']"
		reg `var'_`date' if treatment==0 `condition' & `subpop'==0 [pw=`weight']
		local `i'_mean = round(_b[_cons], 10^(min(-3, int(log10(abs(_b[_cons])))-2)))
		local `i'_sd = round(e(rmse), 10^(min(-3, int(log10(abs(e(rmse))))-2)))
	
		matrix `resultsB'[`row'+2, `count'+2] = ``i'_mean'
		matrix `resultsB'[`row'+3, `count'+2] = ``i'_sd'
		
		di "ivregress 2sls `var'_`date' (ohp_all_ever_30sep2009=treatment) nnn* ``var'_`date'_controls' if `subpop'==0 `subsample' [pw=`weight'], cluster(household_id)"
		ivregress 2sls `var'_`date' (ohp_all_ever_30sep2009=treatment) nnn* ``var'_`date'_controls' if `subpop'==0 `subsample' [pw=`weight'], cluster(household_id)
	
		local `i'_late_beta= round(_b[ohp_all_ever_30sep2009], .001)
		local `i'_late_se = round(_se[ohp_all_ever_`source'], 10^(min(-3, int(log10(abs(_se[ohp_all_ever_`source'])))-2)))
		
		matrix postest=r(table)	
		local `i'_late_p=round(postest[4, 1], 10^(min(-3, int(log10(abs(postest[4,1])))-2)))
	
		matrix `resultsB'[`row'+2,`count'+3]= ``i'_late_beta'
		matrix `resultsB'[`row'+3, `count'+3]=``i'_late_se'
		matrix `resultsB'[`row'+2, `count'+4]=``i'_late_p'

		matrix list `resultsB'
		
		*p-val of difference
		di "ivregress 2sls `var'_`date' (ohp_all_ever_30sep2009 ohp_all_ever_`subpop'= treatment treatment_`subpop') nnn* `hhinteract_`subpop'' ``var'_`date'_controls' ``var'_pop_controls' `subpop' if !missing(`subpop') [pweight=`weight'], cluster(household_id)"
				
		ivregress 2sls `var'_`date' (ohp_all_ever_30sep2009 ohp_all_ever_`subpop' = treatment treatment_`subpop') nnn* `hhinteract_`subpop'' ///
				``var'_`date'_controls' ``var'_pop_controls' `subpop' if !missing(`subpop') `subsample' [pweight=`weight'], cluster(household_id)
			matrix postest=r(table)
	
		local `i'_diff_p = round(postest[4,2], 0.01)
		
		local `i'_interact_beta=round(_b[ohp_all_ever_30sep2009],.001)
		local `i'_combined_beta= round(_b[ohp_all_ever_30sep2009]+_b[ohp_all_ever_`subpop'], .001)
		
			matrix `resultsB'[`row'+4,`count'+3]=``i'_diff_p'
		
		
		} // end regexm loop
		
			
			local count=`count'+3
			
				} // End of variables loop
	
	if regexm("`subpop'", "all")!=1 {
		local row=`row'+5
		}
		
	if regexm("`subpop'", "all")==1 {
		local row=`row'+3
		}
		
		} // End of subpop loop
		
		matrix list `resultsB'
		
		
