clear matrix
cap program drop main

program main
    balance_test, condition(overlap_sample == 1) weight(weight_total_inp) ///
        balance_list(birthyear_list female_list english_list self_list first_day_list ///
        have_phone_list pobox_list)
    matrix list overlap_balance
    matrix_to_txt, mat(overlap_balance) saving(../Output/table_a4b_overlap_balance.txt) ///
        title("Appendix Table 4b: Balance in Overlap Sample") ///
        note(Note: -zip code median household income- is omitted, so the pooled F-statistic does ///
        not replicate the paper.) usec user replace 
end

program balance_test
    syntax, balance_list(str) weight(str) [condition(str)]
    local uselist "person_id household_id treatment *_list *_sample weight_total_inp"
    use `uselist' if `condition' using ../Data/data_for_analysis.dta, clear

    xi, prefix(hh) i.numhh_list

    local inflate_list "female_list english_list self_list first_day_list have_phone_list pobox_list"
    foreach inf_var in `inflate_list' {
        replace `inf_var' = 100 * `inf_var'
    }

    foreach var in `balance_list' {	
        *Control mean
        reg `var' if treatment == 0 [pw = `weight']
        local control_mean = _b[_cons]
        
        *Balance
        reg `var' treatment hh* [pw = `weight'], cluster(household_id)

        matrix M = r(table)
        matrix overlap_balance = nullmat(overlap_balance)\ ((`control_mean'\ .), ///
            (_b[treatment]\ _se[treatment]), (M[rownumb(M, "pvalue"), colnumb(M, "treatment")]\ .))
        local rowname "`rowname' `var' se"
    }
    matrix colnames overlap_balance = control_mean tc_diff p
    matrix rownames overlap_balance = `rowname'

    *Control and total N
    count if treatment == 0
    matrix overlap_N = r(N), _N
    matrix rownames overlap_N = N

    *Joint F-test
    local num : word count `balance_list' 
    expand `num'
    bys person_id: gen order = _n
    gen outcome = .
    gen constant = 1
    local i = 0
    foreach var in `balance_list' {
        local ++i
        replace outcome       = `var'    if order == `i'
        gen     treatment_`i' = treatment *(order == `i')
        foreach control of varlist hh* constant {
            gen X`i'`control' = `control' *(order == `i')
        }
    }
    reg outcome treatment_* X* [pw = `weight'], cluster(household_id) nocons
    testparm treatment_* 
    matrix f_stat = J(2,2,.), (r(F)\ r(p))
    matrix rownames f_stat = F_stat pvalue

    matrix overlap_balance = overlap_balance\ f_stat\ (overlap_N, .)
end

*Execute
main
