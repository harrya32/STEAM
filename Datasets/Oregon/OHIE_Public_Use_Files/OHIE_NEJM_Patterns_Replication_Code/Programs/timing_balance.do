clear matrix
cap program drop main

program main
    timing_balance, weight(weight_180days) ///
        lottery_list(birthyear_list female_list english_list self_list first_day_list ///
        have_phone_list pobox_list) preperiod_list(preperiod_any_visits)
    //Note: weight_180days = 1, i.e. there are no weights at 180 days.
    timing_balance, weight(weight_540days) ///
        lottery_list(birthyear_list female_list english_list self_list first_day_list ///
        have_phone_list pobox_list) preperiod_list(preperiod_any_visits)
    timing_balance, weight(weight_720days) ///
        lottery_list(birthyear_list female_list english_list self_list first_day_list ///
        have_phone_list pobox_list) preperiod_list(preperiod_any_visits)
    matrix list timing_balance
    matrix_to_txt, mat(timing_balance) saving(../Output/table_a4a_timing_balance.txt) ///
        title("Appendix Table 4a: Balance in ED Sample with Different Weights") ///
        note(Note: -zip code median household income- and -Number of ED visits, pre-lottery- are ///
        omitted, so the pooled F-statistics do not replicate the paper.) usec user replace
end

program timing_balance
    syntax, weight(str) lottery_list(str) preperiod_list(str)
    local uselist "person_id household_id treatment *_list draw_lottery preperiod_* `weight'"
    use `uselist' using ../Data/data_for_analysis, clear

    xi, prefix(hh) i.numhh_list
    xi, prefix(draw) i.draw_lottery

    local inflate_list "female_list english_list self_list first_day_list have_phone_list pobox_list"
    foreach inf_var in `inflate_list' preperiod_any_visits {
        replace `inf_var' = 100 * `inf_var'
    }
    tempfile bal_temp
    save `bal_temp'

    foreach var in `lottery_list' `preperiod_list' {
        *Control mean
        reg `var' if treatment == 0 [pw = `weight']
        local control_mean = _b[_cons]

        *Balance
        reg `var' treatment hh* draw* [pw = `weight'], cluster(household_id)

        matrix M = r(table)
        matrix balance_`weight' = nullmat(balance_`weight')\ ((`control_mean'\ .), ///
            (_b[treatment]\ _se[treatment]), (M[rownumb(M, "pvalue"), colnumb(M, "treatment")]\ .))
        local rowname "`rowname' `var' se"
    }
    matrix colnames balance_`weight' = control_mean_`weight' tc_diff_`weight' p_`weight'
    matrix rownames balance_`weight' = `rowname'
    
    joint_ftest using `bal_temp', weight(`weight') joint_ftest_list(`lottery_list')
    joint_ftest using `bal_temp', weight(`weight') joint_ftest_list(`preperiod_list')
    joint_ftest using `bal_temp', weight(`weight') joint_ftest_list(`lottery_list' `preperiod_list')
    matrix ftest_`weight' = J(6,2,.), ftest_`weight'
    matrix rownames ftest_`weight' = lottery_list_F lottery_list_p ///
        preperiod_list_F preperiod_list_p lottery_preperiod_F lottery_preperiod_p

    matrix timing_balance = nullmat(timing_balance), (balance_`weight'\ ftest_`weight')
end

    program joint_ftest
        syntax using, weight(str) joint_ftest_list(str)
        use `using', clear
        local num : word count `joint_ftest_list' 
        expand `num'
        bys person_id: gen order = _n
        gen outcome = .
        gen constant = 1
        local i = 0
        foreach var in `joint_ftest_list' {
            local ++i
            replace outcome       = `var'    if order == `i'
            gen     treatment_`i' = treatment *(order == `i')
            foreach control of varlist hh* draw* constant {
                gen X`i'`control' = `control' *(order == `i')
            }
        }
        reg outcome treatment_* X* [pw = `weight'], cluster(household_id) nocons
        testparm treatment_* 

        matrix ftest_`weight' = nullmat(ftest_`weight')\ r(F)\ r(p)
    end

*Execute
main

