cap program drop main

program main
    control_means, sample(ed)      mean_list(birthyear_list female_list english_list self_list ///
        first_day_list have_phone_list pobox_list)
    control_means, sample(overlap) mean_list(birthyear_list female_list english_list self_list ///
        first_day_list have_phone_list pobox_list) weight(weight_total_inp)
    matrix list control_means
    matrix_to_txt, mat(control_means) saving(../Output/table_a2_control_means.txt) ///
        title("Appendix Table 2: Summary Statistics, Control Means") ///
        note(Note: -zip code median household income- is omitted.) usec user replace
end

program control_means
    syntax, sample(str) mean_list(str) [weight(str)]
    cap matrix drop control_mean_`sample' N_`sample'
    use ../Data/data_for_analysis.dta, clear
    keep if `sample'_sample == 1

    local inflate_list "female_list english_list self_list first_day_list have_phone_list pobox_list"
    foreach inf_var in `inflate_list' {
        replace `inf_var' = 100 * `inf_var'
    }

    count if treatment == 0
    matrix N_`sample' = r(N)\ _N

    macro drop _rowname
    foreach var in `mean_list' {
        if "`weight'" != "" {
            reg `var' if treatment == 0 [pw = `weight']
        }
        else {
            reg `var' if treatment == 0
        }
        if "`var'" == "birthyear_list" {
            matrix control_mean_`sample' = nullmat(control_mean_`sample')\ _b[_cons]\ e(rmse)
        }
        else {
            matrix control_mean_`sample' = nullmat(control_mean_`sample')\ _b[_cons]\ .
        }
        local rowname "`rowname' `var' sd"
    }
    matrix colnames control_mean_`sample' = `sample'
    matrix control_means = nullmat(control_means), (control_mean_`sample'\ N_`sample')
    matrix rownames control_means = `rowname' N_control N_total
end

*Execute
main
