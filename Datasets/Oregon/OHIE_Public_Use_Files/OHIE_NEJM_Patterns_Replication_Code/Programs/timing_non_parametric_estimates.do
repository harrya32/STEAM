clear matrix
cap program drop main

program main
    non_parametric_est,   interval(180) measure(ed)  outcomes(any_visit) weights(weight_*days)
    comparison_p_stacked, interval(180) measure(ed)  outcomes(any_visit) weights(weight_*days)
    non_parametric_est,   interval(180) measure(ed)  outcomes(any_visit) weights(weight_720days)
    //Note: We also looped over number of visits ("total_visits") as "outcomes", hospital discharge 
    //      data ("hdd") as "measure", and 60-day as "interval" for published numbers in paper.
end

program non_parametric_est
    syntax, interval(str) measure(str) outcomes(str) weights(str)
    local i = 0
    foreach outcome in `outcomes' {
        local ++i
        macro drop _row_names _col_names

        local uselist "person_id household_id treatment numhh_list draw_lottery"
        local uselist "`uselist' `outcome'_* preperiod_`outcome'* medicaid_*_`interval'p* `weights'"
        use `uselist' using ../Data/data_for_analysis, clear

        xi, prefix(hh) i.numhh_list
        xi, prefix(draw) i.draw_lottery
 
        forv x = `interval'(`interval')720 {
            local weight `weights'
            local w      `weights'
            if "`weights'" == "weight_*days" {
                local weight weight_`x'days
                local w      weight_xdays
                // weight_180days = 1, i.e. there are no weights at 180 days.
            }
            local mat `measure'`w'`i'`interval'

            *Control means
            reg `outcome'_`interval'p_`x' if treatment == 0 [pw = `weight']
            matrix `mat'`x' = _b[_cons], e(rmse)
            local col_names "control_mean control_sd"

            *First stage (control means and estimates)
            reg medicaid_all_`interval'p_period_`x' if treatment == 0 [pw = `weight']
            matrix `mat'`x' = `mat'`x', _b[_cons], e(rmse)
            local col_names "`col_names' fs_control_mean fs_control_sd"

            reg medicaid_all_`interval'p_period_`x' treatment hh* draw* preperiod_`outcome'* ///
                [pw = `weight'], cluster(household_id)
            make_nonpara_est_mat, input_mat(M) output_mat(`mat'`x') est(fs) est_var(treatment)
            local col_names `col_names' `:di r(cnames)'

            *ITT
            reg `outcome'_`interval'p_`x' treatment hh* draw* preperiod_`outcome'* ///
                [pw = `weight'], cluster(household_id)
            make_nonpara_est_mat, input_mat(M) output_mat(`mat'`x') est(itt) est_var(treatment)
            local col_names `col_names' `:di r(cnames)'

            *IV	
            ivregress 2sls `outcome'_`interval'p_`x' ///
                (medicaid_all_`interval'p_period_`x' = treatment) hh* draw* preperiod_`outcome'* ///
                [pw = `weight'], cluster(household_id)
            make_nonpara_est_mat, input_mat(M) output_mat(`mat'`x') est(iv) ///
                est_var(medicaid_all_`interval'p_period_`x')
            local col_names `col_names' `:di r(cnames)'

            matrix `mat' = nullmat(`mat')\ (`x', `mat'`x')
            local col_names "interval `col_names'"
            local row_names "`row_names' d`x'"
        }
    matrix colnames `mat' = `col_names'
    matrix rownames `mat' = `row_names'
    }
end

    program make_nonpara_est_mat, rclass
        syntax, input_mat(str) output_mat(str) est(str) est_var(str)
        matrix `input_mat' = r(table) 
        matrix `output_mat' = `output_mat', ///
            _b[`est_var'], _se[`est_var'], ///
            `input_mat'[rownumb(`input_mat', "pvalue"), colnumb(`input_mat', "`est_var'")], ///
            `input_mat'[rownumb(`input_mat', "ll"),     colnumb(`input_mat', "`est_var'")], ///
            `input_mat'[rownumb(`input_mat', "ul"),     colnumb(`input_mat', "`est_var'")]
        return local cnames "`est'_beta `est'_se `est'_pvalue `est'_ll `est'_ul"
    end


program comparison_p_stacked
    syntax, interval(str) measure(str) outcomes(str) weights(str)
    local i = 0
    foreach outcome in `outcomes' {
        local ++i
        macro drop _weight _yvariables _insurance_variables

        local uselist "person_id household_id treatment numhh_list draw_lottery"
        local uselist "`uselist' `outcome'_* preperiod_`outcome'* medicaid_*_`interval'p* `weights'"
        use `uselist' using ../Data/data_for_analysis, clear

        xi, prefix(hh) i.numhh_list
        xi, prefix(draw) i.draw_lottery

        forv x = `interval'(`interval')720 {
            local weight "`weight' weight_`x'days"
            local yvariables "`yvariables' `outcome'_`interval'p_`x'"
            local insurance_variables "`insurance_variables' medicaid_all_`interval'p_period_`x'"
        }
        local w weight_xdays
        local mat `measure'`w'`i'`interval'

        local number_of_variables: word count `yvariables'
        expand `number_of_variables'
        bys person_id: gen order = _n
        gen outcome = .	
        gen medicaid_all = .
        gen weight = .
        gen constant = 1
        local order = 0
        foreach yvar in `yvariables' {
            local ++order
            replace outcome           = `yvar'                                   if order == `order'
            replace medicaid_all      =`: word `order' of `insurance_variables'' if order == `order'
            replace weight            =`: word `order' of `weight''              if order == `order'
            gen medicaid_all_X`order' =`: word `order' of `insurance_variables'' *(order == `order')
            gen treatment_`order'     = treatment                                *(order == `order')
            foreach control of varlist hh* draw* preperiod_`outcome'* constant { 
                gen X`order'`control' = `control'                                *(order == `order')
            }
        }

        *Stacked first stage
        reg medicaid_all treatment_* X* [pw = weight], cluster(household_id) nocons

        local no_of_period = 720/`interval'
        matrix `mat'fs = .
        forval p = 2/`no_of_period' {
            test _b[treatment_1] = _b[treatment_`p']
            matrix `mat'fs = nullmat(`mat'fs)\ r(p)
        }
        forval p = 1/`no_of_period' {
            local day = `p'*`interval'
            local fs_est = `mat'[rownumb(`mat', "d`day'"), colnumb(`mat', "fs_beta")]
            assert abs(_b[treatment_`p'] - `fs_est') < 0.00000001
        }

        *Stacked IV
        ivregress 2sls outcome (medicaid_all_X* = treatment_*) X* [pw = weight], ///
            cluster(household_id) nocons

        matrix `mat'iv = .
        forval p = 2/`no_of_period' {
            test _b[medicaid_all_X1] = _b[medicaid_all_X`p']
            matrix `mat'iv = nullmat(`mat'iv)\ r(p)
        }
        forval p = 1/`no_of_period' {
            local day = `p'*`interval'
            local iv_est = `mat'[rownumb(`mat', "d`day'"), colnumb(`mat', "iv_beta")]
            assert abs(_b[medicaid_all_X`p'] - `iv_est') < 0.00000001
        }

        matrix colnames `mat'iv = iv_comp_pvalue
        matrix colnames `mat'fs = fs_comp_pvalue
    }
end

*Execute
main
