cap program drop main
cap mkdir ../Output/estimates

program main
    save_results, interval(180) measure(ed) outcomes(any_visit) weights(weight_*days)
    save_results, interval(180) measure(ed) outcomes(any_visit) weights(weight_720days)

    create_figure_main_text, interval(180) measure(ed) outcomes(any_visit) ///
        weights(weight_xdays)   est(iv)       ylabel_any(0.1(0.05)0.325)
    create_figure_appendix,  interval(180) measure(ed) outcomes(any_visit) ///
        weights(weight_720days) estimates(iv) ylabel_any(0.1(0.1)0.6)

    export_table, interval(180) measure(ed) outcomes(any_visit) weights(weight_xdays) ///
        export_tab(table_a6_timing_nonparametric) title("Appendix Table 6: Impact Over Time on ED Use")

    //Note: We also looped over number of visits ("total_visits") as "outcomes", hospital discharge 
    //      data ("hdd") as "measure", and 60-day as "interval" for published numbers in paper.
end

program save_results
    syntax, interval(str) measure(str) outcomes(str) weights(str)
    local w `weights'
    if "`weights'" == "weight_*days" {
        local w weight_xdays
    }
    local i = 0
    foreach outcome in `outcomes' {
        local ++i
        local mat `measure'`w'`i'`interval'
        capture matrix `mat' = `mat', `mat'iv, `mat'fs
        matrix list `mat'
        clear
        svmat `mat', names(col)
        foreach est in itt iv fs {
            gen `est'_control_mean_plus_beta = control_mean + `est'_beta
            gen `est'_control_mean_plus_ll   = control_mean + `est'_ll
            gen `est'_control_mean_plus_ul   = control_mean + `est'_ul
        }
        save ../Output/estimates/step`interval'days_`measure'_`w'_`outcome', replace
    }
end

program create_figure_main_text
    syntax, interval(str) measure(str) outcomes(str) weights(str) est(str) ///
        [ylabel_total(str) ylabel_any(str)]
    foreach outcome in `outcomes' {
        use ../Output/estimates/step`interval'days_`measure'_`weights'_`outcome', clear
        if "`outcome'" == "any_visit" {
            local ylabel "`ylabel_any'"
            local ytitle "Percentage with any ED visit"
        }
        if "`outcome'" == "total_visits" {
            local ylabel "`ylabel_total'"
            local ytitle "No. of ED visits per person"
        }
        twoway (connected `est'_control_mean_plus_beta interval, msymbol(O) msize(small) ///
            mcolor(gs0)) ///
            (rcap `est'_control_mean_plus_ll `est'_control_mean_plus_ul interval, lpattern(dash) ///
            lcolor(gs0)) ///
            (connected control_mean interval, msymbol(X) msize(large) mcolor(gs6) ///
            lpattern(longdash) lcolor(gs6)), ylabel(`ylabel')  ytitle(`ytitle') ///
            scheme(s2mono) xlabel(`interval'(`interval')720) xtitle(Days since Lottery) ///
            legend(label(1 "Medicaid") label(3 "Control group") label(2 "95% confidence interval"))
    }
    graph export ../Output/figure_1_any_visit.png, width(1375) height(1000) replace
end

program create_figure_appendix
    syntax, interval(str) measure(str) outcomes(str) weights(str) estimates(str) ///
        [ylabel_total(str) ylabel_any(str)]
    foreach est in `estimates' {
        foreach outcome in `outcomes' {
		    use ../Output/estimates/step`interval'days_`measure'_`weights'_`outcome', clear
            if "`outcome'" == "any_visit" {
                local ylabel "`ylabel_any'"
            }
            if "`outcome'" == "total_visits" | "`outcome'" == "total_days" {
                local ylabel "`ylabel_total'"
            }
            twoway (connected `est'_control_mean_plus_beta interval, ///
                msymbol(O) msize(small) mcolor(gs0)) ///
                (rcap `est'_control_mean_plus_ll `est'_control_mean_plus_ul interval, ///
                lpattern(dash) lcolor(gs0)) ///
                (connected control_mean interval, msymbol(X) msize(large) mcolor(gs6) ///
                lpattern(longdash) lcolor(gs6)), ylabel(`ylabel') ///
                scheme(s2mono) xlabel(`interval'(`interval')720) xtitle(Days since Lottery) ///
                legend(label(1 "Estimated impact of Medicaid plus control mean") ///
                label(3 "Control mean") label(2 "95 percent confidence interval") col(1))
            graph export ../Output/figure_a3_any_visit_constant_weight.png, ///
                width(1375) height(1000) replace
        }
    }
end

program export_table
    syntax, interval(str) measure(str) outcomes(str) weights(str) export_tab(str) title(str)
    foreach outcome in `outcomes' {
        cap macro drop _rowname
        cap matrix drop non_para_`measure'`outcome'
        use ../Output/estimates/step`interval'days_`measure'_`weights'_`outcome', clear
        forv t = 180(180)720 {
            local export_list "fs_control_mean fs_control_sd fs_beta fs_se fs_comp_pvalue"
            local export_list "`export_list' control_mean control_sd"
            local export_list "`export_list' itt_beta itt_se iv_beta iv_se iv_comp_pvalue"
            foreach var in `export_list' {
                sum `var' if interval == `t'
                local `var' = r(mean)
            }
            matrix non_para_`measure'`outcome' = nullmat(non_para_`measure'`outcome')\ ///
                ((`fs_control_mean'\ `fs_control_sd'), (`fs_beta'\ `fs_se'), (`fs_comp_pvalue'\ .), ///
                (`control_mean'\ `control_sd'), (`itt_beta'\ `itt_se'), ///
                (`iv_beta'\ `iv_se'), (`iv_comp_pvalue'\ .))
            local rowname "`rowname' `t' se"
        }
    }
    matrix colnames non_para_`measure'any_visit = fs_control_mean fs_beta fs_comp_pvalue ///
        control_mean itt_beta iv_beta iv_comp_pvalue
    matrix rownames non_para_`measure'any_visit = `rowname'
    matrix list non_para_`measure'any_visit
    matrix_to_txt, mat(non_para_`measure'any_visit) saving(../Output/`export_tab'.txt) ///
        title(`title') usec user replace
end

*Execute
main
