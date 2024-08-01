clear matrix
cap program drop main

program main
    joint_impact_analysis, outcomes(doc_inp ed_admin ed_admin_doc_inp) ///
        condition(overlap_sample == 1) insure_var(ohp_all_ever_inperson) weight(weight_total_inp) ///
        export_tab(table_a10_joint_outcomes) ///
        title("Appendix Table 10: Testing the Independence of Medicaid-Induced Increases in Office Visits and ED Visits")
    joint_impact_analysis, outcomes(doc_inp ed_inp ed_inp_doc_inp) ///
        condition(overlap_sample == 1 & ed_inp !=.) insure_var(ohp_all_ever_inperson) ///
        weight(weight_total_inp) export_tab(table_a11_joint_outcomes_robustness) ///
        title("Appendix Table 11: Testing the Independence of Medicaid-Induced Increases in Office Visits and ED Visits, Robustness to Self-Reported ED Visits")
end

program joint_impact_analysis
    syntax, outcomes(str) [condition(str)] insure_var(str) weight(str) [export_tab(str) title(str)]
    cap matrix drop mu_c mu_cc mu_cn mu_nt fs fs_se pi_nt late_stacked N late_unstacked

    local uselist "person_id household_id treatment numhh_list overlap_sample ed_sample"
    local uselist "`uselist' doc_inp ed_inp ed_admin ed_admin_doc_inp ed_inp_doc_inp"
    local uselist "`uselist' ohp_all_ever_inperson weight_total_inp"
    use `uselist' if `condition' using ../Data/data_for_analysis.dta, clear

    xi, prefix(hh) i.numhh_list

    local mat `: word 3 of `outcomes''
    local i = 0
    foreach var in `outcomes' {
        local ++i
        local row "`row' `var' se"

        *control mean - mu_c
        reg `var' if treatment==0 [pw=`weight']
        make_overlap_mat, input_est(_b[_cons]) temp_local(mu_c_`i') output_mat(mu_c) rownames(`row')

        *first stage (comliance rate) - fs (pi_c)
        reg `insure_var' treatment hh* [pw=`weight'], cluster(household_id)
        make_overlap_mat, input_est(_b[treatment])  temp_local(fs)    output_mat(fs)
        make_overlap_mat, input_est(_se[treatment]) temp_local(fs_se) output_mat(fs_se) ///
            rownames(`row')
        
        *fraction of never taker - pi_nt
        gen never_taker =(treatment == 1 & `insure_var' == 0)
        reg never_taker if treatment == 1 [pw=`weight']
        make_overlap_mat, input_est(_b[_cons]) temp_local(pi_nt) output_mat(pi_nt)
        drop never_taker

        *never taker mean - mu_nt
        reg `var' if treatment == 1 & `insure_var' == 0 [pw=`weight']
        make_overlap_mat, input_est(_b[_cons]) temp_local(mu_nt_`i') output_mat(mu_nt)

        *mean among controls who do not take up - mu_cn
        reg `var' if treatment == 0 & `insure_var' == 0 [pw=`weight']
        make_overlap_mat, input_est(_b[_cons]) temp_local(mu_cn_`i') output_mat(mu_cn)
        
        *calculate control complier mean - mu_cc
        local mu_cc_`i' = ((${fs} + ${pi_nt})*${mu_cn_`i'} - ${pi_nt}*${mu_nt_`i'})/${fs}
        make_overlap_mat, input_est(`mu_cc_`i'') temp_local(mu_cc_`i') output_mat(mu_cc)

        *unstacked LATE
        ivregress 2sls `var' (`insure_var' = treatment) hh* [pw = `weight'], cluster(household_id) 
        matrix late_unstacked = nullmat(late_unstacked)\ _b[`insure_var']\ _se[`insure_var']
        matrix colnames late_unstacked = late_unstacked

        *N
        qui sum `var'
        make_overlap_mat, input_est(r(N)) temp_local(N) output_mat(N)
    }
    matrix stats_`mat' = mu_c, mu_cc, mu_cn, mu_nt, fs, fs_se, pi_nt
    matrix stats_`mat' = stats_`mat'[1, 1...]\  stats_`mat'[3, 1...]\ stats_`mat'[5, 1...]
    matrix list stats_`mat'


    *Stacked LATE
    local number_of_variables: word count `outcomes' 
    expand `number_of_variables'
    bys person_id: gen order = _n
    gen outcome = .
    gen constant = 1
    local i = 0
    foreach var in `outcomes' {
        local ++i
        di "`i' `var'"
        replace outcome        = `var'       if order == `i'
        gen     ohp_all_X`i'   = `insure_var' *(order == `i')
        gen     treatment_`i'  = treatment    *(order == `i')
        foreach control of varlist hh* constant {
            gen X`i'`control'  = `control'    *(order == `i')
        }	
    }
    ivregress 2sls outcome (ohp_all_X* = treatment_*) X* [pw = `weight'], cluster(household_id) ///
        nocons

    estimates store stacked
    make_late_mat, input_mat(M) output_mat(late_stacked) est(ohp_all_X1 ohp_all_X2 ohp_all_X3)
    forval i = 1(2)5 {
        assert abs(late_unstacked[`i',1] - late_stacked[`i',1]) < 0.00000001
    }
    matrix `mat' = mu_c, late_stacked
    matrix `mat'_check = N, late_unstacked


    *Test for Independence
    estimates restore stacked
    local exp "(`mu_cc_2'*_b[ohp_all_X1] + `mu_cc_1'*_b[ohp_all_X2]+ _b[ohp_all_X2]*_b[ohp_all_X1])"

    nlcom (`exp'), post

    local se_pred = _se[_nl_1]
    local p_pred  = 2 * normal( - abs(_b[_nl_1] / _se[_nl_1]))

    estimates restore stacked
    testnl _b[ohp_all_X3] - `exp' = 0

    matrix joint_pred = (.\ .), (`exp'\ `se_pred'), (`p_pred'\ .)
    matrix rownames joint_pred = joint_pred se
    matrix f_stat = J(2,2,.), (r(chi2)\ r(p))
    matrix rownames f_stat = F_stat pvalue
    matrix `mat' = `mat'\ joint_pred\ f_stat
    matrix list `mat'
    matrix list `mat'_check
    if "`export_tab'" != "" {
        matrix_to_txt, mat(`mat') saving(../Output/`export_tab'.txt) title(`title') usec user replace
    }
end

    program make_overlap_mat, rclass
        syntax, input_est(str) temp_local(str) output_mat(str) [rownames(str)]
        global `temp_local' = `input_est'
        matrix `output_mat' = nullmat(`output_mat')\ ${`temp_local'} \.
        matrix colnames `output_mat' = `output_mat'
        matrix rownames `output_mat' = `rownames'
    end

    program make_late_mat
        syntax, input_mat(str) output_mat(str) est(str)
        matrix `input_mat' = r(table)
        matrix `output_mat' = ///
            ((_b[`: word 1 of `est'']\ _se[`: word 1 of `est'']), ///
            (M[rownumb(M, "pvalue"), colnumb(M, "`: word 1 of `est''")]\. ))\ ///
            ((_b[`: word 2 of `est'']\ _se[`: word 2 of `est'']), ///
            (M[rownumb(M, "pvalue"), colnumb(M, "`: word 2 of `est''")]\. ))\ ///
            ((_b[`: word 3 of `est'']\ _se[`: word 3 of `est'']), ///
            (M[rownumb(M, "pvalue"), colnumb(M, "`: word 3 of `est''")]\. ))
        matrix colnames `output_mat' = `output_mat' p_value
    end
    
*Execute
main
