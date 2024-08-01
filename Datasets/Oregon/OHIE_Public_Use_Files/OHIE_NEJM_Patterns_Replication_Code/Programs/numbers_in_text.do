clear matrix
cap program drop main

program main
    dates_timing_data
    overlap_stats
    overlap_avg_interview_dt
end

program dates_timing_data
    foreach measure in ed {
        use *_min *max using ../Data/data_for_analysis, clear
        foreach stat in min max {
            foreach var in visit {
                qui sum `var'_dt_`stat'
                di "The `stat' `var' date in the visit-level `measure' data is `: display %td r(mean)'."
            }
        }
    }
end

program overlap_stats
    qui include joint_impact_analysis.do
    qui joint_impact_analysis, outcomes(doc_inp ed_admin ed_admin_doc_inp) ///
        condition(overlap_sample == 1) insure_var(ohp_all_ever_inperson) weight(weight_total_inp)
    matrix list stats_ed_admin_doc_inp
    di "- Column mu_cc in the matrix listed above shows the control complier means for joint outcomes analysis presented in Appendix Table 10"
    local fs = round(fs[1,1], 0.00001)
    local fs_se = round(fs_se[1,1], 0.00001)
    di "- Overlap sample FS = `fs' (se = `fs_se')."
end

program overlap_avg_interview_dt
    use ../Data/data_for_analysis.dta, clear
    qui keep if overlap_sample == 1
    qui reg dt_completed_inp [pw = weight_total_inp]
    local overlap_avg_interview_dt = _b[_cons]
    di "- The average interview date for the overlap sample was:`: display %td `overlap_avg_interview_dt'',"

    qui gen months_after_notify = (dt_completed_inp - dt_notify_treat)/30.5
    qui reg months_after_notify [pw = weight_total_inp]
    local months_after_notify = round(_b[_cons], 0.001)
    di "or `months_after_notify' months after notification for individuals in the treatment group."
end

*Execute
main
