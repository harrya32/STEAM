clear matrix
cap program drop main 
cap program drop weight_dist

program main
    weight_dist using ../Data/data_for_analysis, weight(540days)
    weight_dist using ../Data/data_for_analysis, weight(540days)   keep(if weight_540days != 0)
    weight_dist using ../Data/data_for_analysis, weight(720days)
    weight_dist using ../Data/data_for_analysis, weight(720days)   keep(if weight_720days != 0)
    weight_dist using ../Data/data_for_analysis, weight(total_inp) keep(if overlap_sample == 1)
    matrix rownames weight_dist = ///
        full_540        control_540        treat_540 ///
        full_540nonzero control_540nonzero treat_540nonzero ///
        full_720        control_720        treat_720 ///
        full_720nonzero control_720nonzero treat_720nonzero ///
        full_overlap    control_overlap    treat_overlap
    matrix list weight_dist
    matrix_to_txt, mat(weight_dist) saving(../Output/table_a3_dist_of_weights.txt) ///
        title("Appendix Table 3: Distribution of the Weights") usec user replace
end

program weight_dist
    syntax using, weight(str) [keep(str)]
    use `using' `keep', clear
    foreach condition in "treatment != ." "treatment == 0" "treatment == 1" {
        count if weight_`weight' == 0 & `condition'
        local zero_w = r(N)
        sum weight_`weight' if `condition', d
        matrix weight_dist = nullmat(weight_dist)\ ///
            (r(mean), r(sd), r(min), r(p50), r(p75), r(p95), r(max), r(N), (`zero_w'/r(N))*100)
    }
    matrix colnames weight_dist = mean sd min median p75 p95 max N perc_zero
end

*Execute
main
