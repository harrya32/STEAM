cap program drop main joint_impact_data

program main
    joint_impact_data
end

program joint_impact_data
    use ../Data/oregonhie_patterns_vars.dta, clear
    gen ed_sample = 1
    gen overlap_sample = 1 if ed_sample==1 & sample_inp_resp==1 & doc_any_incl_probe_inp!=.
    count if overlap_sample == 1
    assert r(N) == 10156
    count if ed_sample == 1
    assert r(N) == 24646

    gen ed_admin         =  any_inp_match_ed
    gen ed_inp           =  ed_any_incl_probe_inp  if overlap_sample == 1
    gen doc_inp          =  doc_any_incl_probe_inp if overlap_sample == 1
    gen ed_admin_doc_inp = (any_inp_match_ed==1 & doc_any_incl_probe_inp==1) if overlap_sample == 1
    gen ed_inp_doc_inp   = (ed_any_incl_probe_inp==1 & doc_any_incl_probe_inp==1) ///
        if doc_any_incl_probe_inp!=. & ed_any_incl_probe_inp!=. & overlap_sample == 1

    label var ed_admin          "Went to the ED (ED admin data)"
    label var ed_inp            "Went to the ED (inperson survey, defined for ED/Inp overlap)"
    label var doc_inp           "Went to doctor (inperson survey, defined for ED/Inp overlap)"
    label var ed_admin_doc_inp  "Went to the ED (ED admin data)   and the doctor (inperson survey)"
    label var ed_inp_doc_inp    "Went to the ED (inperson survey) and the doctor (inperson data)"

    assert any_inp_match_ed!=. if overlap_sample == 1
    assert doc_any_incl_probe_inp!=. if overlap_sample == 1
    assert any_inp_match_ed==1 if ed_admin_doc_inp == 1
    assert doc_any_incl_probe_inp==1 if ed_admin_doc_inp == 1

    save ../Data/data_for_analysis.dta, replace
end

*Execute
main
