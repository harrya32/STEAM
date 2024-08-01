/**************************************
*This code replicates exhibits and numbers in text from "The Effect of Medicaid Coverage on ED Use — 
Further Evidence from Oregon’s Experiment" (Finkelstein et al. 2016).

It runs on Stata 13 or later.

Estimates that use administrative hospital discharge records, or data about zip code of residence 
(median household income in zip code), number of ED visits for 60- or 180-day periods, or any ED 
visit in 60-day periods cannot be replicated because these data are not public.

Exhibits replicated are:
    Figure 1: Right Panel
	Appendix Table 2 - "zip code median household income" is omitted. 
	Appendix Table 3
	Appendix Table 4a - "zip code median household income" and "Number of ED visits, pre-lottery" 
                        are omitted, so the pooled F-statistics do not replicate the paper.
	Appendix Table 4b - "zip code median household income" is omitted, so the pooled F-statistic 
                        does not replicate the paper.
    Appendix Table 6: Columns 1-7
    Appendix Figure 3: Right Panel
    Appendix Table 10
    Appendix Table 11
    
The code also replicates numbers reported in text that are not in any of the tables. These include 
the date range for ED data, first stage and control complier means for the joint outcomes analysis, 
and average in-person survey interview date for the overlap sample.

The table-generating code is divided among several .do files: this is the master file that 
calls all of them. 

To run this file:

Save this .do file to the folder where you wish to do your work.

Save the following .do files to a subfolder named "Programs"
	joint_impact_prepare_data.do
	control_means.do
	dist_of_weights.do
	timing_balance.do
	joint_impact_balance.do
	timing_non_parametric_estimates.do
	timing_non_parametric_tables_and_figures.do
	joint_impact_analysis.do
    numbers_in_text.do

Save the following datasets to a subfolder named "Data"
	oregonhie_patterns_vars.dta

Create an empty subfolder named "Output"
	
Currently the code will generate all tables and figures listed above. Add an "*" in front of the 
code in program "tables_and_figures" below to choose which tables to skip. For example, leaving the 
code as "do Programs/control_means.do" will run Appendix Table 2, and changing it to 
"*do Programs/control_means.do" will skip Table 2.

The full log file outputs to oregonhie_patterns_replication.log. In addition, a 
txt file with each generated table or a png picture outputs to the subfolder named "Output".

**************************************/
clear all
program drop _all
set more off
cap log close
cap cd Programs
cap mkdir ../Output
adopath + SubPrograms/

log using ../oregonhie_patterns_replication.log, text replace

program master_main
    prepare_data
    tables_and_figures
end

program prepare_data
    do joint_impact_prepare_data.do
end

program tables_and_figures
    *Summary Statistics for the Control Group (Appendix Table 2)
    do control_means.do

    *Distribution of Weights (Appendix Table 3)
    do dist_of_weights.do

    *Treatment-Control Balance for ED Sample(Appendix Table 4a)
    do timing_balance.do

    *Treatment-Control Balance for Overlap Sample(Appendix Table 4b)
    do joint_impact_balance.do

    *Timing Non-Parametric Analyses (Figure 1; Appendix Figures 3; Appendix Tables 6)
    do timing_non_parametric_estimates.do
    do timing_non_parametric_tables_and_figures.do

    *Joint Outcomes Analysis (Appendix Tables 10 and 11)
    do joint_impact_analysis.do

    *numbers_in_text
    do numbers_in_text.do
end

*Execute
master_main

log close
