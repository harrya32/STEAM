/**************************************
*This code replicates tables from Finkelstein et al (QJE 2012)
*It runs on Stata versions 11 and 12

Tables replicated are:
	Table 1
	Table 2: Panel A, rows 2 and 3 
	Table 3: Columns 1, 2, 5, and 6
	Table 5
	Table 6
	Table 8
	Table 9 
	Table 10
	Table 11

Estimates that use administrative hospital records, credit report data, or 
data about zip code of residence (median household income in zip code)
cannot be replicated because these data are not public.

Note that when running versions of Stata 12 or higher, Figure 1 can only be 
generated if "version 11" is specified at the top of the file (as it is below).

The table-generating code is divided among several .do files: this is the master file that 
calls all of them. 

To run this file:

Save this .do file to the folder where you wish to do your work.

Save the following .do files to a subfolder named "SubPrograms"
	prepare_data.do
	quantiles.do
	survey_results_comparison.do
	tables_analysis.do
	tables_balance.do
	tables_firststage.do
	tables_means.do
	

Save the following datasets to a subfolder named "Data"
	oregonhie_descriptive_vars.dta
	oregonhie_stateprograms_vars.dta
	oregonhie_mortality_vars.dta
	oregonhie_survey0m_vars.dta
	oregonhie_survey6m_vars.dta
	oregonhie_survey12m_vars.dta
	
Edit the global "switches" below to choose which tables to run. For example,
leaving the code as "global New 1 1" will run Table 1, and changing it to 
"global New1 0" will skip Table 1.

Edit the global "iterations" below to choose the number of iterations for the 
multiple inference adjustment. This is set to 10,000 for replication; setting 
to a much smaller number (for example, 2) will reduce runtime and change
only the family-wise p-values.

The full log file outputs to oregon_hie_qje_replication.log. In addition, a 
log file with just the generated tables outputs to all_tables_multiinf.log
(or all_tables.log if the iterations for the multiple inference adjustment 
is set below 10,000). 

**************************************/

set more off
capture log close
log using oregon_hie_qje_replication.log, text replace
set linesize 120
set matsize 800

clear all
clear matrix
matrix drop _all

set seed 457906


*******************
* SWITCHES
*******************

global iterations = 2 // Number of multiple inference iterations - should be 10,000

global New1 1 // Table 1: Demographic characteristics of study population (controls)
global New2 1 // Table 2: Balance (note that F-statistics computed do not replicate Finkelstein et al (2012) due to the exclusion of median household income in zip code)
global New3 1 // Table 3: First Stage Estimates
* global New4 1 // Table 4: Impact of Health Insurance on Hospital Utilization (Administrative Data) // - DATA NOT PUBLICLY AVAILABLE
global New5 1 // Table 5: Impact of Health Insurance on Health Care Utilization (Survey Data)
global New6 1 // Table 6: Compliance with recommended preventive care, Survey data
* global New7 1 // Table 7: Impact of Health Insurance on Financial Well Being (Administrative Data) // - DATA NOT PUBLICLY AVAILABLE
global New8 1 // Table 8: Impact of Health Insurance on Financial Well Being (Survey Data)
global New9 1 // Table 9: Impact of Health Insurance on Health
global New10 1 // Table 10: Potential Mechanisms for Improved Health, Survey Data
global New11 1 // Table 12: Earlier Surveys

global ALT 0

local I2_switch = $New2
local I3_switch = $New1
local I4_switch = $New3
local I5_switch = $New3

local I2_2_switch = `I2_switch'
local I2_3_switch = `I2_switch'
local I2_4_switch = `I2_switch'
local I2_F_switch = `I2_switch'
local I2_4_extra_switch = `I2_switch'

local i4_1_switch = `I4_switch'
local i4_2_switch = `I4_switch'
local i4_3_switch = `I4_switch'
local fs_switch = $New3


local U1_survey_switch = $New5
local P1_survey_switch = $New5

local M1_3_switch = $New6

local P2_survey_switch = $New8

local P3_admin_switch = $New9 
local P3_survey_switch = $New9

local M1_1_switch = $New10
local M1_2_switch = $New10
local E1_1_switch = $New10
local M1_4_switch = $New10

local New11_switch = $New11


*********************
* GLOBALS AND LOCALS
*********************

**Initial tables

global baseline_list "birthyear_list female_list english_list self_list first_day_list have_phone_list pobox_list zip_msa"

global desc_var "race_white_12m race_black_12m race_hisp_12m edu_12m_1 edu_12m_2 edu_12m_3 edu_12m_4 employ_hrs_12m_1 employ_hrs_12m_2 employ_hrs_12m_3 employ_hrs_12m_4 dia_dx_12m ast_dx_12m hbp_dx_12m emp_dx_12m dep_dx_12m fpl_categ_12m_1 fpl_categ_12m_2 fpl_categ_12m_3 fpl_categ_12m_4 fpl_categ_12m_5 ins_any_12m ins_ohp_12m ins_private_12m ins_other_12m ins_months_12m"


**Primary tables

global hospital_use_list "days_all totcharge_all numproc_all"

*global credit_finance_list "any_bankruptcy_n any_lien_n any_judg_n any_coll_n any_dlq_trade_n"

global survey_usetot_list "rx_num_mod_12m doc_num_mod_12m er_num_mod_12m hosp_num_mod_12m"

global survey_finance_list "cost_any_oop_12m cost_any_owe_12m cost_borrow_12m cost_refused_12m"

global survey_health_list "health_genflip_bin_12m health_notpoor_12m health_chgflip_bin_12m notbaddays_tot_12m notbaddays_phys_12m notbaddays_ment_12m nodep_screen_12m"

**Balance tables

*global credit_balance_list "any_bankruptcy_n any_lien_n any_judg_n any_coll_n any_dlq_trade_n any_med_coll_n any_nmed_coll_n have_score thick_file tot_open"

*global hospital_balance_list "ad_all ad_noner ad_er days_all numproc_all totcharge_all days_noner numproc_noner totcharge_noner days_er numproc_er totcharge_er"


** Supporting tables

global hospital_ad "ad_all ad_er ad_noner"
global hospital_noner "days_noner totcharge_noner numproc_noner"
global hospital_er "days_er totcharge_er numproc_er"

** OLD tables

*global credit_medical_list "any_med_coll_n tot_med_coll_bal"
*global credit_nonmed_list "any_nmed_coll_n tot_nmed_coll_bal"  

global survey_useext_list "rx_any_12m doc_any_12m er_any_12m hosp_any_12m"

global survey_access_list "usual_clinic_12m usual_doc_12m needmet_med_12m needmet_rx_12m not_er_noner_12m"
global survey_prevent_list "chl_chk_bin_12m dia_chk_bin_12m mam_chk_bin_12m pap_chk_bin_12m"

local list_P3_admin "alive"

*******************
* PREPARE DATA 
*******************
** This piece does the following:
***** 1. merges the various analytic data sets
***** 2. creates any other variables needs for the analysis
***** 3. saves a limited local dataset called "data_for_analysis.dta" in the Data subfolder

include SubPrograms/prepare_data.do 

* Filler matrices for assembling tables
matrix define filler = J(3,1,.)
matrix define filler2 = J(1,6,.)
matrix define filler3 = J(2,4,.)
matrix define filler4 = J(1,1,.)
matrix define filler5 = J(6,1,.)
matrix define filler6 = J(12,2,.)
matrix define filler7= J(13,2,.)
matrix define filler8=J(30, 2,.)

******************
* ANALYSIS
******************

******************
* TABLE I2
******************	

if `I2_switch' ==1 {

**Columns 1 and 2 : Balance of treatment and controls in full sample
		
	* Locals for running the balance table code
	
	local balance_table I2_2
	local balance_list "$baseline_list"
	local balance_subset "$baseline_list"
	local balance_matchvar ""
	local balance_condition ""
	local balance_controls "nnn*"
	local balance_weight "noweight"
	
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Run table
	include SubPrograms/tables_balance.do


****** Column 3 : Balance of treatment and controls in credit data

	* Locals for running the balance table code
	
	local balance_table I2_3
	local balance_list "$baseline_list"
	local balance_subset "$baseline_list"
	local balance_matchvar " "
	local balance_condition "& 1 == 1"
	local balance_controls "nnn*"
	local balance_weight "noweight"
		
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
			
	*Run table
	include SubPrograms/tables_balance.do 
	

****** Column 4: Balance of treatment and controls in 12m mail survey responders

	* Locals for running the balance table code
		
	local balance_table I2_4
	local balance_list "$baseline_list"
	local balance_subset "$baseline_list"
	local balance_matchvar "returned_12m"
	local balance_condition "& returned_12m==1"
	local balance_controls "ddd*"
	local balance_weight "weight_12m"
	
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Run table
	include SubPrograms/tables_balance.do


****** For footnote: Balance of treatment and controls in 12m mail survey sample

	* Locals for running the balance table code
		
	local balance_table I2_F
	local balance_list "$baseline_list"
	local balance_subset "$baseline_list"
	local balance_matchvar "sample_12m"
	local balance_condition "& sample_12m==1"
	local balance_controls "nnn*"
	local balance_weight "noweight"
	
	* Set up data
	clear
	use Data/data_for_analysis.dta
	
	* Run table
	include SubPrograms/tables_balance.do

	
****** Final row: response time

	* Set up data
	clear
	use Data/data_for_analysis.dta
	matrix TableI2_4_extra = J(3,4,.)
	matrix rownames TableI2_4_extra = "mail_to_response_12m" "se" "p"
	
				reg mail_to_response_12 if treatment==0 [pw=weight_12m]
				matrix TableI2_4_extra[1,1] = round(_b[_cons])
				matrix TableI2_4_extra[2,1] = round(e(rmse)*1000)/1000
				
				reg mail_to_response_12 treatment ddd* [pw=weight_12m], cluster(household_id)
				matrix TableI2_4_extra[1,4] = round(_b[treatment],.001)
				matrix TableI2_4_extra[2,4] = round(_se[treatment]*1000)/1000
				matrix TableI2_4_extra[3,4] = round(2*ttail(e(df_r), abs(_b[treatment]/_se[treatment]))*1000)/1000
				

	matrix TableI2_new = (TableI2_2[1...,1...],filler8[4...,2],TableI2_4[4...,2])\(filler8[1..3,1],filler[1...,1...],filler8[1..3,2],filler[1...,1...])\(TableI2_4[1..3,1],filler[1...,1...],filler[1...,1...],TableI2_4[1..3,2])\TableI2_4_extra[1...,1...]
	matrix list TableI2_new	

} // end of table switch

*******************************************************************


******************
* TABLE I3
******************

if `I3_switch'==1 {

****** TABLE I3: Descriptive statistics from 12m survey
		
		* Locals for running the means table code
	
		local table I3
		local list "$desc_var hhinc_mid_12m"
	    local condition "& sample_12m_resp==1 & treatment==0"
		local weight "weight_12m"
	    local shape "wide"
	
		* Set up data
		clear
		use Data/data_for_analysis.dta

      * Run table
		include SubPrograms/tables_means.do
		
****** Table I3_extra: descriptive statistics - full sample
		local table I3_extra
		local list "female_list older younger english_list zip_msa zip_hh_inc_list"
		local condition "& treatment==0"
		local weight "noweight"
		local shape "wide"
		
		* set up data
		use Data/data_for_analysis.dta, clear
		
		* run table
		include SubPrograms/tables_means.do
		
****** Table I3_extra12m: descriptive statistics - 12m responders
		local table I3_extra12m
		local list "female_list older younger english_list zip_msa zip_hh_inc_list"
		local condition "& sample_12m_resp==1 & treatment==0"
		local weight "weight_12m"
		local shape "wide"
		
		* set up data
		use Data/data_for_analysis.dta, clear
		
		* run table
		include SubPrograms/tables_means.do		
	
} // end of table switch


*******************************************************************

if `I4_switch'==1 {

******************
* TABLE I4 & SNAP FS
******************

	
****** TABLE I4: First stage estimates

	* Set up data
	clear
	use Data/data_for_analysis.dta

   * Hospital discharge data
	
	local table i4_1
	local list  "ohp_all_ever_admin ohp_std_ever_admin ohp_all_mo_admin ohp_all_end_admin "  
	local sample "sample_hdd"
	local controls "lll* nnn*"
	local weight "noweight"
					
	include SubPrograms/tables_firststage.do


	/* Credit report data
	
	local table i4_2
	local list  "ohp_all_ever_admin ohp_std_ever_admin ohp_all_mo_admin ohp_all_end_admin"  
	local sample "sample_credit"
	local controls "lll* nnn*"
	local weight "noweight"
	
	include SubPrograms/tables_firststage.do */

	* Survey responders
	
	local table i4_3
	local list  "ohp_all_ever_survey ohp_std_ever_survey ohp_all_mo_survey ohp_all_end_survey"  
	local sample "sample_12m_resp"
	local controls "ddd*"
	local weight "weight_12m"
	
	include SubPrograms/tables_firststage.do

use Data/data_for_analysis.dta, clear
foreach prog in tanf snap {
	foreach type in bin hh_amt {
		foreach sample in hdd 12m_resp {

			if "`sample'"!="12m_resp" {
			local table fs_`prog'_`type'_`sample'
			local list "postn_`prog'_`type'"
			local sample "sample_`sample'"
			local controls "nnn* lll* prenany_`prog'_`type'"
			local weight "noweight"
			}
			if "`sample'"=="12m_resp" {
			local table fs_`prog'_`type'_`sample'
			local list "postn_survey12m_`prog'_`type'"
			local sample "sample_`sample'"
			local controls "ddd* pren_survey12m_`prog'_`type'"
			local weight "weight_12m"
			}

	qui: include SubPrograms/tables_firststage.do		
			
			
		} // end of sample loop
	} // end of type loop
} // end of prog loop



*(Tablefs_snap_bin_hdd[1...,1...],Tablefs_snap_bin_credit[1...,1...],Tablefs_snap_bin_12m_resp[1...,1...])\(Tablefs_snap_amt_hdd[1...,1...],Tablefs_snap_amt_credit[1...,1...],Tablefs_snap_amt_12m_resp[1...,1...])
foreach prog in tanf snap {
matrix Tablefs_`prog'= (Tablefs_`prog'_bin_hdd[1..2,1...],filler6[1..2,1...],Tablefs_`prog'_bin_12m_resp[1..2,1...])\(Tablefs_`prog'_hh_amt_hdd[1..2,1...],filler6[1..2,1...],Tablefs_`prog'_hh_amt_12m_resp[1..2,1...])
}
} // end of table switch

*******************************************************************

******************
* TABLE I5
******************

if `I5_switch'==1 {

****** TABLE I5: Other sources of insurance

	* Set up data
	clear
	use Data/data_for_analysis.dta	

	local table I5
	local list  "ohp_all_at_12m ins_ohp_12m ins_private_12m ins_any_12m ins_months_12m ohp_all_mo_12m"
	local sample "sample_12m_resp"
	local controls "ddd*"
	local weight "weight_12m"
	
	include SubPrograms/tables_firststage.do
	
} // end of table switch


******************************************************************************
******************************************************************************
* ANALYSIS TABLES
* Will run survey portions of all tables separately
* Will assemble tables at the end
******************************************************************************
******************************************************************************

***************
***************
* SURVEY
***************
***************
local list_P1_survey "$survey_usetot_list"
local mult_inf_P1_survey ""
local spend_est_P1_survey 1

local list_P2_survey "$survey_finance_list"
local mult_inf_P2_survey ""
local spend_est_P2_survey 0

local list_P3_survey "$survey_health_list"
local mult_inf_P3_survey ""
local spend_est_P3_survey 0

local list_U1_survey "$survey_useext_list"
local mult_inf_U1_survey ""
local spend_est_U1_survey 0

local list_M1_1 "$survey_access_list"
local mult_inf_M1_1 ""
local spend_est_M1_1 0

local list_M1_2 "med_qual_bin_12m"
local mult_inf_M1_2 "no"
local spend_est_M1_2 0

local list_M1_3 "$survey_prevent_list"
local mult_inf_M1_3 ""
local spend_est_M1_3 0

local list_M1_4 "nonsmk_curr more_active"
local mult_inf_M1_4 ""
local spend_est_M1_4 0

local list_E1_1 "poshappiness_bin_12m"
local mult_inf_E1_1 "no"
local spend_est_E1_1 0


* Run Tables
foreach tab in P1_survey P2_survey P3_survey U1_survey M1_1 M1_2 M1_3 M1_4 E1_1 {
if ``tab'_switch' == 1 {
	local table "`tab'"
	
	local list_survey "`list_`tab''"
	local condition_survey "& sample_12m_resp==1"
	local controls_survey "ddd*"
    local weight_survey "weight_12m"
	
	local list_admin ""
	local condition_admin ""
	local controls_admin ""
	local weight_admin ""

	local add_vars "sample_12m_resp"
	local exclude_from_ste ""
	local mult_inf "`mult_inf_`tab''"
	local spend_est "`spend_est_`tab''"
	
use Data/data_for_analysis.dta, clear	

include SubPrograms/tables_analysis.do
	
	* Clean Up
		local mult_inf ""
		local add_vars ""
		local exclude_from_ste ""
		local spend_est 0
		
	}  // end of table switch	

} // end of survey tables loop


***************
***************
* HOSP ADMIN
***************
***************

local list_P3_admin "alive"
local mult_inf_P3_admin "no"

/*local list_U2_1 "$hospital_ad"
local mult_inf_U2_1 "no"
local list_U2_2 "$hospital_use_list"
local mult_inf_U2_2 ""
local list_U2_3 "$hospital_er"
local mult_inf_U2_3 ""
local list_U2_4 "$hospital_noner"
local mult_inf_U2_4 "" */


* Set variable controls
local var_controls "$hospital_use_list $hospital_ad $hospital_noner $hospital_er "

foreach var of local var_controls {
	local `var'_controls "pren_`var'"
		}
		
* Run tables
foreach tab in P3_admin { // U2_1 U2_2 U2_3 U2_4 commented out above and from this loop (data not publically available)

if ``tab'_switch' == 1 {
	
	local table "`tab'"
	
	local list_survey ""
	local condition_survey ""
	local controls_survey ""
    local weight_survey ""
	
	local list_admin "`list_`tab''"
	local condition_admin ""
	local controls_admin "nnn* lll*"
	local weight_admin "noweight"
	
	local add_vars ""
	local exclude_from_ste ""
	local mult_inf "`mult_inf_`tab''"
	local spend_est 0
	
use Data/data_for_analysis.dta, clear

	foreach var of local list_admin {
		rename postn_`var' `var'
		local add_vars "`add_vars' ``var'_controls'"
		}

include SubPrograms/tables_analysis.do

	* Clean Up
		local mult_inf ""
		local add_vars ""
		local exclude_from_ste ""
		
	}  // end of table switch	

} // end of hosp admin tables loop

* Clean up var controls
	foreach var of local var_controls {
		local `var'_controls ""
		}
	
	
	
******************
* Table 11
******************

if `New11_switch' == 1 {
include SubPrograms/survey_results_comparison.do
	}	

***********************************
***********************************
* ASSEMBLING TABLES
***********************************
***********************************

set linesize 255

if $New1==1{
matrix TableNew1 = TableI3_extra[1...,1]\filler4[1...,1...]\TableI3_extra12m[1...,1]\filler4[1...,1...]\TableI3[1...,1]
di "full sample / 12m responders / 12m responders"
}

if $New2==1{
matrix TableNew2 = TableI2_new[31..36,1...]
}

if $New3==1{
matrix TableNew3 = (Tablei4_1[1..12,1...], filler6[1..12,1...], Tablei4_3[1..12,1...])\(filler6[1...,1...],filler6[1...,1...],(TableI5[10..12,1...]\TableI5[7..9,1...]\TableI5[4..6,1...]\TableI5[1..3,1...]))\(Tablei4_1[13,1...],filler7[13,1...],Tablei4_3[13,1...])
matrix list TableNew3
matrix TableNew3 = TableNew3[1..2,1...]\TableNew3[4..5,1...]\TableNew3[7..8,1...]\TableNew3[10..11,1...]\TableNew3[13..14,1...]\TableNew3[16..17,1...]\TableNew3[19..20,1...]\TableNew3[22..23,1...]\TableNew3[25,1...]
matrix list TableNew3
matrix TableNew3 = TableNew3[1..16,1...]\Tablefs_tanf\Tablefs_snap\TableNew3[17,1...]
matrix list TableNew3
matrix rownames TableNew3 = "evermedicaid" "se" "everOHPSTD" "se" "#mthsOHP" "se" "ohp_all_at_12m" "se" "anyins" "se" "privins" "se" "ohpcurrent-svy" "se" "ohpcurrent-adm" "se" "tanf_bin" "se" "tanf_hh_amt" "se" "snap_bin" "se" "snap_hh_amt" "se" "N"
}


if $New5==1{
matrix TableNew5 = (TableU1_survey[3..12,1...],TableP1_survey[3..12,1...])\(filler3,spend_est_P1_survey)
matrix rownames TableNew5 = "rx" "se" "doc" "se" "er" "se" "hosp" "se" "ste" "se" "spendest" "se"
}

if $New6==1{
matrix TableNew6 = TableM1_3[3..12,1...]
}

if $New8==1{
matrix TableNew8 = TableP2_survey[3..12,1...]
}

if $New9==1{
matrix TableNew9 = TableP3_admin[1..2, 1...]\TableP3_survey[3..18,1...]
}

if $New10==1{
matrix TableNew10 = TableM1_1[3..14,1...]\TableM1_2[3..4,1...]\TableE1_1[3..4,1...]
}

if $New11==1{
matrix TableNew11 = STE
}


cap log close


***********************************
* Printing results to separate file
***********************************

if $iterations<10000 {
log using all_tables.log, text replace
set linesize 255
}

if $iterations>=10000 {
log using all_tables_multinf.log, text replace
set linesize 255
}

qui {

noi: di _newline
noi: di "***********MAIN TABLES***********"
noi: di _newline

	if $New1==1{
		noi: di _newline
		noi: matrix list TableNew1, title(Table 1: Demographic characteristics of study population (controls))
		noi: di "survey: N: `TableI3_N'"
		noi: di "all: N: `TableI3_extra_N'"
		}
		
	if $New2==1{
		noi: di _newline
		noi: matrix list TableNew2, title(Table 2: Balance)
		}

	if $New3==1{
		noi: di _newline
		noi: matrix list TableNew3, title(Table 3: First Stage Estimates)
		}


	if $New5==1{
		noi: di _newline
		noi: matrix list TableNew5, title(Table 5: Impact of Health Insurance on Health Care Utilization (Survey Data))
		}

	if $New6==1{
		noi: di _newline
		noi: matrix list TableNew6, title(Table 6: Compliance with recommended preventive care, Survey data)
		}


	if $New8==1{
		noi: di _newline
		noi: matrix list TableNew8, title(Table 8: Impact of Health Insurance on Financial Well Being (Survey Data))
		}

	if $New9==1{
		noi: di _newline
		noi: matrix list TableNew9, title(Table 9: Impact of Health Insurance on Health)
		}

	if $New10==1{
		noi: di _newline
		noi: matrix list TableNew10, title(Table 10: Potential Mechanisms for Improved Health, Survey Data)
		}

	if $New11==1{
		noi: di _newline
		noi: matrix list TableNew11, title(Table 11: Earlier Surveys)
		}

	} // end of quiet loop	
		
	log close
	exit
	
