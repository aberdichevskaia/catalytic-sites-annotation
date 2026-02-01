#!/bin/bash

#proteins=("1W84_A" "2JKX_A" "2ZB1_A" "3AIR_A" "3FHR_A" "3FLS_A" "3OD6_X" "3WWC_A" "3WWE_A" "4F9W_A" "5XYX_A" "6ANL_A" "6RFO_A" "6XLT_A" "6ZQS_A" "7S4X_A" "Q588Z2_A" "Q8BGG7_A" "Q8CIN4_A" "Q8VZ10_A" "Q9LCQ7_A" "Q9VCE9_A" ) classic
#proteins=("6ZQS_A" "6RFO_A" "Q8CIN4_A" "5XYX_A" "3OD6_X" "4F9W_A" "6ANL_A" "3FLS_A" "2ZB1_A" "1W84_A") worst
#proteins=("2YA0_A" "A0A7J6K7Y0_A" "A0R6E0_A" "A5YVK8_A" "A6BMK7_A" "3ZIZ_A" "8A5L_B" "D3GDK4_A" "D3ZSZ3_A" "D5EY13_A" "5YJ9_D" "7MP8_A" "E0W1I1_A" "6L42_A" "6JOQ_A" "6K9A_A" "8CTB_A" "8CYI_A" "O14815_A" "1ZIV_A" "O14936_A" "2VKY_B" "2VNL_A" "2AMG_A" "4JR7_A" "3POO_A" "5IZF_A" "6C0U_A" "2HWE_2" "3QQF_A" "3QZG_A" "6YPS_A" "6YQK_A" "Q0KEN8_A" "Q0TA53_A" "Q1I6M7_A" "Q2FIN3_A" "Q2G045_A" "Q2KU93_A" "Q2T1P5_A" "Q2U492_A" "Q31TY7_A") #best
#proteins=("4XW6_A" "1E40_A" "3CUI_A" "P08413_A")
#proteins=("4Q6X_A" "B0BNF9_A" "O53547_A" "Q9LSS3_A" "A0AVT1_A" "A2VDM8_A" "P09938_A" "P0ABN1_A")
proteins=("A2VDM8_A" "C7NBY4_A" "O00391_A" "P06744_A" "P0DTD1_A" "Q53079_A")
python_script="/home/iscb/wolfson/annab4/ScanNet_Ub/predict_bindingsites.py"

error_log="prediction_errors.log"
> "$error_log"
folds=(1 2 3 4 5)
for fold in "${folds[@]}" 
    do
    mode="cv_catalytic${fold}"
    echo "Start for $mode"
    for protein in "${proteins[@]}"
    do
        echo "Launch for $protein"
        if python "$python_script" "$protein" --assembly --mode "$mode"; then
            echo "Ready for $protein"
        else 
            echo "Error for $protein" | tee -a "$error_log"
        fi
    done
    echo "Ready for $mode"
done

echo "All predictions are ready."
