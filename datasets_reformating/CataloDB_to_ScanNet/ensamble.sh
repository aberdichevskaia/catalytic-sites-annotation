versions=(1 2 3 4 5 6)

for ver in "${versions[@]}"
do
    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/datasets_reformating/CataloDB_to_ScanNet/ensamble.py \
    --pickle_paths \
        /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold1_v$ver/test_results.pkl \
        /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold2_v$ver/test_results.pkl \
        /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold3_v$ver/test_results.pkl \
        /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold4_v$ver/test_results.pkl \
        /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold5_v$ver/test_results.pkl \
    --out_dir /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_ensemble_out_v$ver \
    --ensemble_name CataloDB_EnsembleMean
done


