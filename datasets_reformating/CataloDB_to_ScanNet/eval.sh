versions=(1 2 3 4 5 6)

for version in "${versions[@]}"
do
    echo "Evaluating version $version"
    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/datasets_reformating/CataloDB_to_ScanNet/eval.py find-threshold \
      --val_pickles \
      /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold5_v$version/validation_results.pkl \
      /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold2_v$version/validation_results.pkl \
      /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold3_v$version/validation_results.pkl \
      /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold4_v$version/validation_results.pkl \
      /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold5_v$version/validation_results.pkl \
    --out_json /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold5_v$version/best_thr.json


    python /home/iscb/wolfson/annab4/catalytic-sites-annotation/datasets_reformating/CataloDB_to_ScanNet/eval.py eval-f1 \
    --pickle /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold5_v$version/test_results.pkl \
    --thr_json /home/iscb/wolfson/annab4/scannet_retrains/ablate11_esm2_3B_CataloDB_fold5_v$version/best_thr.json

    echo "Finished evaluating version $version"
done