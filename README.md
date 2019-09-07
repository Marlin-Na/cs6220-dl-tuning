
## Preprocess

The resized images were generated with:

```
mkdir raw_data/train200_resize
mkdir raw_data/train200_resize/cat
mkdir raw_data/train200_resize/dog
mkdir raw_data/train2000_resize
mkdir raw_data/train2000_resize/cat
mkdir raw_data/train2000_resize/dog
mkdir raw_data/test40_resize
mkdir raw_data/test400_resize
mkdir raw_data/test_outlier_resize
python resize_img.py --dim 50 --indir raw_data/train200/cat  --outdir raw_data/train200_resize/cat
python resize_img.py --dim 50 --indir raw_data/train200/dog  --outdir raw_data/train200_resize/dog
python resize_img.py --dim 50 --indir raw_data/train2000/cat --outdir raw_data/train2000_resize/cat
python resize_img.py --dim 50 --indir raw_data/train2000/dog --outdir raw_data/train2000_resize/dog
python resize_img.py --dim 50 --indir raw_data/test40        --outdir raw_data/test40_resize
python resize_img.py --dim 50 --indir raw_data/test400       --outdir raw_data/test400_resize
python resize_img.py --dim 50 --indir raw_data/test_outlier  --outdir raw_data/test_outlier_resize
```

## Training

```
python retrain.py --image_dir=raw_data/train200/ --bottleneck_dir=bottleneck/ --how_many_training_steps=500 --output_graph=trained_model/retrained_graph.pb --output_labels=trained_model/retrained_labels.txt --summaries_dir=summaries
```

## Reference

- https://github.com/tensorflow/hub/tree/master/examples/image_retraining
