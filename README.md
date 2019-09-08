
This repository contains the code to train a neural network for cat-dog image classification.

The `retrain.py` and `label_image.py` scripts were downloaded from:

- https://github.com/tensorflow/hub/tree/master/examples/image_retraining
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image

You can install the dependency with

```
pip install -r requirements.txt
```

`raw_data` folder contains eight different datasets with varying volume and resolution.

`run_all.sh` will retrain the Inception V3 model with the four training datasets and test them
with the corresponding test datasets. It will create the following folders which contains
both the model and the log.

- `m_train_200`
- `m_train_2000`
- `m_train_200_resize`
- `m_train_2000_resize`

`test_outlier` will run the four classifier on the ten animated cat/dog images in `raw_data`
and save the results to the corresponding model folder.

The `retrain.py` and `label_image.py` may also be used independently. Sample usage:

```
python retrain.py \
--image_dir=raw_data/train200/ \
--bottleneck_dir=bottleneck/ \
--how_many_training_steps=2000 \
--output_graph=trained_model/retrained_graph.pb \
--output_labels=trained_model/retrained_labels.txt \
--summaries_dir=summaries
```

```
python label_image.py \
--graph=trained_model/retrained_graph.pb --labels=trained_model/retrained_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=raw_data/test40/cat.5000.jpg 2> /dev/null
```

## Notes on datasets

The datasets were a subset of [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset on Kaggle.

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

## Reference

- https://github.com/tensorflow/hub/tree/master/examples/image_retraining
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image

