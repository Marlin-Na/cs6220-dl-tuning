#!/usr/bin/env bash


# ========== Train 200 ====================

TRAIN_DATA_DIR=raw_data/train200/
TEST_DATA_DIR=raw_data/test40/
MODEL_DIR=m_train_200

mkdir $MODEL_DIR
mkdir $MODEL_DIR/trained_model
rm $MODEL_DIR/test_res 2> /dev/null
rm $MODEL_DIR/test_outlier 2> /dev/null

python retrain.py --image_dir=$TRAIN_DATA_DIR \
       --bottleneck_dir=$MODEL_DIR/bottleneck/ \
       --how_many_training_steps=2000 \
       --testing_percentage 20 \
       --output_graph=$MODEL_DIR/trained_model/retrained_graph.pb \
       --output_labels=$MODEL_DIR/trained_model/retrained_labels.txt \
       --summaries_dir=$MODEL_DIR/summaries > $MODEL_DIR/train.log 2> $MODEL_DIR/train.err

for file in $TEST_DATA_DIR/*; do
    class=$(
        python label_image.py \
               --graph=$MODEL_DIR/trained_model/retrained_graph.pb \
               --labels=$MODEL_DIR/trained_model/retrained_labels.txt \
               --input_layer=Placeholder \
               --output_layer=final_result \
               --image=$file 2> $MODEL_DIR/test.err | head -n 1
         )
    echo $file $class >> $MODEL_DIR/test_res
done


# ========== Train 200 (resize) ====================

TRAIN_DATA_DIR=raw_data/train200_resize/
TEST_DATA_DIR=raw_data/test40_resize/
MODEL_DIR=m_train_200_resize

mkdir $MODEL_DIR
mkdir $MODEL_DIR/trained_model
rm $MODEL_DIR/test_res 2> /dev/null
rm $MODEL_DIR/test_outlier 2> /dev/null

python retrain.py --image_dir=$TRAIN_DATA_DIR \
       --bottleneck_dir=$MODEL_DIR/bottleneck/ \
       --how_many_training_steps=2000 \
       --testing_percentage 20 \
       --output_graph=$MODEL_DIR/trained_model/retrained_graph.pb \
       --output_labels=$MODEL_DIR/trained_model/retrained_labels.txt \
       --summaries_dir=$MODEL_DIR/summaries > $MODEL_DIR/train.log 2> $MODEL_DIR/train.err

for file in $TEST_DATA_DIR/*; do
    class=$(
        python label_image.py \
               --graph=$MODEL_DIR/trained_model/retrained_graph.pb \
               --labels=$MODEL_DIR/trained_model/retrained_labels.txt \
               --input_layer=Placeholder \
               --output_layer=final_result \
               --image=$file 2> $MODEL_DIR/test.err | head -n 1
         )
    echo $file $class >> $MODEL_DIR/test_res
done

# ========== Train 2000 ====================

TRAIN_DATA_DIR=raw_data/train2000/
TEST_DATA_DIR=raw_data/test400/
MODEL_DIR=m_train_2000

mkdir $MODEL_DIR
mkdir $MODEL_DIR/trained_model
rm $MODEL_DIR/test_res 2> /dev/null
rm $MODEL_DIR/test_outlier 2> /dev/null

python retrain.py --image_dir=$TRAIN_DATA_DIR \
       --bottleneck_dir=$MODEL_DIR/bottleneck/ \
       --how_many_training_steps=2000 \
       --testing_percentage 20 \
       --output_graph=$MODEL_DIR/trained_model/retrained_graph.pb \
       --output_labels=$MODEL_DIR/trained_model/retrained_labels.txt \
       --summaries_dir=$MODEL_DIR/summaries > $MODEL_DIR/train.log 2> $MODEL_DIR/train.err

for file in $TEST_DATA_DIR/*; do
    class=$(
        python label_image.py \
               --graph=$MODEL_DIR/trained_model/retrained_graph.pb \
               --labels=$MODEL_DIR/trained_model/retrained_labels.txt \
               --input_layer=Placeholder \
               --output_layer=final_result \
               --image=$file 2> $MODEL_DIR/test.err | head -n 1
         )
    echo $file $class >> $MODEL_DIR/test_res
done

# ========== Train 2000 (resize) ====================

TRAIN_DATA_DIR=raw_data/train2000_resize/
TEST_DATA_DIR=raw_data/test400_resize/
MODEL_DIR=m_train_2000_resize

mkdir $MODEL_DIR
mkdir $MODEL_DIR/trained_model
rm $MODEL_DIR/test_res 2> /dev/null
rm $MODEL_DIR/test_outlier 2> /dev/null

python retrain.py --image_dir=$TRAIN_DATA_DIR \
       --bottleneck_dir=$MODEL_DIR/bottleneck/ \
       --how_many_training_steps=2000 \
       --testing_percentage 20 \
       --output_graph=$MODEL_DIR/trained_model/retrained_graph.pb \
       --output_labels=$MODEL_DIR/trained_model/retrained_labels.txt \
       --summaries_dir=$MODEL_DIR/summaries > $MODEL_DIR/train.log 2> $MODEL_DIR/train.err

for file in $TEST_DATA_DIR/*; do
    class=$(
        python label_image.py \
               --graph=$MODEL_DIR/trained_model/retrained_graph.pb \
               --labels=$MODEL_DIR/trained_model/retrained_labels.txt \
               --input_layer=Placeholder \
               --output_layer=final_result \
               --image=$file 2> $MODEL_DIR/test.err | head -n 1
         )
    echo $file $class >> $MODEL_DIR/test_res
done


