#!/usr/bin/env bash

OUTLIER_DATA_DIR=raw_data/test_outlier

#### ===================================

MODEL_DIR=m_train_200

rm $MODEL_DIR/test_outlier 2> /dev/null
for file in $OUTLIER_DATA_DIR/*; do
    class=$(
        python label_image.py \
              --graph=$MODEL_DIR/trained_model/retrained_graph.pb \
              --labels=$MODEL_DIR/trained_model/retrained_labels.txt \
              --input_layer=Placeholder \
              --output_layer=final_result \
              --image=$file 2> $MODEL_DIR/test_outlier.err | head -n 1
         )
    echo $file $class >> $MODEL_DIR/test_outlier
done


#### ===================================

MODEL_DIR=m_train_200_resize

rm $MODEL_DIR/test_outlier 2> /dev/null
for file in $OUTLIER_DATA_DIR/*; do
    class=$(
        python label_image.py \
               --graph=$MODEL_DIR/trained_model/retrained_graph.pb \
               --labels=$MODEL_DIR/trained_model/retrained_labels.txt \
               --input_layer=Placeholder \
               --output_layer=final_result \
               --image=$file 2> $MODEL_DIR/test_outlier.err | head -n 1
         )
    echo $file $class >> $MODEL_DIR/test_outlier
done

#### ===================================

MODEL_DIR=m_train_2000

rm $MODEL_DIR/test_outlier 2> /dev/null
for file in $OUTLIER_DATA_DIR/*; do
    class=$(
        python label_image.py \
               --graph=$MODEL_DIR/trained_model/retrained_graph.pb \
               --labels=$MODEL_DIR/trained_model/retrained_labels.txt \
               --input_layer=Placeholder \
               --output_layer=final_result \
               --image=$file 2> $MODEL_DIR/test_outlier.err | head -n 1
         )
    echo $file $class >> $MODEL_DIR/test_outlier
done

#### ===================================

MODEL_DIR=m_train_2000_resize

rm $MODEL_DIR/test_outlier 2> /dev/null
for file in $OUTLIER_DATA_DIR/*; do
    class=$(
        python label_image.py \
               --graph=$MODEL_DIR/trained_model/retrained_graph.pb \
               --labels=$MODEL_DIR/trained_model/retrained_labels.txt \
               --input_layer=Placeholder \
               --output_layer=final_result \
               --image=$file 2> $MODEL_DIR/test_outlier.err | head -n 1
         )
    echo $file $class >> $MODEL_DIR/test_outlier
done

