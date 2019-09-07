
## Training

```
python retrain.py --image_dir=raw_data/train200/ --bottleneck_dir=bottleneck/ --how_many_training_steps=500 --output_graph=trained_model/retrained_graph.pb --output_labels=trained_model/retrained_labels.txt --summaries_dir=summaries
```

## Reference

- https://github.com/tensorflow/hub/tree/master/examples/image_retraining
