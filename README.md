### Training

To train the model on your dataset:

```bash
python train.py \
    --data_path "/path/to/your/data" \
    --sam_checkpoint "./path/to/sam_vit_l.pth" \
    --epochs 20 \
    --batch_size 1 \
    --iter_point 8
```

*Key parameters include `--data_path`, `--sam_checkpoint`, and `--iter_point`.*

### Testing

To evaluate the trained model:

```bash
python test.py \
    --data_path "/path/to/test/data" \
    --sam_checkpoint "/path/to/trained_model.pth" \
    --iter_point 8 \
    --save_pred True \
    --metrics iou dice f1 pre rec
```
