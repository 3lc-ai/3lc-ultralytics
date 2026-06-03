# Active Labeling

Active Labeling is a technique that can significantly accelerate dataset labeling workflows. This guide demonstrates how to combine 3LC and Ultralytics YOLO models to efficiently create larger labeled datasets and train improved models.

## The Approach

1. **Start with an unlabeled YOLO dataset** and create an initial `tlc.Table`:

    ```python
    from tlc_ultralytics import create_tables_from_yaml_file

    tables = create_tables_from_yaml_file(
        "/path/to/yolo/dataset.yaml",
        task="detect",
        project_name="Active Labeling Example",
        splits=("train",),
    )
    table = tables["train"]
    ```

    `create_tables_from_yaml_file` parses the YOLO dataset YAML — including the `names` map — and calls
    `tlc.Table.from_yolo_url()` once per split. If you instead have a single folder/text file of images and the
    category map at hand, you can call `tlc.Table.from_yolo_url(images_url=..., categories=..., task="detect")`
    directly.

    Open the `tlc.Table` in the [3LC Dashboard](dashboard.3lc.ai) and manually label a diverse set of images. Aim to label images that represent different object categories and scene variations. Set the weights of labeled images to one and unlabeled images to zero.

    > **Tip**: Create a column with random numbers and sort by this column to ensure you select a diverse set of images for labeling.

2. **Train a model on the labeled data** and validate on the same dataset:

    ```python
    import tlc

    from tlc_ultralytics import Settings, YOLO

    model = YOLO("yolo11m.pt")

    # Train and validate on the train split
    table = tlc.Table.from_url("<3lc project root>/Active Labeling Example/datasets/train/tables/initial")
    tables = {"train": table, "val": table}

    # Train only on labeled (non-zero weighted) samples, but predict on all rows during validation
    settings = Settings(exclude_zero_weight_training=True, collect_val_only=True)

    model.train(tables=tables, epochs=10, settings=settings)
    ```

    Notice how we set `collect_val_only=True` to only validate and collect metrics on the validation split. This avoids a second pass on the training data.

    > **Tip**: To prevent overfitting, monitor prediction quality on unseen data and adjust the number of epochs accordingly. For better validation, consider labeling a diverse hold-out validation set to ensure overfitting is avoided.

3. **Refine predictions in the 3LC Dashboard** by adjusting filters for columns such as `Confidence` and `IoU` to capture high-quality predictions. Review at least several dozen images to verify prediction quality and estimate error rates. Use the batch assignment dialog to assign verified predictions to the dataset. Set the weight of these samples to one and commit the changes.

You can repeat the last two steps iteratively to gradually expand your high-quality labeled dataset, which will improve subsequent model training.

## Additional Resources

The following resources provide more detailed walkthroughs beyond the steps outlined above:

- [Scaling an Instance Segmentation Dataset with Active Labeling in 3LC](https://3lc.ai/scaling-an-instance-segmentation-dataset-with-active-labeling-in-3lc/): This article demonstrates how to produce tens of thousands of high-quality labels in a single day.
- [Breaking the Human Accuracy Barrier in Computer Vision Labeling](https://3lc.ai/breaking-the-human-accuracy-barrier-in-computer-vision-labeling/): While this article focuses on Grounding DINO, it illustrates the Active Labeling methodology and serves as another valuable reference.
