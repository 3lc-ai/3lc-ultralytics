from tlc_ultralytics import YOLO, Settings

# fmt: off
COCO_PERSON_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

COCO_PERSON_SKELETON = [
    3, 1, 4, 2, 1, 0, 0, 2, 5, 6,
    5, 7, 6, 8, 7, 9, 8, 10, 11, 12,
    11, 13, 12, 14, 13, 15, 14, 16,
    5, 11, 6, 12
]
# fmt: on

if __name__ == "__main__":
    model = YOLO("yolo11n-pose.pt")
    settings = Settings(
        image_embeddings_dim=2,
        sampling_weights=False,
        collect_loss=True,
        kpt_names=COCO_PERSON_KEYPOINT_NAMES,
        lines=COCO_PERSON_SKELETON,
        run_name="3lc-yolo-pose-train-example",
        project_name="3lc-yolo-examples",
    )

    model.train(
        data="coco8-pose.yaml",
        settings=settings,
        epochs=10,
        workers=0,
    )
