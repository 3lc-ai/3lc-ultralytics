from tlc_ultralytics import YOLO, Settings

DATA_YAML = "C:/Project/datasets/hand-keypoints/data.yaml"

# fmt: off
YOLO_HAND_LANDMARKS = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MMC",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGERMCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGERPIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}

YOLO_HAND_SKELETON = [
    0, 1,
    0, 5,
    0, 17,
    1, 2,
    2, 3,
    3, 4,
    5, 9,
    5, 6,
    6, 7,
    7, 8,
    9, 13,
    9, 10,
    10, 11,
    11, 12,
    13, 17,
    13, 14,
    14, 15,
    15, 16,
    17, 18,
    18, 19,
    19, 29,
]
# fmt: on

if __name__ == "__main__":
    model = YOLO("yolo11m-pose.pt")
    settings = Settings(
        image_embeddings_dim=2,
        sampling_weights=False,
        collect_loss=True,
        kpt_names=list(YOLO_HAND_LANDMARKS.values()),
        lines=YOLO_HAND_SKELETON,
        run_name="train-hands-kpts-example",
        project_name="3lc-yolo-hands",
    )

    model.train(
        data=DATA_YAML,
        settings=settings,
        epochs=50,
        workers=8,
    )
