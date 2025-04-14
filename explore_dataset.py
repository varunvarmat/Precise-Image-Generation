import fiftyone as fo
import fiftyone.core.dataset as fod

# Load your dataset (adjust to your folder format)
dataset = fod.Dataset.from_dir(
    dataset_type=fo.types.ImageDirectory,  # or use fo.types.ImageClassificationDirectoryTree
    dataset_dir="data/images"
)

session = fo.launch_app(dataset)