import datetime
from google.cloud import firestore

# IMPORTANT: Project ID is explicitly hardcoded as a string
PROJECT_ID = "qwiklabs-gcp-04-17cb16fe3675"
COLLECTION_NAME = "bic_mac_experiments"


def seed_firestore():
    print(f"Connecting to Firestore for project: {PROJECT_ID}")
    db = firestore.Client(project=PROJECT_ID)

    sample_experiments = [
        {
            "experiment_id": "exp-001-unet-baseline",
            "model_name": "3D-UNet-Base",
            "modalities": ["NAC-PET", "MRI"],
            "mae": 19.5,
            "psnr": 30.2,
            "ssim": 0.895,
            "notes": "Initial baseline model trained on NAC-PET and MRI slices.",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        {
            "experiment_id": "exp-002-gan-attenuation",
            "model_name": "CrossModal-cGAN",
            "modalities": ["NAC-PET", "MRI", "2D-Topogram"],
            "mae": 16.8,
            "psnr": 32.7,
            "ssim": 0.924,
            "notes": "Added 2D Topogram modality. Significant MAE reduction.",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        {
            "experiment_id": "exp-003-diffusion-synth",
            "model_name": "LatentDiffusion-PseudoCT",
            "modalities": ["NAC-PET", "MRI", "2D-Topogram"],
            "mae": 14.3,
            "psnr": 34.1,
            "ssim": 0.948,
            "notes": "Current state-of-the-art candidate model.",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    ]

    collection_ref = db.collection(COLLECTION_NAME)
    for exp in sample_experiments:
        doc_ref = collection_ref.document(exp["experiment_id"])
        doc_ref.set(exp)
        print(f"Seeded document: {exp['experiment_id']}")

    print("Firestore seeding complete!")


if __name__ == "__main__":
    seed_firestore()
