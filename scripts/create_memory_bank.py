import vertexai

PROJECT_ID = "qwiklabs-gcp-04-17cb16fe3675"
LOCATION = "us-central1"


def create_memory_bank():
    print(f"Initializing Vertex AI for project={PROJECT_ID}, location={LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
    print("Creating Memory Bank instance...")
    memory_bank = client.agent_engines.create()

    resource_name = memory_bank.api_resource.name
    memory_bank_id = resource_name.split("/")[-1]
    print(f"✅ Memory Bank Instance Created!")
    print(f"MEMORY_BANK_ID: {memory_bank_id}")
    print(f"Resource Name : {resource_name}")
    return memory_bank_id


if __name__ == "__main__":
    create_memory_bank()
