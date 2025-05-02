#!/bin/bash

# --- Load Environment Variables ---
ENV_FILE="$(dirname "$0")/.env"

if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment variables from ${ENV_FILE}"
    # Use set -a to export all variables sourced from the file
    set -a
    source "${ENV_FILE}"
    set +a
else
    echo "Error: Environment file ${ENV_FILE} not found."
    echo "Please create it based on .env.example."
    exit 1
fi

# --- Validate Required Variables ---
required_vars=("GCP_PROJECT" "TPU_NAME" "ZONE" "ACCELERATOR_TYPE" "RUNTIME_VERSION")
missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "Error: Missing required environment variables in ${ENV_FILE}:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

# --- Configuration Summary ---
echo "Using configuration:"
echo "  Project:         ${GCP_PROJECT}"
echo "  TPU Name:        ${TPU_NAME}"
echo "  Zone:            ${ZONE}"
echo "  Accelerator:     ${ACCELERATOR_TYPE}"
echo "  Runtime Version: ${RUNTIME_VERSION}"
if [ -n "${EMAIL}" ]; then
    echo "  Notification Email: ${EMAIL}"
else
    echo "  Notification Email: (Not set)"
fi
echo # Add a newline for separation

# # --- 0. Attempt to delete existing queued resource (ignore errors if not found) ---
# echo "Attempting to delete existing queued resource ${TPU_NAME}..."
# gcloud compute tpus queued-resources delete ${TPU_NAME} \
#     --project ${GCP_PROJECT} \
#     --zone ${ZONE} \
#     --quiet || true # Continue even if delete fails (e.g., resource not found)

# --- 1. Request Queued Resource ---
echo "Requesting TPU Queued Resource: ${TPU_NAME} (${ACCELERATOR_TYPE}) in ${ZONE}..."
gcloud compute tpus queued-resources create ${TPU_NAME} \
    --project ${GCP_PROJECT} \
    --zone ${ZONE} \
    --node-id ${TPU_NAME} \
    --accelerator-type ${ACCELERATOR_TYPE} \
    --runtime-version ${RUNTIME_VERSION}

# Check if the create command was successful (basic check)
if [ $? -ne 0 ]; then
    echo "Failed to submit queued resource request. Exiting."
    exit 1
fi

echo "Queued resource request submitted. Waiting for allocation..."

# --- 2. Wait Loop ---
while true; do
    # Construct the full resource name for filtering
    FULL_RESOURCE_NAME="projects/${GCP_PROJECT}/locations/${ZONE}/queuedResources/${TPU_NAME}"

    # Use 'alpha' for queued-resources list and extract the state
    # Filter using the full resource name identifier
    TPU_STATE=$(gcloud alpha compute tpus queued-resources list \
        --project ${GCP_PROJECT} \
        --zone ${ZONE} \
        --filter="name=${FULL_RESOURCE_NAME}" \
        --format="value(state.state)") # Use state.state for potentially nested state info

    # Check if state is empty or indicates an issue
    if [[ -z "$TPU_STATE" ]]; then
       echo "Could not retrieve TPU state. Waiting and retrying..."
       sleep 60
       continue
    fi

    echo "Current TPU state: ${TPU_STATE}"

    # Check if the TPU is still waiting for resources
    # Common states are WAITING_FOR_RESOURCES, PROVISIONING, ACTIVE, FAILED etc.
    if [[ "$TPU_STATE" != "WAITING_FOR_RESOURCES" && "$TPU_STATE" != "CREATING" && "$TPU_STATE" != "PROVISIONING" ]]; then
        echo "TPU ${TPU_NAME} is no longer waiting/provisioning. Final State: ${TPU_STATE}"

        # Send email notification if email is set and state is ACTIVE (or similar success state)
        if [[ -n "$EMAIL" && ("$TPU_STATE" == "ACTIVE" || "$TPU_STATE" == "READY") ]]; then
            echo "Sending notification email to ${EMAIL}..."
            # Ensure msmtp is configured or replace with your preferred mail command
            echo -e "Subject: TPU ${TPU_NAME} (${ACCELERATOR_TYPE}) Assignment Notification\n\nThe TPU ${TPU_NAME} has been assigned and is now in state: ${TPU_STATE}." | msmtp "$EMAIL"
        elif [[ -n "$EMAIL" ]]; then
             echo "TPU entered a non-waiting state (${TPU_STATE}), but not sending email for this state."
        fi
        break # Exit the loop
    fi

    echo "TPU ${TPU_NAME} is still in state ${TPU_STATE}. Checking again in 60 seconds..."
    sleep 60
done

# --- 3. SSH into the TPU (once ready) ---
echo "Attempting to SSH into TPU: ${TPU_NAME}..."
echo "Run the following command:"
echo "gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${GCP_PROJECT} --zone ${ZONE}"

# Optionally, you can try to SSH directly here, but it might fail if the TPU isn't fully ready
# gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${GCP_PROJECT} --zone ${ZONE}
