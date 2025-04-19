import json
import boto3
from datetime import datetime, timedelta

# Initialize AWS clients
cloudwatch = boto3.client("cloudwatch")
sagemaker_runtime = boto3.client("sagemaker-runtime")

# Replace with your actual SageMaker endpoint name
ENDPOINT_NAME = "RF-custom-model-2025-03-29-20-28-00"

# Required CloudWatch metrics
METRIC_MAP = {
    "CPUUtilization": {"Namespace": "AWS/EC2", "Metric": "CPUUtilization", "Stat": "Average"},
    "DiskReadOps": {"Namespace": "AWS/EC2", "Metric": "DiskReadOps", "Stat": "Sum"},
    "DiskWriteOps": {"Namespace": "AWS/EC2", "Metric": "DiskWriteOps", "Stat": "Sum"},
    "NetworkIn": {"Namespace": "AWS/EC2", "Metric": "NetworkIn", "Stat": "Sum"},
    "NetworkOut": {"Namespace": "AWS/EC2", "Metric": "NetworkOut", "Stat": "Sum"}
}

def get_instance_metrics(instance_id):
    """Fetches the latest CloudWatch metrics for the given EC2 instance."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=5)  # Fetch last 5 minutes of data

    metrics_data = {}
    
    for key, metric in METRIC_MAP.items():
        response = cloudwatch.get_metric_statistics(
            Namespace=metric["Namespace"],
            MetricName=metric["Metric"],
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,  # 5-minute period
            Statistics=[metric["Stat"]],
            Unit="Percent" if key == "CPUUtilization" else "Count"
        )
        
        # Extract the most recent datapoint
        data_points = response.get("Datapoints", [])
        if data_points:
            latest_value = sorted(data_points, key=lambda x: x["Timestamp"], reverse=True)[0][metric["Stat"]]
            metrics_data[key] = latest_value
        else:
            metrics_data[key] = 0  # Default to zero if no data

    return metrics_data


def lambda_handler(event, context):
    try:
        # Parse input payload
        if "body" in event:
            input_payload = json.loads(event["body"])  # If called via API Gateway
        else:
            input_payload = event  # If called directly (for local testing)

        # Validate instance ID
        instance_id = input_payload.get("InstanceId")
        if not instance_id:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required field: InstanceId"})
            }

        # Fetch metrics from CloudWatch
        instance_metrics = get_instance_metrics(instance_id)

        # Convert data into the correct format (list of values)
        payload = json.dumps([list(instance_metrics.values())])

        # Invoke SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload
        )

        # Read response body
        result = json.loads(response["Body"].read().decode())

        # Extract prediction
        #predicted_type = result.get("predicted_instance_type") or result.get("prediction") or list(result.values())[0]


        # Handle response format properly
        if isinstance(result, list):  # If SageMaker returns a list
            predicted_type = result[0]  # Extract the first item
        elif isinstance(result, dict):  # If it returns a dictionary
            predicted_type = result.get("predicted_instance_type") or result.get("prediction") or list(result.values())[0]
        else:
            predicted_type = str(result)  # Convert to string if it's something unexpected


        # CPU-based recommendation logic
        cpu_utilization = instance_metrics["CPUUtilization"]

        if cpu_utilization < 20:
            recommendation = "Scale down to a smaller instance to reduce costs."
        elif cpu_utilization > 80:
            recommendation = "Scale up to a larger instance type for better performance."
        else:
            recommendation = "CPU utilization is within an optimal range. No action needed."

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "SageMaker invocation successful",
                "predicted_instance_type": predicted_type,
                "recommendation": recommendation,
                "metrics": instance_metrics
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "message": "Failed to invoke SageMaker endpoint"
            })
        }
