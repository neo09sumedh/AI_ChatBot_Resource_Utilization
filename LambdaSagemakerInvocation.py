import json
import boto3
from datetime import datetime, timedelta

# AWS clients
cloudwatch = boto3.client("cloudwatch")
sagemaker_runtime = boto3.client("sagemaker-runtime")
dynamodb = boto3.resource("dynamodb")

# Replace with your actual SageMaker endpoint and DynamoDB table name
ENDPOINT_NAME = "RF-custom-model-2025-04-18-23-40-47"
TABLE_NAME = "CloudCostUtilizationResponse"

# CloudWatch metrics to fetch
METRIC_MAP = {
    "CPUUtilization": {"Namespace": "AWS/EC2", "Metric": "CPUUtilization", "Stat": "Average"},
    "DiskReadOps": {"Namespace": "AWS/EC2", "Metric": "DiskReadOps", "Stat": "Sum"},
    "DiskWriteOps": {"Namespace": "AWS/EC2", "Metric": "DiskWriteOps", "Stat": "Sum"},
    "NetworkIn": {"Namespace": "AWS/EC2", "Metric": "NetworkIn", "Stat": "Sum"},
    "NetworkOut": {"Namespace": "AWS/EC2", "Metric": "NetworkOut", "Stat": "Sum"}
}

def get_instance_metrics(instance_id):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=5)
    metrics_data = {}

    for key, metric in METRIC_MAP.items():
        response = cloudwatch.get_metric_statistics(
            Namespace=metric["Namespace"],
            MetricName=metric["Metric"],
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=[metric["Stat"]],
            Unit="Percent" if key == "CPUUtilization" else "Count"
        )

        datapoints = response.get("Datapoints", [])
        if datapoints:
            latest_value = sorted(datapoints, key=lambda x: x["Timestamp"], reverse=True)[0][metric["Stat"]]
            metrics_data[key] = latest_value
        else:
            metrics_data[key] = 0

    return metrics_data

def lambda_handler(event, context):
    try:
        # Extract required parameters from Step Function event
        request_id = event.get("request_id")
        session_id = event.get("session_id")
        user_query = event.get("user_query")
        instance_id = event.get("instance_id")

        if not all([request_id, session_id, user_query, instance_id]):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required fields in Step Function input."})
            }

        # Fetch metrics
        metrics = get_instance_metrics(instance_id)
        payload = json.dumps([list(metrics.values())])

        # Invoke SageMaker
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload
        )
        result = json.loads(response["Body"].read().decode())

        # Parse prediction
        if isinstance(result, list):
            predicted_type = result[0]
        elif isinstance(result, dict):
            predicted_type = result.get("predicted_instance_type") or result.get("prediction") or list(result.values())[0]
        else:
            predicted_type = str(result)

        # Generate recommendation
        cpu = metrics["CPUUtilization"]
        if cpu < 20:
            recommendation = "Scale down to a smaller instance to reduce costs."
        elif cpu > 80:
            recommendation = "Scale up to a larger instance type for better performance."
        else:
            recommendation = "CPU utilization is within an optimal range. No action needed."

        full_response = f"{recommendation} Suggested type: {predicted_type}"

        # Write to DynamoDB
        table = dynamodb.Table(TABLE_NAME)
        table.put_item(Item={
            "session_id": session_id,
            "request_id": request_id,
            "request": user_query,
            "response": full_response,
            "status": "ready"
        })

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Response written to DynamoDB.",
                "session_id": session_id,
                "request_id": request_id,
                "response": full_response
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "message": "Failed to process request"
            })
        }
