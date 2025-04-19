import boto3
import json
from datetime import datetime, timedelta

# Initialize AWS Clients
ce_client = boto3.client('ce')
cw_client = boto3.client('cloudwatch')
ddb_client = boto3.client('dynamodb')

# Config
DDB_TABLE_NAME = "CloudCostUtilizationResponse"

CW_NAMESPACES = {
    "EC2": "AWS/EC2",
    "RDS": "AWS/RDS",
    "S3": "AWS/S3",
    "Lambda": "AWS/Lambda",
    "EBS": "AWS/EBS",
    "EKS": "AWS/EKS",
    "CloudFront": "AWS/CloudFront",
    "DynamoDB": "AWS/DynamoDB",
    "ElastiCache": "AWS/ElastiCache",
    "Step Functions": "AWS/States",
    "API Gateway": "AWS/ApiGateway",
    "Kinesis": "AWS/Kinesis",
    "SNS": "AWS/SNS",
    "SQS": "AWS/SQS",
    "CloudTrail": "AWS/CloudTrail",
    "Auto Scaling": "AWS/AutoScaling"
}

SERVICE_COST_MAPPING = {
    "EC2": "Amazon Elastic Compute Cloud - Compute",
    "S3": "Amazon Simple Storage Service",
    "Lambda": "AWS Lambda",
    "SQS": "Amazon Simple Queue Service",
    "RDS": "Amazon Relational Database Service",
    "DynamoDB": "Amazon DynamoDB",
    "EBS": "Amazon Elastic Block Store",
    "CloudFront": "Amazon CloudFront",
    "ElastiCache": "Amazon ElastiCache",
    "Step Functions": "AWS Step Functions",
    "API Gateway": "Amazon API Gateway",
    "EKS": "Amazon Elastic Kubernetes Service",
    "Kinesis": "Amazon Kinesis",
    "SNS": "Amazon Simple Notification Service",
    "CloudTrail": "AWS CloudTrail",
    "Auto Scaling": "Auto Scaling"
}

def get_cost_data(service_name, start_date, end_date):
    actual_service_name = SERVICE_COST_MAPPING.get(service_name, service_name)

    response = ce_client.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )

    for result in response['ResultsByTime']:
        for group in result['Groups']:
            if group['Keys'][0] == actual_service_name:
                return float(group['Metrics']['UnblendedCost']['Amount'])
    return 0.0

def get_cloudwatch_metrics(service, start_date, end_date):
    namespace = CW_NAMESPACES.get(service)
    if not namespace:
        return {}

    response = cw_client.list_metrics(Namespace=namespace)
    metrics = response.get('Metrics', [])[:3]  # Limiting for safety

    total_utilization = {}
    start_time = datetime.fromisoformat(start_date + "T00:00:00")
    end_time = datetime.fromisoformat(end_date + "T23:59:59")

    for metric in metrics:
        metric_name = metric['MetricName']
        dimensions = metric.get('Dimensions', [])
        query_id = metric_name.lower().replace(' ', '_')

        if not dimensions:
            continue

        try:
            metric_query = {
                'Id': query_id,
                'MetricStat': {
                    'Metric': {
                        'Namespace': namespace,
                        'MetricName': metric_name,
                        'Dimensions': dimensions
                    },
                    'Period': 86400,
                    'Stat': 'Average'
                },
                'ReturnData': True
            }

            response = cw_client.get_metric_data(
                MetricDataQueries=[metric_query],
                StartTime=start_time,
                EndTime=end_time
            )

            values = response['MetricDataResults'][0].get('Values', [])
            total_utilization[metric_name] = sum(values)

        except Exception as e:
            total_utilization[metric_name] = f"Error: {str(e)}"

    return total_utilization

def store_in_dynamodb(request_id, session_id, request_data, response_data):
    ddb_client.put_item(
        TableName=DDB_TABLE_NAME,
        Item={
            'session_id': {'S': session_id},
            'request_id': {'S': request_id},
            'request': {'S': json.dumps(request_data)},
            'response': {'S': json.dumps(response_data)},
            'status': {'S': 'ready'}
        }
    )

def lambda_handler(event, context):
    session_id = event.get("session_id")
    request_id = event.get("request_id")
    user_query = event.get("user_query")
    intent_name = event.get("intent_name")
    service_name = event.get("service_name")
    start_date = event.get("from_date")
    end_date = event.get("to_date")

    if not start_date or not end_date:
        end_date_dt = datetime.utcnow().date()
        start_date_dt = end_date_dt - timedelta(days=7)
        start_date = str(start_date_dt)
        end_date = str(end_date_dt)

    cost = get_cost_data(service_name, start_date, end_date)
    utilization = get_cloudwatch_metrics(service_name, start_date, end_date)

    utilization_summary = ", ".join([f"{k}: {v}" for k, v in utilization.items()])
    response_text = f"Service: {service_name}, Cost: {cost:.2f} USD, Utilization Summary: {utilization_summary}"

    store_in_dynamodb(request_id, session_id, user_query, response_text)

    return {
        "statusCode": 200,
        "body": response_text
    }
