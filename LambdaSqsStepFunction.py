import json
import boto3

# Initialize the Step Functions client
sfn_client = boto3.client('stepfunctions')

def lambda_handler(event, context):
    # Extract the SQS message from the event
    # This assumes the Lambda is triggered by an SQS event
    sqs_message = event['Records'][0]['body']

    # Define the Step Function ARN
    step_function_arn = 'arn:aws:states:us-east-1:324037300355:stateMachine:InvokeCloudUtilizationStepFucntion'

    # Create the input for the Step Function execution
    input_data = {
        "message": sqs_message
    }

    # Start the Step Function execution
    response = sfn_client.start_execution(
        stateMachineArn=step_function_arn,
        input=json.dumps(input_data)
    )

    # Log the execution ARN (for debugging purposes)
    print(f"Step Function started with execution ARN: {response['executionArn']}")

    return {
        'statusCode': 200,
        'body': json.dumps('Message processed successfully.')
    }

